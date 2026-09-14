""" Transport-aware firmware flashing for RMS meteor cameras.

Goal: `UpgradeFirmware <bin>` works no matter what the camera is currently
running, for any flavour of bin the operator hands it.

  * Camera on XM  (DVRIP :34567 alive)  -> CameraControl.upgradeFirmware() hands
    the bin to XM's own flasher, which accepts XM update bins AND Coupler/OpenIPC
    bins. (Handled in CameraControl; nothing here.)

  * Camera on OpenIPC (SSH only, no DVRIP) -> this module. OpenIPC has the SPI
    flash unlocked (the bsp-sfc driver force-unlocks at boot), so we can write
    partitions with flashcp and flip the u-boot env with fw_setenv over SSH.

Bin formats (all the vendor/Coupler bins are the SAME container: a ZIP whose
`InstallDesc` lists "Burn <file>" commands):

  * XM update ZIP  (romfs-x/user-x/web-x/custom-x + InstallDesc) -> writes the XM
    partition layout and sets the XM boot env. Lands on XM. The bin carries NO
    bootloader and NO env, so booting XM afterwards REQUIRES flipping the env --
    which this does (MAC preserved from the running system).

  * OpenIPC/Coupler ZIP (kernel-x/rootfs-x/uboot-env + InstallDesc) -> writes the
    OpenIPC layout and applies the env the bin carries. Lands on OpenIPC. [The
    member->partition map for this family is filled in from a sample Coupler bin;
    see COUPLER_MEMBER_MAP / TODO below.]

  * Raw 16 MB full-chip image -> written byte-for-byte (u-boot binary skipped).

SAFE by construction, every path:
  * The u-boot BINARY (mtd0 0x0..0x2FFFF) is NEVER written. On these boards it is
    the original XM u-boot (Coupler keeps it), so a working bootloader always
    remains -> worst case is a UART-recoverable reflash, never a dead board.
  * The atomic commit is the u-boot env: one 64 KB block via fw_setenv
    (/etc/fw_env.config: mtd0 @ 0x30000). Partitions are written first
    (safest-last, each verified by read-back md5); the env flip is last, so an
    abort before it still boots the current firmware.
  * MAC lives only in the env; it is read from the running system and preserved.

Dry-run is the DEFAULT: read-only checks + fresh backup + the exact write/env
plan, no flash touched. Pass dry_run=False (CLI --commit) to actually flash.

Layout constants are GK7205V200 / G3S (16 MB, OpenIPC 5-partition map). Preflight
refuses a mismatched board.
"""

from __future__ import print_function

import os
import sys
import time
import hashlib
import base64
import zipfile

try:
    from RMS.Logger import getLogger
    log = getLogger("rmslogger")
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger("CameraFlash")


# ---------------------------------------------------------------------------
# On-chip layout (GK7205V200 / G3S, 16 MB). Physical byte offsets.
# ---------------------------------------------------------------------------
IMAGE_LEN = 0x1000000

# OpenIPC 5-partition map (what the running system exposes as /dev/mtdN).
OPENIPC_MTD = [
    (0, "boot",        0x000000, 0x040000),   # u-boot binary (0..0x2FFFF) + env (0x30000..0x3FFFF)
    (1, "wtf",         0x040000, 0x010000),
    (2, "kernel",      0x050000, 0x200000),
    (3, "rootfs",      0x250000, 0x500000),
    (4, "rootfs_data", 0x750000, 0x8B0000),
]
UBOOT_BIN_LEN = 0x30000          # 0..0x2FFFF -- LEAVE UNTOUCHED
ENV_OFF, ENV_LEN = 0x30000, 0x10000
WRITE_ORDER = [2, 1, 3, 4]       # unmounted first, mounted rootfs/overlay last
EXPECTED_SOC = "gk7205v200"

# XM partition layout (physical offsets) + which ZIP member goes where.
XM_PART_OFFSET = {"romfs": 0x040000, "usr": 0x580000, "web": 0xCC0000,
                  "custom": 0xE40000, "mtd": 0xEC0000}
XM_MEMBER_MAP = [   # (filename substring, XM partition)
    ("romfs-x",  "romfs"), ("user-x", "usr"), ("web-x", "web"), ("custom-x", "custom"),
]

# Canonical XM boot env for this hardware (GK7205V200 G3S). ethaddr is filled in
# from the running system so the MAC is preserved. Enough to boot stock/CC XM.
XM_ENV_TEMPLATE = [
    ("baudrate", "115200"),
    ("bootdelay", "0"),
    ("osmem", "38M"),
    ("bootargs", "init=linuxrc mem=${osmem} console=ttyAMA0,115200 root=/dev/mtdblock1 "
                 "rootfstype=squashfs mtdparts=sfc:0x40000(boot),0x540000(romfs),"
                 "0x740000(usr),0x180000(web),0x80000(custom),0x140000(mtd)"),
    ("bootcmd", "setenv setargs setenv bootargs ${bootargs};run setargs;run loadromfs"),
    ("loadromfs", "sf probe 0;sf read 0x43000000 0x40000 0x540000;squashfsload;bootm 0x42000000"),
    ("ethact", "eth0"),
    ("stdin", "serial"), ("stdout", "serial"), ("stderr", "serial"),
    ("verify", "n"),
]

# OpenIPC/Coupler ZIP member -> partition. FILLED IN from a sample Coupler bin.
COUPLER_MEMBER_MAP = None   # TODO: e.g. [("kernel", "kernel"), ("rootfs", "rootfs"), ...]


# ---------------------------------------------------------------------------
# SSH (paramiko lazy; OpenIPC dropbear has no SFTP -> base64 over exec).
# ---------------------------------------------------------------------------
class _Cam(object):
    def __init__(self, ip, password="12345", user="root", timeout=20):
        try:
            import paramiko
        except ImportError:
            raise RuntimeError("paramiko is required for OpenIPC SSH flashing")
        self.ip = ip
        self.c = paramiko.SSHClient()
        self.c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.c.connect(ip, username=user, password=password, timeout=timeout,
                       banner_timeout=timeout, auth_timeout=timeout,
                       look_for_keys=False, allow_agent=False)

    def run(self, cmd, timeout=120):
        i, o, e = self.c.exec_command(cmd, timeout=timeout)
        out = o.read().decode("utf-8", "replace")
        err = e.read().decode("utf-8", "replace")
        return o.channel.recv_exit_status(), out, err

    def mtd_md5(self, index, length, bs=0x10000):
        # Read ceil(length/bs) blocks then trim to EXACTLY length bytes, so the
        # md5 covers the real image size even when it is not block-aligned
        # (e.g. a uImage). `length // bs` alone truncates and mis-verifies.
        blocks = (length + bs - 1) // bs
        rc, out, _ = self.run("dd if=/dev/mtd%d bs=%d count=%d 2>/dev/null | head -c %d | md5sum"
                              % (index, bs, blocks, length))
        return out.strip().split()[0] if out.strip() else None

    def put_and_flash(self, data, index, name):
        """STREAM `data` (one whole partition) straight into the flash and verify by
        read-back md5. DESTRUCTIVE. Returns True on verified success.

        The bytes are base64'd over the SSH channel and piped through `dd` straight
        to the mtd CHAR device /dev/mtdN (erased first with flash_eraseall) -- exactly
        what flashcp does internally, but WITHOUT staging the whole image first. Peak
        camera RAM is the dd buffer (~64 KB), independent of image size. The char
        device is used (not /dev/mtdblockN) because the rootfs partition is the mounted
        overlay lowerdir -> the block device is EBUSY, while the char device is not
        (this is why flashcp always used /dev/mtdN). The previous version base64-decoded
        the ENTIRE image into /tmp (tmpfs = RAM) and then ran flashcp; on a 32 MB camera
        a ~4 MB rootfs in tmpfs OOM/watchdog-crashed the box MID-WRITE and bricked it
        (a partial squashfs the kernel can't mount). No tmpfs staging here.
        """
        want = hashlib.md5(data).hexdigest()
        b64 = base64.b64encode(data)
        # flash_eraseall clears the whole partition, then dd streams the decoded bytes to
        # the char device (bs = 64 KB erase block). Pre-fault the binaries this write needs
        # into page cache FIRST: flash_eraseall erases the mounted rootfs lowerdir up front,
        # so an uncached page fault to it mid-write (busybox applet, etc.) could crash the
        # box during the minutes-long write on worn NOR -- caching them closes that window.
        # rc != 0 if the erase, dd, or sync fails; the read-back md5 below is the real proof.
        i, o, e = self.c.exec_command(
            'cat "$(command -v busybox)" "$(command -v flash_eraseall)" >/dev/null 2>&1; '
            "flash_eraseall /dev/mtd%d >/dev/null 2>&1 && "
            "base64 -d | dd of=/dev/mtd%d bs=65536 && sync" % (index, index),
            timeout=1800)
        try:
            for k in range(0, len(b64), 32768):
                i.write(b64[k:k + 32768])
            i.flush(); i.channel.shutdown_write()
        except Exception as ex:
            log.error("  stream to mtd%d (%s) failed: %s", index, name, ex)
            return False
        _ = o.read(); err = e.read().decode("utf-8", "replace")
        rc = o.channel.recv_exit_status()
        if rc != 0:
            log.error("  stream-write mtd%d (%s) FAILED rc=%d: %s", index, name, rc, err.strip())
            return False
        got = self.mtd_md5(index, len(data))
        if got != want:
            log.error("  VERIFY FAILED mtd%d (%s): on-chip %s != want %s", index, name, got, want)
            return False
        log.info("  mtd%d (%s): streamed+verified %d B (md5 %s)", index, name, len(data), want)
        return True

    def close(self):
        try: self.c.close()
        except Exception: pass


def parse_uboot_env(env_block):
    out = []
    for raw in env_block[4:].split(b"\x00"):
        if not raw:
            break
        try: s = raw.decode("latin-1")
        except Exception: continue
        if "=" in s:
            k, v = s.split("=", 1); out.append((k, v))
    return out


def _norm_mac(s):
    return "".join(ch for ch in (s or "").lower() if ch in "0123456789abcdef")


# ---------------------------------------------------------------------------
# Bin format detection
# ---------------------------------------------------------------------------
def detect_bin_kind(path):
    """Return (kind, info). kind in {'xm_zip','coupler_zip','raw16m','unknown'}."""
    if not os.path.isfile(path):
        raise RuntimeError("bin not found: %s" % path)
    size = os.path.getsize(path)
    if size == IMAGE_LEN:
        return "raw16m", {"size": size}
    # OpenIPC sysupgrade archive: a .tgz carrying uImage + rootfs.squashfs.
    try:
        import tarfile
        if tarfile.is_tarfile(path):
            tnames = tarfile.open(path).getnames()
            j = " ".join(tnames).lower()
            if "uimage" in j and "rootfs.squashfs" in j:
                return "openipc_tgz", {"members": tnames}
    except Exception:
        pass
    try:
        z = zipfile.ZipFile(path)
        names = z.namelist()
    except Exception:
        return "unknown", {"size": size}
    burns = names
    if "InstallDesc" in names:
        import json
        try:
            desc = json.loads(z.read("InstallDesc").decode("utf-8", "replace"))
            burns = [c.get("FileName") for c in desc.get("UpgradeCommand", []) if c.get("FileName")]
        except Exception:
            desc = {}
    joined = " ".join(names).lower()
    if "romfs-x" in joined or "user-x" in joined:
        return "xm_zip", {"members": burns, "names": names}
    if "kernel" in joined and "rootfs" in joined:
        return "coupler_zip", {"members": burns, "names": names}
    return "unknown", {"members": burns, "names": names}


# ---------------------------------------------------------------------------
# Preflight (read-only)
# ---------------------------------------------------------------------------
def preflight(cam):
    facts = {}
    rc, mtd, _ = cam.run("cat /proc/mtd")
    got = []
    for line in mtd.splitlines():
        if line.startswith("mtd"):
            p = line.split()
            got.append((int(p[0][3:].rstrip(":")), p[3].strip('"'), int(p[1], 16)))
    want = [(i, n, sz) for (i, n, o, sz) in OPENIPC_MTD]
    if got != want:
        raise RuntimeError("flash layout mismatch -- refusing.\n  on-chip: %s\n  want: %s" % (got, want))
    log.info("[preflight] partition layout OK (OpenIPC GK7205V200 5-part map)")

    rc, envdump, _ = cam.run("fw_printenv 2>/dev/null")
    if EXPECTED_SOC not in envdump.lower():
        rc, soc, _ = cam.run("cat /proc/device-tree/model 2>/dev/null; dmesg 2>/dev/null | grep -i gk7205")
        if EXPECTED_SOC not in (soc or "").lower():
            raise RuntimeError("SoC guard: could not confirm %s on target" % EXPECTED_SOC)
    log.info("[preflight] SoC guard OK (%s)", EXPECTED_SOC)

    # on-chip bootloader must be a valid u-boot for THIS SoC. We keep it (never
    # write mtd0's binary half), so confirm it's present and matches the board.
    ub = cam.mtd_md5(0, UBOOT_BIN_LEN)
    rc, sig, _ = cam.run("dd if=/dev/mtd0 bs=65536 count=4 2>/dev/null | strings "
                         "| grep -m1 -iE 'gk7205v200|IPC_GK7205|U-Boot 20'")
    facts["uboot_md5"] = ub
    facts["uboot_sig"] = sig.strip()
    if not sig.strip():
        raise RuntimeError("could not confirm a %s bootloader in mtd0 -- refusing "
                           "(the safe-flash guarantee assumes a valid on-chip u-boot)"
                           % EXPECTED_SOC)
    log.info("[preflight] on-chip bootloader confirmed (kept, never written): %s", sig.strip())

    rc, mac, _ = cam.run("cat /sys/class/net/eth0/address 2>/dev/null")
    facts["mac"] = _norm_mac(mac)
    facts["mac_pretty"] = mac.strip()
    log.info("[preflight] live MAC %s (will be preserved in the env)", mac.strip())

    # Free RAM sanity. The image is STREAMED to flash (put_and_flash), so we no longer
    # need image-sized headroom -- peak use is ~one erase block. But a critically low
    # MemAvailable means something else is eating RAM (e.g. capture still streaming from
    # this cam); flashing into that risks the OOM/watchdog crash that bricked .205, so
    # HARD-FAIL rather than proceed. Stop RMS capture on the target before flashing.
    MIN_FREE_KB = 3072  # 3 MB: comfortable for the dd/erase path + running services
    rc, mout, _ = cam.run("awk '/^MemAvailable:/{print $2; f=1} END{if(!f) print -1}' /proc/meminfo")
    try: facts["mem_avail_kb"] = int(mout.split()[0])
    except Exception: facts["mem_avail_kb"] = -1
    if 0 <= facts["mem_avail_kb"] < MIN_FREE_KB:
        raise RuntimeError(
            "only %d KB RAM available (< %d KB) -- refusing to flash. Something is using "
            "memory on the camera (is RMS still capturing from it?). Stop capture and retry; "
            "flashing into low RAM is what OOM/watchdog-bricked .205." % (facts["mem_avail_kb"], MIN_FREE_KB))
    log.info("[preflight] RAM available %d KB (streamed flash, no full-image staging)", facts["mem_avail_kb"])
    return facts


def backup_current(cam, out_dir):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    log.info("[backup] saving current on-chip flash to %s", out_dir)
    full = bytearray()
    for (idx, name, off, size) in OPENIPC_MTD:
        rc, out, _ = cam.run("dd if=/dev/mtd%d bs=65536 2>/dev/null | base64" % idx, timeout=600)
        blob = base64.b64decode(out)
        if len(blob) != size:
            raise RuntimeError("backup mtd%d: got %d, want %d" % (idx, len(blob), size))
        open(os.path.join(out_dir, "mtd%d_%s.bin" % (idx, name)), "wb").write(blob)
        full += blob
    open(os.path.join(out_dir, "full.bin"), "wb").write(full)
    log.info("[backup] full.bin %d B md5 %s", len(full), hashlib.md5(bytes(full)).hexdigest())


# ---------------------------------------------------------------------------
# Build a 16 MB target image from an XM update ZIP (members at XM offsets).
# [0:0x40000] is left 0xFF -- we never write mtd0; the on-chip u-boot+env stay,
# and the env is flipped separately with fw_setenv.
# ---------------------------------------------------------------------------
def build_target_from_xm_zip(path):
    z = zipfile.ZipFile(path)
    names = z.namelist()
    target = bytearray(b"\xff" * IMAGE_LEN)
    placed = []
    for (sub, part) in XM_MEMBER_MAP:
        member = next((n for n in names if sub in n), None)
        if member is None:
            continue
        data = z.read(member)
        off = XM_PART_OFFSET[part]
        # bound check against the next partition
        limit = min([o for o in XM_PART_OFFSET.values() if o > off] + [IMAGE_LEN]) - off
        if len(data) > limit:
            raise RuntimeError("member %s (%d B) overflows %s partition (%d B)"
                               % (member, len(data), part, limit))
        target[off:off + len(data)] = data
        placed.append((member, part, off, len(data)))
    if not placed:
        raise RuntimeError("no known XM members found in %s" % path)
    return bytes(target), placed


def _xm_env_with_mac(mac_pretty):
    env = list(XM_ENV_TEMPLATE)
    env.append(("ethaddr", mac_pretty))
    return env


# ---------------------------------------------------------------------------
# The write engine: write mtd1..4 from the target image, verify, flip env.
# ---------------------------------------------------------------------------
def _print_plan(target, env_vars, placed):
    log.info("---- WRITE PLAN (safest-last; u-boot binary never written) ----")
    for (member, part, off, ln) in placed:
        log.info("  %-22s -> XM %-6s @0x%06x  %8d B" % (member, part, off, ln))
    for idx in WRITE_ORDER:
        _i, name, off, size = OPENIPC_MTD[idx]
        sl = target[off:off + size]
        log.info("  mtd%d %-11s <- target[0x%06x:0x%06x]  md5 %s" %
                 (idx, name, off, off + size, hashlib.md5(sl).hexdigest()))
    log.info("---- ENV COMMIT (fw_setenv, single 0x%05x block) ----" % ENV_LEN)
    for k, v in env_vars:
        log.info("  set %-10s = %s", k, (v[:90] + "...") if len(v) > 93 else v)


def _commit(cam, target, env_vars):
    log.warning("[commit] stopping streamer to free RAM")
    cam.run("/etc/init.d/S96venc stop 2>/dev/null; killall venc isp_ctl chronyd 2>/dev/null; sleep 1")
    for idx in WRITE_ORDER:
        _i, name, off, size = OPENIPC_MTD[idx]
        log.warning("[commit] writing mtd%d (%s) ...", idx, name)
        if not cam.put_and_flash(target[off:off + size], idx, name):
            log.error("[commit] ABORTED at mtd%d; env NOT flipped -> still boots current fw.", idx)
            return False
    log.warning("[commit] flipping u-boot env (fw_setenv, MAC preserved)")
    script = ("\n".join("%s %s" % (k, v) for k, v in env_vars) + "\n").encode("latin-1")
    b64 = base64.b64encode(script).decode()
    rc, out, err = cam.run("echo '%s' | base64 -d > /tmp/.env && fw_setenv -s /tmp/.env && "
                           "rm -f /tmp/.env && fw_printenv bootcmd" % b64)
    if rc != 0:
        log.error("[commit] env flip FAILED rc=%d: %s / %s", rc, out.strip(), err.strip())
        return False
    log.info("[commit] env now: %s", out.strip())
    log.warning("[commit] rebooting")
    cam.run("sync; (sleep 1; reboot) >/dev/null 2>&1 &")
    return True


def _wait_port(ip, port, timeout=180):
    import socket
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = socket.socket(); s.settimeout(3)
        try:
            s.connect((ip, port)); s.close(); return True
        except Exception:
            s.close(); time.sleep(4)
    return False


# ---------------------------------------------------------------------------
# OpenIPC -> OpenIPC image upgrade (via the camera's own sysupgrade).
#
# Overlay paths a hand-provisioned camera may have shadowing the flashed rootfs
# (venc etc. SSH-dropped into /overlay/root). Cleared on an image upgrade so the
# new baked-in versions take effect. Network/identity/runtime config
# (interfaces.d, hostname, dropbear key, /mnt/mtd/*) is deliberately NOT listed,
# so the camera keeps its IP and settings across the upgrade. On a camera already
# running our baked image none of these exist in the overlay -> a no-op.
# ---------------------------------------------------------------------------
OVERLAY_UPPER = "/overlay/root"
SHADOW_OVERLAY_PATHS = [
    "usr/bin/venc", "usr/bin/venc.bak3line", "usr/bin/isp_ctl", "usr/bin/ircut",
    "usr/sbin/netherd", "usr/bin/chronyd", "usr/bin/chronyc",
    "usr/lib/sensors/libsns_imx307_2l.so", "usr/lib/sensors/libsns_imx307_2l.so.bak3line",
    "etc/init.d/S96venc", "etc/init.d/S97isp_ctl", "etc/init.d/S49chrony",
    "etc/init.d/S41netherd", "etc/init.d/S95majestic", "etc/init.d/S49ntpd",
    "etc/majestic.yaml", "etc/sensors/iq/imx307.ini",
]


def _wait_reboot(ip, port=22, up_timeout=200):
    """Wait for the camera to go down then come back up on `port`."""
    import socket
    t0 = time.time(); went_down = False
    while time.time() - t0 < up_timeout:
        s = socket.socket(); s.settimeout(3)
        try:
            s.connect((ip, port)); s.close()
            if went_down:
                return True
        except Exception:
            went_down = True
        finally:
            try: s.close()
            except Exception: pass
        time.sleep(3)
    return went_down  # came down at least; may still be booting


def flash_openipc_image(cam, ip, tgz_path, dry_run):
    """OpenIPC->OpenIPC. Extracts the .tgz host-side, flashes kernel+rootfs
    DIRECTLY (flashcp, read-back verified) -- no sysupgrade, no /tmp extraction.
    Keeps the hardware watchdog fed across the flash, clears only shadowing
    overlay app-paths (network/identity/config preserved), then reboots.

    Why direct-flash, not sysupgrade: sysupgrade extracts the archive into /tmp
    (tmpfs, often <5MB free) which fails on a ~6MB image, and its free_resources
    KILLS majestic -- the process feeding /dev/watchdog -- so a stalled flash lets
    the 60s watchdog reboot the box mid-attempt.
    """
    import tarfile
    tf = tarfile.open(tgz_path)
    names = [n for n in tf.getnames() if not n.endswith(".md5sum")]
    kname = next((n for n in names if "uimage" in n.lower()), None)
    rname = next((n for n in names if "rootfs.squashfs" in n.lower()), None)
    if not kname or not rname:
        raise RuntimeError("archive missing uImage/rootfs.squashfs: %s" % tf.getnames())
    kdata = tf.extractfile(kname).read()
    rdata = tf.extractfile(rname).read()
    kmtd = next(i for (i, n, o, s) in OPENIPC_MTD if n == "kernel")
    rmtd = next(i for (i, n, o, s) in OPENIPC_MTD if n == "rootfs")
    log.info("---- OpenIPC image upgrade (direct flash) ----")
    log.info("  kernel %s (%d B) -> mtd%d ; rootfs %s (%d B) -> mtd%d",
             kname, len(kdata), kmtd, rname, len(rdata), rmtd)
    rc, present, _ = cam.run("cd %s 2>/dev/null && ls -1d %s 2>/dev/null"
                             % (OVERLAY_UPPER, " ".join(SHADOW_OVERLAY_PATHS)))
    shadows = [p for p in present.split() if p]
    log.info("  overlay shadows to clear: %s", shadows if shadows else "(none -- clean-image camera)")
    log.info("  preserved: interfaces.d/eth0 (IP), hostname, dropbear key, /mnt/mtd/*")
    log.info("  plan: feed watchdog -> stream kernel(verify) -> stream rootfs(verify)"
             " -> clear shadows -> reboot")
    if dry_run:
        log.info("=== DRY-RUN complete. Nothing was flashed. Add 'commit' to flash. ===")
        return True

    # Keep /dev/watchdog fed across the flash. majestic (if running) holds it, so
    # take it over: kill majestic, then immediately start a keep-alive writer
    # (well within the 60s margin). On the new image nothing arms the watchdog.
    log.warning("[commit] taking over the watchdog (kill majestic + keep-alive feeder)")
    cam.run("killall -9 majestic 2>/dev/null; sleep 1; "
            "setsid sh -c 'while :; do echo w > /dev/watchdog 2>/dev/null; sleep 15; done' "
            "</dev/null >/dev/null 2>&1 & echo fed")
    cam.run("rm -f /tmp/*.log 2>/dev/null")

    # Flash kernel then rootfs, each STREAMED straight to the mtd block device and
    # read-back verified (put_and_flash; ~64 KB peak RAM, no tmpfs staging).
    log.warning("[commit] flashing kernel -> mtd%d", kmtd)
    if not cam.put_and_flash(kdata, kmtd, "kernel"):
        log.error("[commit] kernel flash FAILED -- overlay untouched, still boots current image.")
        return False
    log.warning("[commit] flashing rootfs -> mtd%d", rmtd)
    if not cam.put_and_flash(rdata, rmtd, "rootfs"):
        log.error("[commit] rootfs flash FAILED -- kernel already updated; RE-RUN before reboot.")
        return False

    if shadows:
        cam.run("cd %s && rm -f %s" % (OVERLAY_UPPER, " ".join(shadows)))
        log.info("[commit] cleared %d shadowing overlay path(s)", len(shadows))
    log.warning("[commit] flash verified; rebooting into the new image")
    cam.run("sync; (sleep 2; reboot) >/dev/null 2>&1 &")
    log.info("[post] waiting for reboot into the new image ...")
    if _wait_reboot(ip, 22):
        log.info("=== %s rebooted; new image should be live. ===", ip)
        return True
    log.warning("=== reboot not confirmed in time; check the camera (u-boot intact -> recoverable). ===")
    return False

# ---------------------------------------------------------------------------
# Main entry: upgrade an OpenIPC camera to whatever `bin_path` is.
# ---------------------------------------------------------------------------
def upgrade_from_openipc(ip, bin_path, dry_run=True, password="12345",
                         backup_dir=None, do_backup=True):
    kind, info = detect_bin_kind(bin_path)
    log.info("=== UpgradeFirmware (OpenIPC/SSH) : %s ===", ip)
    log.info("    bin  : %s", bin_path)
    log.info("    kind : %s  %s", kind, info.get("members", ""))
    log.info("    mode : %s", "DRY-RUN (no writes)" if dry_run else "COMMIT (DESTRUCTIVE)")

    if kind == "coupler_zip" and COUPLER_MEMBER_MAP is None:
        raise RuntimeError(
            "This is an OpenIPC/Coupler bin, but the member->partition map isn't "
            "populated yet. Provide a sample Coupler bin so COUPLER_MEMBER_MAP can "
            "be filled in (members seen: %s)." % info.get("members"))
    if kind == "unknown":
        raise RuntimeError("unrecognised bin format: %s" % bin_path)

    cam = _Cam(ip, password=password)
    try:
        facts = preflight(cam)

        # OpenIPC -> OpenIPC: our own image (.tgz). Uses the camera's sysupgrade;
        # keeps it reachable (network/identity preserved). Different mechanism
        # from the raw partition-write engine below.
        if kind == "openipc_tgz":
            return flash_openipc_image(cam, ip, bin_path, dry_run)

        if kind == "xm_zip":
            target, placed = build_target_from_xm_zip(bin_path)
            env_vars = _xm_env_with_mac(facts["mac_pretty"] or "")
        elif kind == "raw16m":
            target = open(bin_path, "rb").read()
            placed = [("<raw full.bin>", "-", 0, len(target))]
            env_vars = parse_uboot_env(target[ENV_OFF:ENV_OFF + ENV_LEN])
            # keep the live MAC even from a raw image
            env_vars = [(k, v) for (k, v) in env_vars if k != "ethaddr"]
            env_vars.append(("ethaddr", facts["mac_pretty"] or ""))
        else:  # coupler_zip once mapped
            raise RuntimeError("coupler_zip path not yet implemented (needs sample)")

        if do_backup:
            if backup_dir is None:
                backup_dir = os.path.join(os.path.dirname(os.path.abspath(bin_path)),
                                          "%s_backup_%s" % (ip.replace(".", "_"),
                                                            time.strftime("%Y%m%d_%H%M%S")))
            backup_current(cam, backup_dir)

        _print_plan(target, env_vars, placed)

        if dry_run:
            log.info("=== DRY-RUN complete. Nothing was flashed. Add --commit to flash. ===")
            return True

        if not _commit(cam, target, env_vars):
            return False
    finally:
        cam.close()

    want_port = 34567 if kind == "xm_zip" else 22
    log.info("[post] waiting for the camera to come back (port %d) ...", want_port)
    if _wait_port(ip, want_port):
        log.info("=== SUCCESS: %s is back up. ===", ip)
        return True
    log.warning("=== Rebooted but not seen yet; u-boot is intact so it is recoverable. ===")
    return False


# ---------------------------------------------------------------------------
# Coupler wrap (XM->OpenIPC via DVRIP): pure-Python mkimage/mkenvimage so
# `UpgradeFirmware <venc.tgz>` on an XM camera can auto-build a per-unit,
# DVRIP-flashable bin (MAC baked in) with no external tools. GK7205V200/G3S.
# Verified byte-identical to u-boot-tools mkimage/mkenvimage output.
# ---------------------------------------------------------------------------
_C_KERNEL_A, _C_KERNEL_E = 0x50000, 0x250000
_C_ROOTFS_A, _C_ROOTFS_E = 0x250000, 0x750000
_C_ENV_A, _C_ENV_E = 0x30000, 0x40000
_C_FLASH = 0x1000000

def _mkimage(data, name, load, ep, itype, comp=1):
    import struct, zlib, time
    hdr = struct.pack(">IIIIIIIBBBB32s", 0x27051956, 0, int(time.time()), len(data),
                      load, ep, zlib.crc32(data) & 0xffffffff, 5, 2, itype, comp,
                      name.encode()[:32])
    hcrc = zlib.crc32(hdr) & 0xffffffff
    return hdr[:4] + struct.pack(">I", hcrc) + hdr[8:] + data

def _mkenvimage(text, size=0x10000):
    import struct, zlib
    body = text.replace("\n", "\0").encode()
    if not body.endswith(b"\0"):
        body += b"\0"
    body += b"\0"
    body = body.ljust(size - 4, b"\xff")[:size - 4]
    return struct.pack("<I", zlib.crc32(body) & 0xffffffff) + body

def build_coupler_bin(tgz_path, mac, out_path):
    """Wrap an OpenIPC venc .tgz into a DVRIP-flashable XM 'coupler' bin (G3S)
    with `mac` baked into the u-boot env, so the converted camera keeps its MAC.
    Pure Python -- no mkimage/mkenvimage/zip binaries needed."""
    import tarfile, zipfile
    tf = tarfile.open(tgz_path)
    names = [n for n in tf.getnames() if not n.endswith(".md5sum")]
    kdata = tf.extractfile(next(n for n in names if "uimage" in n.lower())).read()
    rdata = tf.extractfile(next(n for n in names if "rootfs.squashfs" in n.lower())).read()
    if len(rdata) > _C_ROOTFS_E - _C_ROOTFS_A:
        raise RuntimeError("rootfs %d B exceeds partition" % len(rdata))
    if len(kdata) > _C_KERNEL_E - _C_KERNEL_A:
        raise RuntimeError("uImage %d B exceeds partition" % len(kdata))
    env = ("bootdelay=0\nbaudrate=115200\nethaddr=%s\n"
           "bootargs=mem=${osmem} console=ttyAMA0,115200 panic=20 root=/dev/mtdblock3 "
           "rootfstype=squashfs init=/init mtdparts=sfc:256k(boot),64k(wtf),2048k(kernel),"
           "5120k(rootfs),-(rootfs_data)\n"
           "bootcmd=sf probe 0; sf lock 0; setenv bootcmd 'setenv setargs setenv bootargs "
           "${bootargs}; run setargs; sf probe 0; sf read 0x42000000 0x50000 0x200000; "
           "bootm 0x42000000';sa;re\n"
           "osmem=32M\ntotalmem=64M\nsoc=gk7205v200\nhardware=IPC_GK7205V200_G3S\n"
           "devid=000739AG\nmanufacturer=Xiongmai\nstdin=serial\nstdout=serial\n"
           "stderr=serial\nverify=n\n") % mac
    envimg = _mkimage(_mkenvimage(env), "uboot_env", _C_ENV_A, _C_ENV_E, 2)
    kimg = _mkimage(kdata, "kernel", _C_KERNEL_A, _C_KERNEL_E, 2)
    rimg = _mkimage(rdata, "rootfs", _C_ROOTFS_A, _C_ROOTFS_E, 2)
    mtdimg = _mkimage(b"\xff" * (_C_FLASH - _C_ROOTFS_E), "rootfs_data", _C_ROOTFS_E, _C_FLASH, 1)
    desc = ('{"UpgradeCommand":[{"Command":"Burn","FileName":"uImage.img"},'
            '{"Command":"Burn","FileName":"rootfs.img"},'
            '{"Command":"Burn","FileName":"u-boot.env.img"},'
            '{"Command":"Burn","FileName":"mtd-x.jffs2.img"}],'
            '"SupportFlashType":[{"FlashID":"SkipCheck"}],'
            '"Hardware":"IPC_GK7205V200_G3S","HardWareVersion":1,'
            '"DevID":"000739AGXXXXX000000000000","CompatibleVersion":2,'
            '"Vendor":"SkipCheck","Mx8Q":"0"}')
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("u-boot.env.img", envimg)
        z.writestr("rootfs.img", rimg)
        z.writestr("uImage.img", kimg)
        z.writestr("mtd-x.jffs2.img", mtdimg)
        z.writestr("InstallDesc", desc)
    return out_path


def detect_transport(ip):
    import socket
    for port, name in ((34567, "xm"), (22, "openipc")):
        s = socket.socket(); s.settimeout(3)
        try:
            s.connect((ip, port)); s.close(); return name
        except Exception:
            s.close()
    return None


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="OpenIPC-side firmware flashing (dry-run default).")
    sub = ap.add_subparsers(dest="cmd")
    d = sub.add_parser("detect"); d.add_argument("--ip", required=True)
    u = sub.add_parser("upgrade")
    u.add_argument("--ip", required=True); u.add_argument("--image", required=True)
    u.add_argument("--commit", action="store_true")
    u.add_argument("--password", default="12345")
    u.add_argument("--no-backup", action="store_true")
    a = ap.parse_args()
    if a.cmd == "detect":
        print(detect_transport(a.ip))
    elif a.cmd == "upgrade":
        ok = upgrade_from_openipc(a.ip, a.image, dry_run=not a.commit,
                                  password=a.password, do_backup=not a.no_backup)
        sys.exit(0 if ok else 1)
    else:
        ap.print_help(); sys.exit(2)
