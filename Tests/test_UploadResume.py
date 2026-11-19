""" Tests of the resumable SFTP upload in RMS.UploadManager.

Files are uploaded under a temporary name which carries a fingerprint of their content and renamed once
complete. An interrupted upload is resumed from a point before the end of the partial upload, after checking
that the partial upload holds the file before that point. The uploads run against a local OpenSSH sftp-server
over a socket pair, so the real paramiko client and the real server are used; link drops are simulated by
failing the socket after a given number of bytes. A failed upload which moved the file forward does not count
as a failed try of the upload manager.
"""

from __future__ import print_function, division, absolute_import

import hashlib
import os
import queue
import select
import shutil
import socket
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest


paramiko = pytest.importorskip("paramiko")

import RMS.UploadManager as um


# Remote directory of the uploads, relative to the root of the test server
DIR_REMOTE = "files"

# Name of the uploaded archive
ARCHIVE_NAME = "XX0001_20261118_010203_123456_detected.tar.bz2"

MIB = 1024*1024


def findSFTPServer():
    """ Find the OpenSSH sftp-server binary.

    Return:
        [str] Path to the binary, or None if it is not installed.
    """

    for path in ("/usr/lib/openssh/sftp-server", "/usr/libexec/openssh/sftp-server", "/usr/libexec/sftp-server",
                 "/usr/lib/ssh/sftp-server", "/usr/libexec/ssh/sftp-server"):
        if os.path.isfile(path):
            return path

    return shutil.which("sftp-server")


SFTP_SERVER = findSFTPServer()


class SocketChannel(object):
    """ A stand-in for a paramiko channel on a local socket, with an optional simulated link drop. """

    def __init__(self, sock, drop_after=None):
        """
        Arguments:
            sock: [socket.socket] Connected socket.

        Keyword arguments:
            drop_after: [int] Raise a socket error once this many bytes have been sent. None by default.
        """

        self.sock = sock
        self.drop_after = drop_after
        self.bytes_sent = 0

    def send(self, data):

        # Simulate a dropped link, the server keeps what it received before
        if (self.drop_after is not None) and (self.bytes_sent + len(data) > self.drop_after):
            self.sock.close()
            raise socket.error("Simulated link drop")

        n = self.sock.send(data)
        self.bytes_sent += n

        return n

    def recv(self, n):
        return self.sock.recv(n)

    def get_name(self):
        return "local sftp-server"

    def recv_ready(self):
        ready, _, _ = select.select([self.sock], [], [], 0)
        return bool(ready)

    def close(self):
        self.sock.close()


class LocalSFTPServer(object):
    """ A local OpenSSH sftp-server serving a directory, one process per connection as with SSH. """

    def __init__(self, root_dir, denied_requests=None):
        """
        Arguments:
            root_dir: [str] Directory where the server starts (the home directory of a station).

        Keyword arguments:
            denied_requests: [list] Requests the server refuses (e.g. posix-rename, fsync). None by default.
        """

        self.root_dir = root_dir
        self.denied_requests = denied_requests
        self.processes = []

    def connect(self, drop_after=None):
        """ Start a server process and connect a paramiko SFTP client to it.

        Keyword arguments:
            drop_after: [int] Simulate a link drop after this many bytes sent by the client. None by default.

        Return:
            [paramiko.SFTPClient] Connected client.
        """

        client_sock, server_sock = socket.socketpair()

        # Run the server on its end of the socket pair
        cmd = [SFTP_SERVER, "-d", self.root_dir]
        if self.denied_requests:
            cmd += ["-P", ",".join(self.denied_requests)]
        process = subprocess.Popen(cmd, stdin=server_sock, stdout=server_sock)
        server_sock.close()
        self.processes.append(process)

        return paramiko.SFTPClient(SocketChannel(client_sock, drop_after=drop_after))

    def close(self):
        """ Wait for all server processes to finish (they exit when the client closes the socket). """

        for process in self.processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


class FakeSSH(object):
    """ The SSH client returned next to the SFTP client, only closed by uploadSFTP. """

    def close(self):
        pass


def makeFile(path, size, seed):
    """ Write a file of pseudo-random bytes.

    Arguments:
        path: [str] Path to the file.
        size: [int] Size in bytes.
        seed: [int] Seed of the content, files with the same seed have the same bytes.

    Return:
        [str] Path to the file.
    """

    data = bytearray()
    block = 0
    while len(data) < size:
        data += (seed*1000003 + block).to_bytes(8, 'little')*8192
        block += 1

    with open(path, 'wb') as f:
        f.write(bytes(data[:size]))

    return path


def readFile(path):
    with open(path, 'rb') as f:
        return f.read()


@pytest.fixture
def station(tmp_path, monkeypatch):
    """ A station with a local data directory and a remote home directory served by a local sftp-server.
        uploadSFTP connects to it through a patched getSSHAndSFTP; set station.drop_after to simulate a link
        drop in the next upload. """

    if SFTP_SERVER is None:
        pytest.skip("OpenSSH sftp-server not installed")

    local_dir = tmp_path/"local"
    remote_root = tmp_path/"remote"
    local_dir.mkdir()
    (remote_root/DIR_REMOTE).mkdir(parents=True)

    server = LocalSFTPServer(str(remote_root))

    st = SimpleNamespace(local_dir=str(local_dir), remote_dir=str(remote_root/DIR_REMOTE), server=server,
                         drop_after=None)

    # Connect uploadSFTP to the local server
    def getSSHAndSFTP(hostname, **kwargs):
        sftp = st.server.connect(drop_after=st.drop_after)
        st.drop_after = None
        return FakeSSH(), sftp

    monkeypatch.setattr(um, "getSSHAndSFTP", getSSHAndSFTP)

    yield st

    st.server.close()


def upload(st, file_name=ARCHIVE_NAME):
    """ Upload one file of the station with uploadSFTP.

    Return:
        (status, resume_offset): [tuple] Upload status and the byte from which the upload started.
    """

    resume_offsets = {}
    status = um.uploadSFTP("localhost", "xx0001", st.local_dir, DIR_REMOTE, [file_name],
                           resume_offsets=resume_offsets)

    # Wait for the server to finish the writes it received before checking the files
    st.server.close()

    return status, resume_offsets.get(file_name)


def remoteFiles(st):
    return sorted(os.listdir(st.remote_dir))


def partPath(st, local_file):
    return os.path.join(st.remote_dir, um.partialUploadName(os.path.basename(local_file), local_file))


def test_fresh_upload(station):
    """ A new file is uploaded and renamed, no partial upload is left behind. """

    local_file = makeFile(os.path.join(station.local_dir, ARCHIVE_NAME), 3*MIB + 123, seed=1)

    status, resume_offset = upload(station)

    assert status
    assert resume_offset == 0
    assert remoteFiles(station) == [ARCHIVE_NAME]
    assert readFile(os.path.join(station.remote_dir, ARCHIVE_NAME)) == readFile(local_file)


def test_resume_after_link_drop(station):
    """ A link drop leaves a partial upload, which the next upload resumes before its end. """

    local_file = makeFile(os.path.join(station.local_dir, ARCHIVE_NAME), 20*MIB + 77, seed=2)

    # The link drops after about 12 MiB
    station.drop_after = 12*MIB
    status, resume_offset = upload(station)

    assert not status
    assert resume_offset == 0
    assert remoteFiles(station) == [os.path.basename(partPath(station, local_file))]
    part_size = os.path.getsize(partPath(station, local_file))
    assert 8*MIB < part_size < 13*MIB

    # The next upload resumes UPLOAD_RESUME_BACK_OFF bytes before the end of the partial upload
    status, resume_offset = upload(station)

    assert status
    assert resume_offset == part_size - um.UPLOAD_RESUME_BACK_OFF
    assert remoteFiles(station) == [ARCHIVE_NAME]
    assert readFile(os.path.join(station.remote_dir, ARCHIVE_NAME)) == readFile(local_file)


def test_repeated_link_drops(station):
    """ Every upload interrupted by a link drop moves the file forward until it is complete. """

    local_file = makeFile(os.path.join(station.local_dir, ARCHIVE_NAME), 30*MIB, seed=3)

    resume_offsets = []
    for _ in range(3):
        station.drop_after = 9*MIB
        status, resume_offset = upload(station)
        assert not status
        resume_offsets.append(resume_offset)

    status, resume_offset = upload(station)
    resume_offsets.append(resume_offset)

    assert status
    assert resume_offsets == sorted(resume_offsets)
    assert resume_offsets[-1] > 10*MIB
    assert readFile(os.path.join(station.remote_dir, ARCHIVE_NAME)) == readFile(local_file)


def test_rebuilt_archive_not_resumed(station):
    """ An archive rebuilt under the same name and size gets a new partial upload, the old one is removed. """

    local_file = makeFile(os.path.join(station.local_dir, ARCHIVE_NAME), 12*MIB, seed=4)
    station.drop_after = 9*MIB
    upload(station)
    old_part = partPath(station, local_file)
    assert os.path.isfile(old_part)

    # Rebuild the archive with other content (a later modification time changes the cached fingerprint)
    time.sleep(0.01)
    makeFile(local_file, 12*MIB, seed=5)

    status, resume_offset = upload(station)

    assert status
    assert resume_offset == 0
    assert remoteFiles(station) == [ARCHIVE_NAME]
    assert readFile(os.path.join(station.remote_dir, ARCHIVE_NAME)) == readFile(local_file)


def test_partial_upload_with_other_content_restarts(station):
    """ A partial upload whose bytes before the resume point differ from the file is written from the start. """

    local_file = makeFile(os.path.join(station.local_dir, ARCHIVE_NAME), 12*MIB, seed=6)
    data = bytearray(readFile(local_file)[:10*MIB])
    data[6*MIB - 100] ^= 0xFF
    with open(partPath(station, local_file), 'wb') as f:
        f.write(bytes(data))

    status, resume_offset = upload(station)

    assert status
    assert resume_offset == 0
    assert readFile(os.path.join(station.remote_dir, ARCHIVE_NAME)) == readFile(local_file)


def test_damaged_end_rewritten(station):
    """ Zeros at the end of a partial upload (e.g. after a crash of the server) are cut off and rewritten. """

    local_file = makeFile(os.path.join(station.local_dir, ARCHIVE_NAME), 16*MIB, seed=7)
    data = readFile(local_file)[:12*MIB - 2*MIB] + bytes(2*MIB)
    with open(partPath(station, local_file), 'wb') as f:
        f.write(data)

    status, resume_offset = upload(station)

    assert status
    assert resume_offset == 12*MIB - um.UPLOAD_RESUME_BACK_OFF
    assert readFile(os.path.join(station.remote_dir, ARCHIVE_NAME)) == readFile(local_file)


def test_partial_upload_larger_than_file_restarts(station):
    """ A partial upload larger than the file can't be part of it and is written from the start. """

    local_file = makeFile(os.path.join(station.local_dir, ARCHIVE_NAME), 6*MIB, seed=8)
    with open(partPath(station, local_file), 'wb') as f:
        f.write(readFile(local_file) + b'x'*1000)

    status, resume_offset = upload(station)

    assert status
    assert resume_offset == 0
    assert readFile(os.path.join(station.remote_dir, ARCHIVE_NAME)) == readFile(local_file)


def test_complete_partial_upload_renamed(station):
    """ A complete partial upload whose rename was lost is checked, its end rewritten and renamed. """

    local_file = makeFile(os.path.join(station.local_dir, ARCHIVE_NAME), 10*MIB, seed=9)
    shutil.copy(local_file, partPath(station, local_file))

    status, resume_offset = upload(station)

    assert status
    assert resume_offset == 10*MIB - um.UPLOAD_RESUME_BACK_OFF
    assert remoteFiles(station) == [ARCHIVE_NAME]
    assert readFile(os.path.join(station.remote_dir, ARCHIVE_NAME)) == readFile(local_file)


def test_small_partial_upload_restarts(station):
    """ A partial upload shorter than the back-off is written again from the start. """

    local_file = makeFile(os.path.join(station.local_dir, ARCHIVE_NAME), 3*MIB, seed=10)
    with open(partPath(station, local_file), 'wb') as f:
        f.write(readFile(local_file)[:2*MIB])

    status, resume_offset = upload(station)

    assert status
    assert resume_offset == 0
    assert readFile(os.path.join(station.remote_dir, ARCHIVE_NAME)) == readFile(local_file)


def test_old_and_plain_partial_uploads_removed(station):
    """ A plain .part left by an earlier version, and partial uploads of other files not written to in
        PARTIAL_UPLOAD_MAX_AGE_DAYS, are removed; a recent partial upload of another file is kept. """

    makeFile(os.path.join(station.local_dir, ARCHIVE_NAME), 1*MIB, seed=11)

    plain_part = os.path.join(station.remote_dir, ARCHIVE_NAME + ".part")
    old_part = os.path.join(station.remote_dir, "XX0001_20261101_010203_123456_detected.tar.bz2.0123456789abcdef.part")
    recent_part = os.path.join(station.remote_dir, "XX0001_20261117_010203_123456_detected.tar.bz2.fedcba9876543210.part")
    other_file = os.path.join(station.remote_dir, "XX0001_20261101_010203_123456_imgdata.tar.bz2")

    for path in (plain_part, old_part, recent_part, other_file):
        with open(path, 'wb') as f:
            f.write(b'x'*1000)

    # Make the partial upload and the other file older than the cleanup age
    old_time = time.time() - (um.PARTIAL_UPLOAD_MAX_AGE_DAYS + 1)*24*3600
    for path in (old_part, other_file):
        os.utime(path, (old_time, old_time))

    status, _ = upload(station)

    assert status
    assert remoteFiles(station) == sorted([ARCHIVE_NAME, os.path.basename(recent_part),
                                           os.path.basename(other_file)])


def test_truncated_final_file_replaced(station):
    """ A truncated file under the final name (left by an upload without a temporary name) is replaced. """

    local_file = makeFile(os.path.join(station.local_dir, ARCHIVE_NAME), 2*MIB, seed=12)
    with open(os.path.join(station.remote_dir, ARCHIVE_NAME), 'wb') as f:
        f.write(readFile(local_file)[:MIB])

    status, _ = upload(station)

    assert status
    assert remoteFiles(station) == [ARCHIVE_NAME]
    assert readFile(os.path.join(station.remote_dir, ARCHIVE_NAME)) == readFile(local_file)


def test_server_without_extensions(station):
    """ A server without posix-rename and fsync still gets the file under its final name, and a resumed upload
        works. """

    station.server.denied_requests = ["posix-rename", "fsync"]

    local_file = makeFile(os.path.join(station.local_dir, ARCHIVE_NAME), 12*MIB, seed=13)
    with open(os.path.join(station.remote_dir, ARCHIVE_NAME), 'wb') as f:
        f.write(b'old')

    station.drop_after = 9*MIB
    status, _ = upload(station)
    assert not status

    status, resume_offset = upload(station)

    assert status
    assert resume_offset > 0
    assert remoteFiles(station) == [ARCHIVE_NAME]
    assert readFile(os.path.join(station.remote_dir, ARCHIVE_NAME)) == readFile(local_file)


def test_empty_file(station):
    """ An empty file is uploaded and renamed. """

    makeFile(os.path.join(station.local_dir, ARCHIVE_NAME), 0, seed=14)

    status, resume_offset = upload(station)

    assert status
    assert resume_offset == 0
    assert remoteFiles(station) == [ARCHIVE_NAME]
    assert os.path.getsize(os.path.join(station.remote_dir, ARCHIVE_NAME)) == 0


def test_same_file_on_server_skipped(station):
    """ A file already on the server with the same size is not uploaded again. """

    local_file = makeFile(os.path.join(station.local_dir, ARCHIVE_NAME), MIB, seed=15)
    shutil.copy(local_file, os.path.join(station.remote_dir, ARCHIVE_NAME))

    status, resume_offset = upload(station)

    assert status
    assert resume_offset is None


def test_fingerprint_cached_per_file_version(tmp_path):
    """ The fingerprint is the SHA-256 of the content and follows a rewrite of the file. """

    path = makeFile(str(tmp_path/"a.bin"), 100000, seed=16)
    fingerprint = um.fileFingerprint(path)
    assert fingerprint == hashlib.sha256(readFile(path)).hexdigest()[:16]
    assert um.fileFingerprint(path) == fingerprint

    time.sleep(0.01)
    makeFile(path, 100000, seed=17)
    assert um.fileFingerprint(path) == hashlib.sha256(readFile(path)).hexdigest()[:16]
    assert um.fileFingerprint(path) != fingerprint


def runUploadData(tmp_path, monkeypatch, attempts, retries):
    """ Run UploadManager.uploadData on one queued file with uploadSFTP replaced by scripted attempts.

    Arguments:
        attempts: [list] (status, resume_offset) of each upload attempt, in order.
        retries: [int] Number of failed tries before uploadData gives up.

    Return:
        (calls, delays): [tuple] Number of upload attempts made and the retry delays (s).
    """

    file_path = makeFile(str(tmp_path/ARCHIVE_NAME), 1000, seed=18)
    calls = []
    delays = []

    def uploadSFTP(hostname, username, dir_local, dir_remote, file_list, resume_offsets=None, **kwargs):
        status, resume_offset = attempts[len(calls)]
        calls.append(file_list[0])
        resume_offsets[file_list[0]] = resume_offset
        return status

    monkeypatch.setattr(um, "uploadSFTP", uploadSFTP)
    monkeypatch.setattr(um, "time", SimpleNamespace(time=time.time, sleep=lambda delay: delays.append(delay)))

    # A stand-in for the upload manager with only what uploadData uses
    file_queue = queue.Queue()
    file_queue.put(file_path)
    manager = SimpleNamespace(
        upload_in_progress=um.AtomicFlag(),
        loadQueue=lambda: None,
        saveQueue=lambda overwrite=False: None,
        file_queue=file_queue,
        file_queue_lock=threading.Lock(),
        config=SimpleNamespace(hostname="localhost", stationID="XX0001", remote_dir=DIR_REMOTE,
                               rsa_private_key=None, host_port=22),
    )

    um.UploadManager.uploadData(manager, retries=retries)

    return len(calls), delays


def test_upload_moving_forward_not_counted(tmp_path, monkeypatch):
    """ Failed uploads which resume further each time do not use up the tries, so the file gets through. """

    attempts = [(False, 0), (False, 5*MIB), (False, 10*MIB), (False, 15*MIB), (True, 20*MIB)]
    calls, delays = runUploadData(tmp_path, monkeypatch, attempts, retries=2)

    assert calls == 5
    assert delays == [15, 15, 15, 15]


def test_stuck_upload_counted(tmp_path, monkeypatch):
    """ Failed uploads which do not resume further count as failed tries, with growing delays. """

    attempts = [(False, 0), (False, 8*MIB), (False, 8*MIB), (False, 4*MIB), (True, 0)]
    calls, delays = runUploadData(tmp_path, monkeypatch, attempts, retries=3)

    assert calls == 4
    assert delays == [15, 15, 30, 60]
