""" Tests for running the flux preparation stage in its own process.

The stage is split out of the capture parent so its memory is returned to the kernel on
exit (see RMS.RunFluxStage). These tests pin the two things that split can silently break:
the parent's in-memory config/mask/platepar must reach the stage unchanged, and a stage
that fails, times out or cannot be started must never propagate into the capture loop.
"""

from __future__ import absolute_import, division, print_function

import logging
import os
import subprocess
import sys
import threading
import types

import pytest

from RMS.Pickling import savePickle
from RMS.RunFluxStage import STATE_FILE_NAME, runFluxStageFromState
from RMS.SlotGate import GATE_WORK_START_MARKER


class DummyPlatepar(object):
    def __init__(self, vignetting_coeff=0.001):
        self.vignetting_coeff = vignetting_coeff


class DummyConfig(object):
    def __init__(self, station_id="US005X"):
        self.stationID = station_id
        self.use_flat = True


class DummyMask(object):
    def __init__(self, tag="mask"):
        self.tag = tag


@pytest.fixture
def stubFlux(monkeypatch):
    """ Put a stub Utils.Flux in place and record what prepareFluxFiles is called with. """

    calls = []

    module = types.ModuleType("Utils.Flux")

    def prepareFluxFiles(config, dir_path, ftpdetectinfo_path, mask=None, platepar=None,
        allow_model_fit=True):
        calls.append(dict(config=config, dir_path=dir_path,
                          ftpdetectinfo_path=ftpdetectinfo_path, mask=mask,
                          platepar=platepar, allow_model_fit=allow_model_fit))

    module.prepareFluxFiles = prepareFluxFiles

    monkeypatch.setitem(sys.modules, "Utils.Flux", module)

    return calls


### The handoff ###

def test_handoffCarriesTheParentsInMemoryObjects(tmpdir, stubFlux):
    """ The stage must use the objects the parent held, not the copies on disk.

    processNight zeroes platepar.vignetting_coeff in memory when a flat is in use, after
    the platepar has already been written out. A stage that reloaded from disk would score
    the night with a different vignetting correction and nothing would report it.
    """

    night_dir = str(tmpdir)

    platepar = DummyPlatepar(vignetting_coeff=0.0)
    mask = DummyMask("in-memory")
    config = DummyConfig()

    savePickle(dict(config=config, mask=mask, platepar=platepar, allow_model_fit=False),
               night_dir, STATE_FILE_NAME)

    runFluxStageFromState(night_dir, os.path.join(night_dir, "FTPdetectinfo.txt"))

    assert len(stubFlux) == 1
    call = stubFlux[0]

    assert call["platepar"].vignetting_coeff == 0.0
    assert call["mask"].tag == "in-memory"
    assert call["config"].stationID == "US005X"
    assert call["allow_model_fit"] is False
    assert call["dir_path"] == night_dir


def test_handoffDefaultsAllowModelFitToTrue(tmpdir, stubFlux):
    """ A state written without the flag must not silently disable the model fit. """

    night_dir = str(tmpdir)

    savePickle(dict(config=DummyConfig(), mask=None, platepar=None), night_dir,
               STATE_FILE_NAME)

    runFluxStageFromState(night_dir, os.path.join(night_dir, "FTPdetectinfo.txt"))

    assert stubFlux[0]["allow_model_fit"] is True


def test_missingStateIsReportedNotSilentlySkipped(tmpdir, stubFlux):
    """ A stage that cannot read its handoff must fail loudly rather than run on defaults.

    Running with mask=None/platepar=None would look like success while quietly scoring the
    night against whatever happens to be on disk.
    """

    with pytest.raises(IOError):
        runFluxStageFromState(str(tmpdir), "FTPdetectinfo.txt")

    assert stubFlux == []


def test_corruptStateIsReported(tmpdir, stubFlux):
    """ loadPickle returns None on a corrupt file - that must not be treated as state. """

    night_dir = str(tmpdir)

    with open(os.path.join(night_dir, STATE_FILE_NAME), "wb") as f:
        f.write(b"not a pickle")

    with pytest.raises(IOError):
        runFluxStageFromState(night_dir, "FTPdetectinfo.txt")

    assert stubFlux == []


### The module is invocable as an entry point ###

def test_moduleRunsAsAnEntryPoint():
    """ The parent invokes the stage as "python -m RMS.RunFluxStage" - prove that resolves. """

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    proc = subprocess.Popen([sys.executable, "-m", "RMS.RunFluxStage", "--help"],
                            cwd=repo_root, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, universal_newlines=True)
    out, _ = proc.communicate()

    assert proc.returncode == 0, out
    assert "FTPDETECTINFO_PATH" in out


### The parent side ###

reprocess = pytest.importorskip("RMS.Reprocess",
    reason="the full RMS import chain is not available in this environment")


class FakePopen(object):
    """ Stand-in for the stage process, so the parent's handling can be tested directly. """

    def __init__(self, returncode=0, output="", raise_timeout=False):
        self._returncode = returncode
        self.raise_timeout = raise_timeout
        self.killed = False
        self.cmd = None
        self.cwd = None

        import io
        self.stdout = io.StringIO(output)

    def wait(self, timeout=None):
        if self.raise_timeout and not self.killed:
            raise subprocess.TimeoutExpired("stage", timeout)
        return self._returncode

    def kill(self):
        self.killed = True


@pytest.fixture
def capturedLog():
    """ Collect records emitted on the RMS logger. """

    records = []

    class Collector(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = Collector()
    logger = logging.getLogger("rmslogger")
    logger.addHandler(handler)

    yield records

    logger.removeHandler(handler)


def installFakePopen(monkeypatch, fake):
    """ Route the parent's Popen call into the given fake, recording how it was invoked. """

    def fakePopen(cmd, cwd=None, **kwargs):
        fake.cmd = cmd
        fake.cwd = cwd
        return fake

    monkeypatch.setattr(reprocess.subprocess, "Popen", fakePopen)

    return fake


def stateDirOf(cmd):
    """ Pull the --state-dir the parent handed to the stage. """

    return cmd[cmd.index("--state-dir") + 1]


def test_stageSuccessWritesStateAndCleansItUp(tmpdir, monkeypatch):
    """ The handoff pickle is written for the child and removed once the stage is done. """

    night_dir = str(tmpdir)
    fake = installFakePopen(monkeypatch, FakePopen(returncode=0))

    ok = reprocess.runFluxStage(DummyConfig(), night_dir,
                                os.path.join(night_dir, "FTPdetectinfo.txt"),
                                mask=DummyMask(), platepar=DummyPlatepar())

    assert ok is True
    assert not os.path.exists(stateDirOf(fake.cmd))

    # Invoked as a module from the repository root, so "-m RMS.RunFluxStage" resolves
    assert fake.cmd[1:3] == ["-m", "RMS.RunFluxStage"]
    assert fake.cmd[3] == night_dir
    assert os.path.isfile(os.path.join(fake.cwd, "RMS", "RunFluxStage.py"))


def test_handoffStaysOutOfTheNightDirectory(tmpdir, monkeypatch):
    """ The pickle holds the config, and the night directory is tarred and uploaded.

    Keeping the handoff outside it means no future change to the archive's file filters can
    sweep a pickled config into an upload.
    """

    night_dir = str(tmpdir)
    captured = {}

    class Recorder(FakePopen):
        def wait(self, timeout=None):
            # The state must still be on disk while the stage is running
            captured["files"] = sorted(os.listdir(night_dir))
            captured["state_present"] = os.path.isfile(
                os.path.join(stateDirOf(self.cmd), STATE_FILE_NAME))
            return 0

    fake = installFakePopen(monkeypatch, Recorder(returncode=0))

    reprocess.runFluxStage(DummyConfig(), night_dir, "FTPdetectinfo.txt")

    assert captured["state_present"] is True
    assert captured["files"] == []
    assert not os.path.commonpath([stateDirOf(fake.cmd), night_dir]) == night_dir


def test_stageOutputIsForwardedIntoTheStationLog(tmpdir, monkeypatch, capturedLog):
    """ The child logs to stdout; the parent must relay it so the night keeps one log. """

    night_dir = str(tmpdir)
    installFakePopen(monkeypatch, FakePopen(returncode=0,
                                            output="Detecting clouds...\nDone\n"))

    reprocess.runFluxStage(DummyConfig(), night_dir, "FTPdetectinfo.txt")

    assert "Detecting clouds..." in capturedLog
    assert "Done" in capturedLog


def test_stageFailureDoesNotPropagate(tmpdir, monkeypatch):
    """ A failed stage costs the night its flux products, never the capture loop. """

    night_dir = str(tmpdir)
    installFakePopen(monkeypatch, FakePopen(returncode=1))

    assert reprocess.runFluxStage(DummyConfig(), night_dir, "FTPdetectinfo.txt") is False
    assert os.listdir(night_dir) == []


def test_wedgedStageIsKilled(tmpdir, monkeypatch):
    """ A stage that never returns is killed rather than hanging the capture loop. """

    night_dir = str(tmpdir)
    fake = installFakePopen(monkeypatch, FakePopen(raise_timeout=True))

    ok = reprocess.runFluxStage(DummyConfig(), night_dir, "FTPdetectinfo.txt", timeout=0.01)

    assert ok is False
    assert fake.killed
    assert not os.path.exists(stateDirOf(fake.cmd))


class OneLineStream(object):
    """ Child stdout that yields a single line, then signals EOF to the test.

    The event fires only after the forwarder has come back for the next line, which
    is how a test can know the first line has already been logged and acted upon.
    """

    def __init__(self, line, drained):
        self.lines = [line]
        self.drained = drained

    def readline(self):

        if self.lines:
            return self.lines.pop(0)

        self.drained.set()

        return ''

    def close(self):
        pass


class NeverFinishingPopen(object):
    """ A stage that prints one line and then never exits, recording its budgets. """

    def __init__(self, line):
        self.drained = threading.Event()
        self.waits = []
        self.killed = False
        self.cmd = None
        self.cwd = None
        self.stdout = OneLineStream(line, self.drained)

    def wait(self, timeout=None):

        # Let the forwarder deliver the line before the parent rules on the timeout
        self.drained.wait(10)

        if self.killed:
            return -9

        self.waits.append(timeout)

        raise subprocess.TimeoutExpired("stage", timeout)

    def kill(self):
        self.killed = True


def test_gateWaitIsNotChargedToTheStagesBudget(tmpdir, monkeypatch):
    """ The stage's budget is for its work, and queueing for a slot is not work.

    The parent starts its clock at Popen, but the child first waits on the machine-wide
    flux slot gate - 2 h 48 min of a 4 h budget on a six-camera pod, which killed a stage
    that had already written its scoring products. Once the child says it is through the
    gate, the budget starts again from there.
    """

    night_dir = str(tmpdir)
    fake = installFakePopen(monkeypatch, NeverFinishingPopen(
        "flux gate: {:s}, work begins after 10085 s of waiting\n".format(
            GATE_WORK_START_MARKER)))

    ok = reprocess.runFluxStage(DummyConfig(), night_dir, "FTPdetectinfo.txt",
                                timeout=0.05)

    # It still dies - it never finishes - but only after a second, full budget
    assert ok is False
    assert fake.killed
    assert len(fake.waits) == 2
    assert fake.waits[1] > 0


def test_theBudgetIsGrantedOnlyOnce(tmpdir, monkeypatch):
    """ Ordinary output must not extend the deadline, or the backstop stops backstopping. """

    night_dir = str(tmpdir)
    fake = installFakePopen(monkeypatch, NeverFinishingPopen("Detecting clouds...\n"))

    ok = reprocess.runFluxStage(DummyConfig(), night_dir, "FTPdetectinfo.txt",
                                timeout=0.05)

    assert ok is False
    assert fake.killed
    assert len(fake.waits) == 1


def test_unstartableStageDoesNotPropagate(tmpdir, monkeypatch):
    """ If the stage cannot be spawned at all, the night still gets archived. """

    night_dir = str(tmpdir)

    def explode(*args, **kwargs):
        raise OSError("no such executable")

    monkeypatch.setattr(reprocess.subprocess, "Popen", explode)

    assert reprocess.runFluxStage(DummyConfig(), night_dir, "FTPdetectinfo.txt") is False
    assert os.listdir(night_dir) == []


def test_captureParentDoesNotImportTheFluxStack():
    """ The point of the split: the flux stack must stay out of the capture parent.

    Reprocess is imported by StartCapture, so an import of Utils.Flux here would put the
    whole scoring/stills/tree stack in the address space of every long-lived station
    process, which is what this change exists to stop.
    """

    proc = subprocess.Popen(
        [sys.executable, "-c",
         "import sys; import RMS.Reprocess; "
         "sys.exit(1 if 'Utils.Flux' in sys.modules else 0)"],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    out, _ = proc.communicate()

    assert proc.returncode == 0, "Utils.Flux is imported by RMS.Reprocess:\n" + out
