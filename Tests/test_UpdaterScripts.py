""" Unit tests for updater shell scripts and requirements.txt hygiene. """

from __future__ import print_function, division, absolute_import

import os
import re
import subprocess
import sys

import pytest


RMS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
posix_only = pytest.mark.skipif(sys.platform == 'win32', reason='POSIX bash only')


@posix_only
def testPostIncrementFromZeroAbortsUnderErrexit():
    """ Demonstrate why bare ((n++)) breaks set -e when n=0, while n=$((n + 1)) succeeds. """

    bad = subprocess.run(['bash', '-c', 'set -Eeuo pipefail; n=0; ((n++)); echo reached'],
        stdout=subprocess.PIPE, timeout=30)
    good = subprocess.run(['bash', '-c', 'set -Eeuo pipefail; n=0; n=$((n + 1)); echo reached $n'],
        stdout=subprocess.PIPE, timeout=30)

    assert bad.returncode != 0 and b'reached' not in bad.stdout
    assert good.returncode == 0 and good.stdout.strip() == b'reached 1'


@pytest.mark.parametrize('rel_path', ['Scripts/RMS_Update.sh', 'Scripts/MultiCamLinux/GRMSUpdater.sh'])
def testUpdateScriptsHaveNoBarePostIncrement(rel_path):
    """ Errexit update scripts must not use bare ((var++)) / ((var--)) statements. """

    with open(os.path.join(RMS_ROOT, rel_path)) as f:
        src = f.read()

    assert 'set -Eeuo pipefail' in src
    bare = re.findall(r'^\s*\(\(\s*\w+\s*(?:\+\+|--)\s*\)\)\s*$', src, re.MULTILINE)
    assert bare == []


@posix_only
@pytest.mark.parametrize('rel_path', ['Scripts/RMS_Update.sh', 'Scripts/MultiCamLinux/GRMSUpdater.sh'])
def testUpdateScriptsHaveValidBashSyntax(rel_path):
    """ Both update scripts must have valid bash syntax. """

    script = os.path.join(RMS_ROOT, rel_path)
    res = subprocess.run(['bash', '-n', script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    assert res.returncode == 0, res.stderr.decode('utf-8', errors='replace')


@posix_only
@pytest.mark.parametrize('args', [
    ['--term'], ['--term', ''], ['--term', '--force'],
    ['--profile'], ['--profile', ''], ['--profile', '--reboot'],
])
def testGRMSUpdaterRejectsMissingOptionValues(tmp_path, args):
    """ Missing or empty values for --term / --profile must print usage and exit 1 cleanly. """

    script = os.path.join(RMS_ROOT, 'Scripts/MultiCamLinux/GRMSUpdater.sh')

    # Isolate the test from host syslog (logger) and the global /tmp lock file
    stub_bin = tmp_path/'bin'
    stub_bin.mkdir()
    fake_logger = stub_bin/'logger'
    fake_logger.write_text('#!/bin/sh\nexit 0\n')
    os.chmod(str(fake_logger), 0o755)

    env = dict(os.environ)
    env['HOME'] = str(tmp_path)
    env['PATH'] = str(stub_bin) + os.pathsep + env.get('PATH', '')
    env['RMS_GRMS_UPDATER_LOCKFILE'] = str(tmp_path/'grms_updater.lock')

    res = subprocess.run(['bash', script] + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, timeout=30)
    combined = (res.stdout + res.stderr).decode('utf-8', errors='replace')

    assert res.returncode == 1
    assert 'unbound variable' not in combined.lower()
    assert 'Usage:' in combined


def testRequirementsExcludesUnusedImregDft():
    """ requirements.txt must not pull the unused unpinned imreg_dft repository. """

    with open(os.path.join(RMS_ROOT, 'requirements.txt')) as f:
        text = f.read()

    assert 'imreg_dft' not in text
