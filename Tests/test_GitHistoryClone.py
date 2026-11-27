""" Tests of cloning the commit history used to measure how far the station code is behind the remote. """

from __future__ import print_function, division, absolute_import

import os

import pytest

import RMS.Misc as misc
import RMS.Formats.ObservationSummary as obs
from RMS.Misc import GitCommandTimeout


PROXY_VARIABLES = ("https_proxy", "HTTPS_PROXY", "http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY")

REMOTES = [["origin", "https://github.com/a/RMS.git"], ["upstream", "https://github.com/b/RMS.git"]]


class FakeGit(object):
    """ Stands in for runGitCommand: records the clones and lets each one succeed, fail or time out. """

    def __init__(self, clone_results):

        # One entry per clone attempt: 0 to succeed, 'timeout' to be killed halfway through
        self.clone_results = list(clone_results)
        self.cloned_urls = []

    def __call__(self, args, cwd=None, timeout=None, network=False):

        if args[0] == "clone":

            url, clone_directory = args[1], args[-1]
            self.cloned_urls.append(url)
            result = self.clone_results.pop(0)

            # Like git, refuse to clone into a directory which is not empty
            if os.path.isdir(clone_directory) and os.listdir(clone_directory):
                return 128, "", "fatal: destination path already exists and is not an empty directory"

            # Clones and killed clones both leave a directory behind
            os.makedirs(os.path.join(clone_directory, ".git"))

            if result == "timeout":
                raise GitCommandTimeout("git clone timed out")

            return result, "", ""

        # The remote name of the clone
        if args[0] == "remote" and len(args) == 1:
            return 0, "origin\n", ""

        return 0, "", ""


@pytest.fixture
def noProxy(monkeypatch):
    """ No proxy in the environment, and no git configuration is read. """

    for name in PROXY_VARIABLES:
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setattr(misc, "runGitCommand", lambda *args, **kwargs: (1, "", ""))


def test_partial_clone_removed_before_next_remote(monkeypatch, tmp_path, noProxy):
    """ A clone killed on the timeout leaves a partial directory, which must not block the next remote. """

    fake_git = FakeGit([ "timeout", 0])
    monkeypatch.setattr(obs, "runGitCommand", fake_git)
    monkeypatch.setattr(obs, "hostReachable", lambda host, port: True)

    clone_directory = obs.updateCommitHistoryDirectory(REMOTES, str(tmp_path))

    assert fake_git.cloned_urls == [REMOTES[0][1], REMOTES[1][1]]
    assert clone_directory == os.path.join(str(tmp_path), "commit_history")


def test_unreachable_host_skipped_without_proxy(monkeypatch, tmp_path, noProxy):
    """ Without a proxy an unreachable host is skipped before git is run. """

    fake_git = FakeGit([0, 0])
    monkeypatch.setattr(obs, "runGitCommand", fake_git)
    monkeypatch.setattr(obs, "hostReachable", lambda host, port: False)

    with pytest.raises(RuntimeError):
        obs.updateCommitHistoryDirectory(REMOTES, str(tmp_path))

    assert fake_git.cloned_urls == []


@pytest.mark.parametrize("source", ["environment", "git_config"])
def test_proxy_bypasses_reachability_check(monkeypatch, tmp_path, noProxy, source):
    """ Through a proxy the host cannot be connected to directly, but git can still clone. """

    if source == "environment":
        monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:3128")

    else:
        monkeypatch.setattr(misc, "runGitCommand",
            lambda args, **kwargs: (0, "http.proxy http://proxy.example:3128\n", "")
            if args[:2] == ["config", "--get-regexp"] else (1, "", ""))

    fake_git = FakeGit([0])
    monkeypatch.setattr(obs, "runGitCommand", fake_git)
    monkeypatch.setattr(obs, "hostReachable", lambda host, port: False)

    obs.updateCommitHistoryDirectory(REMOTES[:1], str(tmp_path))

    assert fake_git.cloned_urls == [REMOTES[0][1]]
