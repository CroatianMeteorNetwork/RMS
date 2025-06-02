# RPi Meteor Station
# Copyright (C) 2025 David Rollinson
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


from __future__ import print_function, division, absolute_import


import os
import subprocess

from datetime import datetime, timezone
import shutil
import RMS.ConfigReader as cr




def updateCommitHistoryDirectory(remote_urls, target_directory):

    """

    Clone only the commit history of a remote repository.

    Args:
        remote_urls: the remote url to be cloned
        target_directory: the directory into which to clone

    Returns:
        directory of the repository

    """


    if os.path.exists(target_directory):
        shutil.rmtree(target_directory)

    os.makedirs(target_directory)
    first_remote = True
    for remote_url in remote_urls:
        local_name, url = remote_url[0], remote_url[1]

        if first_remote:
            first_remote = False
            p = subprocess.Popen(["git", "clone", url, "--filter=blob:none", "--no-checkout"], cwd=target_directory,
                             stdout=subprocess.PIPE)
            p.wait()
            # this first remote might have been pulled in with the wrong local_name so rename it
            commit_repo_directory = os.path.join(target_directory, os.listdir(target_directory)[0])
            downloaded_remote_name = subprocess.check_output(["git", "remote"], cwd = commit_repo_directory).strip().decode('utf-8')
            print(downloaded_remote_name)
            if downloaded_remote_name != local_name:
                p = subprocess.Popen(["git", "remote", "rename", downloaded_remote_name, local_name], cwd = commit_repo_directory)
                p.wait()

        else:
            # this is not the first remote so add another remote

            p = subprocess.Popen(["git", "remote", "add", local_name, url], cwd = commit_repo_directory)
            p.wait()
            p = subprocess.Popen(["git", "fetch", "--filter=blob:none", local_name], cwd = commit_repo_directory)
            p.wait()
    return commit_repo_directory

def getCommit(repo):

    """

    Args:
        repo:file location of a repository

    Returns:
        latest commit in that repository
    """

    commit = subprocess.check_output(["git", "log", "-n 1", "--pretty=format:%H"], cwd=repo).decode(
        "utf-8")

    return commit




def getDateOfCommit(repo, commit):

    """

    Args:
        repo: directory of repository
        commit: commit hash

    Returns:
        python datetime object of the date of that commit
    """

    if commit is None:
        return datetime.strptime("2000-01-01 00:00:00 +00:00", "%Y-%m-%d %H:%M:%S %z")
    commit_date  = subprocess.check_output(["git", "show", "-s", "--format=%ci", commit], cwd=repo).decode('utf8').replace("\n","")
    return datetime.strptime(commit_date, "%Y-%m-%d %H:%M:%S %z")

def getRemoteUrls(repo):

    """

    Args:
        repo: diretory of repository

    Returns:
        return a list of [remote, url] where remote is the local name of a remote and URL is the URL of the remote
    """

    urls_and_remotes = subprocess.check_output(["git", "remote", "-v"], cwd=repo).decode("utf-8").split("\n")
    url_remote_list_to_return = []
    for url_and_remote in urls_and_remotes:
        url_and_remote = url_and_remote.split("\t")
        if len(url_and_remote) == 2:
            remote, url = [url_and_remote[0], url_and_remote[1]]
            url = url.split(" ")[0]
            if not [remote, url] in url_remote_list_to_return:
                url_remote_list_to_return.append([remote, url])
    return url_remote_list_to_return


def getBranchOfCommit(repo, commit):

    """

    Args:
        repo: directory of repository
        commit: commit hash

    Returns:
        A branch where a commit exists. There may be several branches, only one will be returned.
    """

    local_branch = subprocess.check_output(["git", "branch", "-a", "--contains", commit], cwd=repo).decode(
         "utf-8").split("\n")[0].replace("*", "").strip()
    return local_branch

def getLatestCommit(repo, commit_branch):

    """

    Args:
        repo:repository directory
        commit_branch: branch

    Returns:
        the hash of the latest commit on commit_branch in repository
    """

    if commit_branch.startswith("remotes/"):
        commit_branch = commit_branch[len("remotes/"):]

    commit_list = subprocess.check_output(["git", "branch", "-r", "-v"], cwd=repo).decode("utf-8").split("\n")
    commit = None
    for branch in commit_list:

        branch_list = branch.split()
        if len(branch_list) > 1:
            remote_branch = branch_list[0]
            remote_commit = branch_list[1]

            if commit_branch == remote_branch:
                commit = remote_commit
                break
    return commit

def getRemoteBranchNameForCommit(repo, commit):

    """

    Get the remote branch name for a commit on a local branch. If the commit does not exist on the remote,
    returns None.

    Args:
        repo: directory of repository
        commit: commit hash

    Returns:
        the full name of the remote branch where commit exists, or None if commit does not exist
    """

    local_branch_list = []
    try:
        local_branch_list = subprocess.check_output(["git", "branch", "-a", "--contains", commit], cwd=repo).decode(
            "utf-8").split("\n")
    except:
        pass

    remote_branch_name = None
    for branch in local_branch_list:
        branch_stripped = branch.strip()
        if branch_stripped.startswith("remotes/"):
            remote_branch_name = branch_stripped

    return remote_branch_name

def daysBehind(syscon):

    """

    Args:
        syscon: RMS config object

    Returns:
        number of days behind the latest remote commit that the latest local commit is on the active branch
    """


    latest_local_commit = getCommit(os.getcwd())
    latest_local_date = getDateOfCommit(os.getcwd(), latest_local_commit)
    target_directory = os.path.join(syscon.data_dir, "CommitHistory")
    remote_urls = getRemoteUrls(os.getcwd())
    commit_repo_directory = updateCommitHistoryDirectory(remote_urls, target_directory)
    remote_branch_of_commit = getRemoteBranchNameForCommit(commit_repo_directory, latest_local_commit)
    if not remote_branch_of_commit is None:
        latest_remote_date = getDateOfCommit(commit_repo_directory, remote_branch_of_commit)
        days_behind = (latest_remote_date - latest_local_date).total_seconds() / (60 * 60 * 24)
        return days_behind
    else:
        return 0

if __name__ == "__main__":

    """
    This code is not intended to be used in production, just for test purposes
    """

    import argparse

    arg_parser = argparse.ArgumentParser(description="""Check if the loca commit is up to date. \
        """, formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('-u', '--url', nargs=1, metavar='REMOTE_REPO_PATH', type=str,
                            help="Path to the repo which will be used instead of https://github.com/CroatianMeteorNetwork/RMS/")

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str,
                            help="Path to a config file which will be used instead of the default one.")

    arg_parser.add_argument('-m', '--commit', nargs=1, metavar='COMMIT', type=str,
                            help="Commit age to be retrieved.")

    cml_args = arg_parser.parse_args()


    if cml_args.url is None:
        remote_repo = "https://github.com/CroatianMeteorNetwork/RMS/"
    else:
        remote_repo = cml_args.url


    if cml_args.config is None:
        config_path = os.path.expanduser("~/source/RMS/.config")
    else:
        config_path = os.path.expanduser(cml_args.url)


    if cml_args.commit   is None:
        commit = None
    else:
        commit = cml_args.commit[0]


    syscon = cr.loadConfigFromDirectory(cml_args.config, os.path.abspath('.'))
    latest_local_commit = getCommit(os.getcwd())
    latest_local_branch = getBranchOfCommit(os.getcwd(), latest_local_commit)
    latest_local_date = getDateOfCommit(os.getcwd(), latest_local_commit)
    target_directory = os.path.join(syscon.data_dir, "CommitHistory")
    remote_urls = getRemoteUrls(os.getcwd())
    commit_repo_directory = updateCommitHistoryDirectory(remote_urls, target_directory)

    remote_branch_of_commit = getRemoteBranchNameForCommit(commit_repo_directory, latest_local_commit)
    if not remote_branch_of_commit is None:
        latest_remote_commit = getLatestCommit(commit_repo_directory, remote_branch_of_commit)

        latest_remote_date = getDateOfCommit(commit_repo_directory, remote_branch_of_commit)
        print("Current local commit is {} on branch {} at {}".format(latest_local_commit, latest_local_branch, latest_local_date))
        print("Latest remote commit is {} on {}".format(latest_remote_commit, latest_remote_date))
        hours_behind = (latest_remote_date - latest_local_date).total_seconds() / (60 * 60)
        print("Local commit {} on branch {} is".format(latest_local_commit, latest_local_branch))

        if hours_behind > 0:
            print("{:.2f} hours behind".format(hours_behind))
        elif hours_behind < 0:
            print("{:.2f} hours ahead of".format(0 - hours_behind), end="")
            print("remote commit {} on {}".format(latest_remote_commit, remote_branch_of_commit))
            print("Probably local commits are not yet pushed")
        elif hours_behind == 0:
            print("{:.2f} hours ahead of ".format(0 - hours_behind), end="")
            print("remote commit {} on {}".format(latest_remote_commit, remote_branch_of_commit))
            print("Local repository is up to date with remote repository")
        if hours_behind > 36:
            print("Local is more than 36 hours behind, something might be wrong.")

    else:
        print("Local commit {} on branch {} with date {} not found on remote".format(latest_local_commit, latest_local_branch, latest_local_date))


    print("Local repo is {:.2f} days behind".format(daysBehind(syscon)))