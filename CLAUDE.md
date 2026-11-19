@AGENTS.md

## Notes for Claude Code

- Before running anything, use the environment with the packages from `requirements.txt` and build the Cython
  modules (`python setup.py build_ext --inplace`).
- Run the full test suite before you call a change done, and report the real result.
- When you edit a CRLF file with a script, write it with `newline='\r\n'`, then check `git diff --stat`.
- For the check that a new test fails without the fix, put the old file back with `git show` and restore yours;
  do not use `git stash` (the stash list belongs to the user).
- On a station, or any machine running RMS, never stop processes by name pattern (`pkill -f`, `killall`):
  several cameras can run on one computer. Stop a process by its PID.
