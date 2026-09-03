#!/usr/bin/env python3
"""Run a command in its own session, surviving anything that kills this shell.

  python3 experiments/geo_v137ds/detach.py <logfile> <cmd> [args...]

Prints the detached pid and exits immediately.

Two 97-comment generation runs were lost at 56/97 and at 97/97-before-writing,
each time as part of a batch in which several unrelated background tasks were
killed at the same moment -- including tasks that had finished long before. The
generation was not the target; it was collateral of a sweep over the session's
background tasks. `nohup` does not help, because that only blocks SIGHUP and
these arrive as SIGTERM to the process group, and macOS has no `setsid`.

`os.setsid()` in a forked child gives the run its own session and process
group, so a signal delivered to this shell's group never reaches it. stdin is
detached from the terminal and stdout/stderr go to the log.
"""
import os
import sys

if len(sys.argv) < 3:
    sys.exit("usage: detach.py <logfile> <cmd> [args...]")

logfile, command = sys.argv[1], sys.argv[2:]

pid = os.fork()
if pid > 0:
    # Parent: report and leave. The child is already reparented to init.
    print(f"detached pid={pid} log={logfile}")
    os._exit(0)

os.setsid()
if os.fork() > 0:          # second fork: cannot reacquire a controlling tty
    os._exit(0)

os.chdir(os.getcwd())
fd = os.open(logfile, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
os.dup2(fd, 1)
os.dup2(fd, 2)
os.execvp(command[0], command)
