@echo off
REM Start the FSRS compact-injector in the foreground (shows its log).
REM Double-click this, or run it once per Claude Code session, from the repo.
REM It waits for result\.compact_request.txt (written by the agent when a
REM compaction is due) and types the /compact command into the claude.exe
REM console for you. Close this window to stop it.
cd /d "%~dp0"
title FSRS compact-injector
python "%~dp0src\autoresearch\compact_injector.py" --foreground
pause
