#!/usr/bin/env python
"""
compact_injector.py -- deliver an auto-generated ``/compact`` into the FSRS
autoresearch Claude Code session, on the human's authority.

WHY THIS EXISTS
---------------
Claude Code has no API/hook to trigger a *focused* compaction, and the agent is
(correctly) not allowed to inject keystrokes into its own live session -- that
would be a self-feeding control loop. So a human runs THIS process. The agent
only writes the command text to a request file; this watcher, started by you,
delivers it. That keeps a person in charge of the self-feeding step.

HOW IT WORKS
------------
Polls ``result/.compact_request.txt``. The agent writes that file (one line:
``/compact <focus>``) right after an auto-tune commit, when compaction is due.
When it appears, this finds the ``claude.exe`` console and "types" the line in
via the Windows console input buffer (AttachConsole + WriteConsoleInputW), then
presses Enter. The request file is deleted so it fires exactly once.

Each injection runs in a short-lived child process: AttachConsole/FreeConsole
are per-process, so isolating them keeps this watcher's own console (and its
live --foreground log) intact.

The mechanism is validated: WriteConsoleInput delivers characters and Enter to a
raw-mode reader, and because a process tree shares one console, attaching to the
top ``claude.exe`` PID reaches the TUI's input no matter which child reads it.

RUN IT  (from the repo root, once per Claude session)
-----------------------------------------------------
    python  src\\autoresearch\\compact_injector.py --foreground   # watch the log live
    pythonw src\\autoresearch\\compact_injector.py                # background, no window

Stop it: close the foreground window, ``Stop-Process`` the pythonw, or create
the file ``result/.compact_injector.stop``.

Windows only (uses the Win32 console API).
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as w
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULT = REPO / "result"
REQUEST = RESULT / ".compact_request.txt"
STOP = RESULT / ".compact_injector.stop"
LOCK = RESULT / ".compact_injector.lock"
LOG = RESULT / ".compact_injector.log"

FOREGROUND = False

k32 = ctypes.WinDLL("kernel32", use_last_error=True)

# ── console input injection ──────────────────────────────────────────────────
KEY_EVENT = 0x0001
GENERIC_RW = 0xC0000000
SHARE_RW = 0x00000003
OPEN_EXISTING = 3
INVALID = w.HANDLE(-1).value


class _KEY(ctypes.Structure):
    _fields_ = [
        ("bKeyDown", w.BOOL), ("wRepeatCount", w.WORD), ("wVirtualKeyCode", w.WORD),
        ("wVirtualScanCode", w.WORD), ("UnicodeChar", ctypes.c_wchar), ("dwControlKeyState", w.DWORD),
    ]


class _U(ctypes.Union):
    _fields_ = [("KeyEvent", _KEY), ("_pad", ctypes.c_byte * 16)]


class INPUT_RECORD(ctypes.Structure):
    _fields_ = [("EventType", w.WORD), ("Event", _U)]


k32.AttachConsole.argtypes = [w.DWORD]; k32.AttachConsole.restype = w.BOOL
k32.FreeConsole.restype = w.BOOL
k32.CreateFileW.argtypes = [w.LPCWSTR, w.DWORD, w.DWORD, ctypes.c_void_p, w.DWORD, w.DWORD, w.HANDLE]
k32.CreateFileW.restype = w.HANDLE
k32.WriteConsoleInputW.argtypes = [w.HANDLE, ctypes.c_void_p, w.DWORD, ctypes.POINTER(w.DWORD)]
k32.WriteConsoleInputW.restype = w.BOOL
k32.CloseHandle.argtypes = [w.HANDLE]


def _key_down(ch: str) -> INPUT_RECORD:
    rec = INPUT_RECORD()
    rec.EventType = KEY_EVENT
    e = rec.Event.KeyEvent
    e.bKeyDown = 1
    e.wRepeatCount = 1
    e.wVirtualKeyCode = 0x0D if ch == "\r" else 0
    e.wVirtualScanCode = 0
    e.UnicodeChar = ch
    e.dwControlKeyState = 0
    return rec


def inject(pid: int, text: str) -> int:
    """Type ``text`` into pid's console as one submitted line.

    Internal CR/LF are flattened to spaces so a multi-line focus can't submit
    early; exactly one trailing CR is appended to submit. NOTE: this calls
    Free/AttachConsole on the *current* process -- run it in a throwaway child
    (see ``_deliver``) so the watcher keeps its own console.
    """
    body = text.replace("\r", " ").replace("\n", " ").rstrip()
    seq = body + "\r"
    k32.FreeConsole()
    if not k32.AttachConsole(pid):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        hin = k32.CreateFileW("CONIN$", GENERIC_RW, SHARE_RW, None, OPEN_EXISTING, 0, None)
        if hin in (0, INVALID):
            raise ctypes.WinError(ctypes.get_last_error())
        total = 0
        for i in range(0, len(seq), 128):
            chunk = seq[i:i + 128]
            recs = [_key_down(c) for c in chunk]
            arr = (INPUT_RECORD * len(recs))(*recs)
            nwr = w.DWORD(0)
            if not k32.WriteConsoleInputW(hin, ctypes.byref(arr), len(recs), ctypes.byref(nwr)):
                err = ctypes.get_last_error()
                k32.CloseHandle(hin)
                raise ctypes.WinError(err)
            total += nwr.value
            time.sleep(0.01)
        k32.CloseHandle(hin)
        return total
    finally:
        k32.FreeConsole()


def _deliver(pid: int, request: Path) -> str:
    """Run one injection in a console-less child so our own console survives."""
    try:
        r = subprocess.run(
            [sys.executable, os.path.abspath(__file__),
             "--_inject-pid", str(pid), "--_inject-file", str(request)],
            capture_output=True, text=True, timeout=20,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return (r.stdout or "").strip() or (r.stderr or "").strip() or f"ERR rc={r.returncode}"
    except Exception as ex:  # noqa: BLE001
        return f"ERR {ex!r}"


# ── process enumeration (Toolhelp32) ─────────────────────────────────────────
TH32CS_SNAPPROCESS = 0x00000002


class PROCENTRY(ctypes.Structure):
    _fields_ = [
        ("dwSize", w.DWORD), ("cntUsage", w.DWORD), ("th32ProcessID", w.DWORD),
        ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)), ("th32ModuleID", w.DWORD),
        ("cntThreads", w.DWORD), ("th32ParentProcessID", w.DWORD), ("pcPriClassBase", ctypes.c_long),
        ("dwFlags", w.DWORD), ("szExeFile", ctypes.c_wchar * 260),
    ]


k32.CreateToolhelp32Snapshot.restype = w.HANDLE


def _procs() -> list[tuple[int, int, str]]:
    snap = k32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if snap in (0, INVALID):
        return []
    out: list[tuple[int, int, str]] = []
    e = PROCENTRY()
    e.dwSize = ctypes.sizeof(e)
    try:
        if k32.Process32FirstW(snap, ctypes.byref(e)):
            while True:
                out.append((e.th32ProcessID, e.th32ParentProcessID, e.szExeFile))
                if not k32.Process32NextW(snap, ctypes.byref(e)):
                    break
    finally:
        k32.CloseHandle(snap)
    return out


def _alive(pid: int) -> bool:
    return any(p[0] == pid for p in _procs())


def find_claude_pid() -> int | None:
    """The top-level claude.exe (the console owner): prefer the one parented to
    explorer.exe; fall back to any claude.exe not parented to another claude."""
    procs = _procs()
    byid = {p[0]: p for p in procs}

    def pname(pid: int) -> str:
        return (byid.get(pid) or (0, 0, ""))[2].lower()

    claude = [p for p in procs if p[2].lower() == "claude.exe"]
    top = [p for p in claude if pname(p[1]) == "explorer.exe"]
    if top:
        return top[0][0]
    non_child = [p for p in claude if pname(p[1]) != "claude.exe"]
    if non_child:
        return non_child[0][0]
    return claude[0][0] if claude else None


# ── watcher loop ─────────────────────────────────────────────────────────────
def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass
    if FOREGROUND:
        print(line, flush=True)


def _read_stable(path: Path, settle: float = 0.2) -> str:
    """Read a file twice with a short gap; return its text only if unchanged
    (guards against reading while the agent is mid-write)."""
    try:
        a = path.read_text(encoding="utf-8")
        time.sleep(settle)
        b = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    return a if a == b else ""


def main() -> None:
    global FOREGROUND
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pid", type=int, default=0, help="claude.exe console PID (default: auto-detect)")
    ap.add_argument("--interval", type=float, default=1.5, help="poll seconds (default 1.5)")
    ap.add_argument("--foreground", action="store_true", help="run in console, echo the log")
    ap.add_argument("--once", action="store_true", help="deliver one pending request, then exit")
    ap.add_argument("--_inject-pid", dest="inject_pid", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--_inject-file", dest="inject_file", default="", help=argparse.SUPPRESS)
    a = ap.parse_args()

    # internal one-shot injection child (keeps the parent watcher's console)
    if a.inject_pid:
        try:
            text = Path(a.inject_file).read_text(encoding="utf-8")
            n = inject(a.inject_pid, text)
            sys.stdout.write(f"OK {n}")
        except Exception as ex:  # noqa: BLE001
            sys.stdout.write(f"ERR {ex!r}")
        return

    FOREGROUND = a.foreground
    RESULT.mkdir(parents=True, exist_ok=True)

    if LOCK.exists():
        try:
            other = int(LOCK.read_text(encoding="utf-8").strip() or 0)
        except (OSError, ValueError):
            other = 0
        if other and other != os.getpid() and _alive(other):
            log(f"another injector (pid {other}) is already running; exiting.")
            return
    LOCK.write_text(str(os.getpid()), encoding="utf-8")
    if STOP.exists():
        STOP.unlink()
    log(f"compact-injector started (pid {os.getpid()}); polling {REQUEST.name} every {a.interval}s")

    try:
        while True:
            if STOP.exists():
                STOP.unlink()
                log("stop file seen; exiting.")
                break
            if REQUEST.exists():
                text = _read_stable(REQUEST)
                if text.strip():
                    pid = a.pid or find_claude_pid()
                    if not pid:
                        log("claude.exe console not found; will retry.")
                    else:
                        res = _deliver(pid, REQUEST)
                        if res.startswith("OK"):
                            log(f"delivered to claude pid {pid} ({res}): {text.strip()[:70]!r}")
                            try:
                                REQUEST.unlink()
                            except OSError:
                                pass
                            if a.once:
                                break
                        else:
                            log(f"inject failed (pid {pid}): {res}; will retry next poll.")
            time.sleep(a.interval)
    finally:
        try:
            if LOCK.exists() and LOCK.read_text(encoding="utf-8").strip() == str(os.getpid()):
                LOCK.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    if os.name != "nt":
        sys.exit("compact_injector.py is Windows-only (uses the Win32 console API).")
    main()
