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
When it appears, this finds the native ``claude.exe`` GUI window, brings it to
the foreground, and "types" the line via Win32 ``SendInput`` (Unicode key events
for the text, then a real Enter to submit). The request file is deleted so it
fires exactly once.

The native ``claude.exe`` renders its TUI in its own GUI window (HWND with title
"Claude") and reads the keyboard from the Windows message queue -- it has no
attachable console -- so console injection (WriteConsoleInput) does NOT work;
SendInput into the focused window is the right mechanism (same as typing into an
Electron/Chromium app). The window must be foreground to receive input, so this
briefly raises it; it verifies focus first, so it can never type elsewhere.

RUN IT  (from anywhere; paths resolve from the script location)
---------------------------------------------------------------
    python  src\\autoresearch\\compact_injector.py --foreground   # watch the log live
    pythonw src\\autoresearch\\compact_injector.py                # background, no window

Stop it: close the foreground window, ``Stop-Process`` the pythonw, or create
the file ``result/.compact_injector.stop``.

Windows only.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as w
import os
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
u32 = ctypes.WinDLL("user32", use_last_error=True)

# ── find the Claude GUI window for a pid ─────────────────────────────────────
WNDENUMPROC = ctypes.WINFUNCTYPE(w.BOOL, w.HWND, w.LPARAM)
u32.EnumWindows.argtypes = [WNDENUMPROC, w.LPARAM]
u32.EnumWindows.restype = w.BOOL
u32.GetWindowThreadProcessId.argtypes = [w.HWND, ctypes.POINTER(w.DWORD)]
u32.GetWindowThreadProcessId.restype = w.DWORD
u32.IsWindowVisible.argtypes = [w.HWND]
u32.IsWindowVisible.restype = w.BOOL
u32.GetWindowTextLengthW.argtypes = [w.HWND]
u32.GetWindowTextLengthW.restype = ctypes.c_int


def find_window_for_pid(pid: int):
    result: list[int] = []

    def _cb(hwnd, _lparam):
        owner = w.DWORD(0)
        u32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if (owner.value == pid and u32.IsWindowVisible(hwnd)
                and u32.GetWindowTextLengthW(hwnd) > 0):
            result.append(hwnd)
            return False  # stop enumerating
        return True

    u32.EnumWindows(WNDENUMPROC(_cb), 0)
    return result[0] if result else None


# ── force a window to the foreground (AttachThreadInput trick) ───────────────
u32.GetForegroundWindow.restype = w.HWND
u32.SetForegroundWindow.argtypes = [w.HWND]
u32.SetForegroundWindow.restype = w.BOOL
u32.BringWindowToTop.argtypes = [w.HWND]
u32.ShowWindow.argtypes = [w.HWND, ctypes.c_int]
u32.IsIconic.argtypes = [w.HWND]
u32.IsIconic.restype = w.BOOL
u32.AttachThreadInput.argtypes = [w.DWORD, w.DWORD, w.BOOL]
u32.AttachThreadInput.restype = w.BOOL
k32.GetCurrentThreadId.restype = w.DWORD
SW_RESTORE = 9


def _force_foreground(hwnd) -> bool:
    if u32.GetForegroundWindow() == hwnd:
        return True
    if u32.IsIconic(hwnd):
        u32.ShowWindow(hwnd, SW_RESTORE)
    cur = k32.GetCurrentThreadId()
    target = u32.GetWindowThreadProcessId(hwnd, None)
    fg = u32.GetForegroundWindow()
    fg_tid = u32.GetWindowThreadProcessId(fg, None) if fg else 0
    att_fg = att_t = False
    try:
        if fg_tid and fg_tid != cur:
            att_fg = bool(u32.AttachThreadInput(cur, fg_tid, True))
        if target and target != cur:
            att_t = bool(u32.AttachThreadInput(cur, target, True))
        u32.BringWindowToTop(hwnd)
        u32.SetForegroundWindow(hwnd)
    finally:
        if att_fg:
            u32.AttachThreadInput(cur, fg_tid, False)
        if att_t:
            u32.AttachThreadInput(cur, target, False)
    time.sleep(0.12)
    return u32.GetForegroundWindow() == hwnd


# ── SendInput keyboard injection ─────────────────────────────────────────────
INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
VK_RETURN = 0x0D
ULONG_PTR = ctypes.c_size_t


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", w.WORD), ("wScan", w.WORD), ("dwFlags", w.DWORD),
        ("time", w.DWORD), ("dwExtraInfo", ULONG_PTR),
    ]


class _IU(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT), ("_pad", ctypes.c_byte * 32)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", w.DWORD), ("u", _IU)]


u32.SendInput.argtypes = [w.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
u32.SendInput.restype = w.UINT


def _ki(wVk: int, wScan: int, flags: int) -> INPUT:
    rec = INPUT()
    rec.type = INPUT_KEYBOARD
    rec.u.ki = KEYBDINPUT(wVk, wScan, flags, 0, 0)
    return rec


def _send(inputs: list[INPUT]) -> None:
    arr = (INPUT * len(inputs))(*inputs)
    sent = u32.SendInput(len(inputs), arr, ctypes.sizeof(INPUT))
    if sent != len(inputs):
        raise ctypes.WinError(ctypes.get_last_error())


def inject(pid: int, text: str) -> int:
    """Type ``text`` into pid's GUI window as one submitted line.

    Internal CR/LF are flattened to spaces so a multi-line focus can't submit
    early; a real Enter is sent at the end to submit.
    """
    body = text.replace("\r", " ").replace("\n", " ").rstrip()
    hwnd = find_window_for_pid(pid)
    if not hwnd:
        raise OSError(f"no visible window for pid {pid}")
    if not _force_foreground(hwnd):
        raise OSError("could not bring the Claude window to the foreground; will retry")
    batch: list[INPUT] = []
    for ch in body:
        code = ord(ch)
        batch.append(_ki(0, code, KEYEVENTF_UNICODE))
        batch.append(_ki(0, code, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP))
        if len(batch) >= 40:
            _send(batch)
            batch = []
            time.sleep(0.004)
    if batch:
        _send(batch)
    time.sleep(0.05)
    _send([_ki(VK_RETURN, 0, 0), _ki(VK_RETURN, 0, KEYEVENTF_KEYUP)])
    return len(body)


# ── process enumeration (Toolhelp32) ─────────────────────────────────────────
TH32CS_SNAPPROCESS = 0x00000002
INVALID = w.HANDLE(-1).value


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
    """The top-level claude.exe (the GUI-window owner): prefer the one parented
    to explorer.exe; fall back to any claude.exe not parented to another claude."""
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
    ap.add_argument("--pid", type=int, default=0, help="claude.exe PID (default: auto-detect)")
    ap.add_argument("--interval", type=float, default=1.5, help="poll seconds (default 1.5)")
    ap.add_argument("--foreground", action="store_true", help="run in console, echo the log")
    ap.add_argument("--once", action="store_true", help="deliver one pending request, then exit")
    a = ap.parse_args()
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
                        log("claude.exe not found; will retry.")
                    else:
                        try:
                            n = inject(pid, text)
                            log(f"delivered {n} chars to claude pid {pid}: {text.strip()[:70]!r}")
                            try:
                                REQUEST.unlink()
                            except OSError:
                                pass
                            if a.once:
                                break
                        except OSError as ex:
                            log(f"inject failed (pid {pid}): {ex}; will retry next poll.")
            time.sleep(a.interval)
    finally:
        try:
            if LOCK.exists() and LOCK.read_text(encoding="utf-8").strip() == str(os.getpid()):
                LOCK.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    if os.name != "nt":
        sys.exit("compact_injector.py is Windows-only.")
    main()
