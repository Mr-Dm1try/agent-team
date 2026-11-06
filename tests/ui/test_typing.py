"""Reading keys from a terminal: the line editor, and the reader that switches the
terminal's echo off so the session can draw what is typed."""

from __future__ import annotations

import asyncio
import os
import sys

import pytest

from agent_team.ui.session import LineEditor, StdinLines


def test_keys_build_the_line_and_enter_finishes_it() -> None:
    editor = LineEditor()
    assert editor.feed("hi") == [("typed", "hi")]
    assert editor.feed("\x7fo there\r\n") == [("typed", "ho there"), ("line", "ho there")]
    assert editor.typed == []


def test_editing_keys_and_escape_sequences() -> None:
    editor = LineEditor()
    assert editor.feed("stop the\x17") == [("typed", "stop ")]
    assert editor.feed("\x15") == [("typed", "")]
    # Arrow keys (ESC [ A, ESC O B) and other control keys never reach the line.
    assert editor.feed("\x1b[Aok\x1bOB\x01") == [("typed", "ok")]
    assert editor.feed("\x7f\x7f\x7f") == [("typed", "")]  # backspace past the start is a no-op
    assert editor.feed("é\tx\n") == [("typed", "é x"), ("line", "é x")]


def test_ctrl_d_ends_the_input_only_on_an_empty_line() -> None:
    editor = LineEditor()
    assert editor.feed("ab\x04") == [("typed", "ab")]
    assert editor.feed("\x15\x04") == [("typed", ""), ("end", "")]


@pytest.mark.skipif(sys.platform == "win32", reason="needs a pty")
async def test_the_reader_echoes_nothing_itself_and_gives_the_terminal_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pty
    import termios

    master, slave = pty.openpty()
    stdin = os.fdopen(slave, "r", closefd=False)
    monkeypatch.setattr(sys, "stdin", stdin)
    before = termios.tcgetattr(slave)
    seen: list[tuple[str, bool]] = []
    reader = StdinLines(lambda text, entered: seen.append((text, entered)))
    try:
        reading = asyncio.ensure_future(reader())
        await asyncio.sleep(0.2)  # the thread has switched the terminal to keys
        assert not termios.tcgetattr(slave)[3] & termios.ECHO
        os.write(master, b"wal?")
        await asyncio.sleep(0.2)
        assert seen == [("wal?", False)]  # drawn by the session, not echoed by the terminal
        os.write(master, b"\x7f\x7f\x7fhy\r")
        assert await asyncio.wait_for(reading, 5) == "why"
        assert seen[-1] == ("why", True)
        os.write(master, b"\x04")
        assert await asyncio.wait_for(reader(), 5) is None
        await asyncio.sleep(0.1)
        assert termios.tcgetattr(slave)[3] == before[3]  # echo and line mode are back
    finally:
        stdin.close()
        os.close(slave)
        os.close(master)
