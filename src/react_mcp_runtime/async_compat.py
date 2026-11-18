"""Cross-platform asyncio event loop compatibility for restricted / socketless sandboxes."""

from __future__ import annotations

import asyncio
import os
import selectors
import sys
import threading

_INSTALLED = False


class _EventWakeupSelector(selectors.BaseSelector):
    """Selector backed by `threading.Event` for socketless thread-safe event-loop wakeups."""

    def __init__(self) -> None:
        super().__init__()
        self._map: dict[int, selectors.SelectorKey] = {}
        self._wake = threading.Event()

    def register(self, fileobj, events, data=None):  # type: ignore[override]
        key = selectors.SelectorKey(fileobj, 0, events, data)
        self._map[id(fileobj)] = key
        return key

    def unregister(self, fileobj):  # type: ignore[override]
        return self._map.pop(id(fileobj))

    def get_map(self):  # type: ignore[override]
        return self._map

    def wakeup(self) -> None:
        self._wake.set()

    def select(self, timeout: float | None = None):  # type: ignore[override]
        if timeout is not None and timeout <= 0:
            self._wake.clear()
            return []
        wait_s = 0.05 if timeout is None else min(timeout, 0.25)
        self._wake.wait(wait_s)
        self._wake.clear()
        return []

    def close(self) -> None:
        self._wake.set()
        self._map.clear()


class SandboxSelectorEventLoop(asyncio.SelectorEventLoop):
    """Event loop replacing TCP `socketpair` self-pipe with in-process `threading.Event`."""

    def __init__(self) -> None:
        self._wake_selector = _EventWakeupSelector()
        super().__init__(self._wake_selector)

    def _make_self_pipe(self) -> None:
        self._ssock = None  # type: ignore[assignment]
        self._csock = None  # type: ignore[assignment]

    def _close_self_pipe(self) -> None:
        self._wake_selector.wakeup()

    def _write_to_self(self) -> None:
        self._wake_selector.wakeup()


class SandboxEventLoopPolicy(asyncio.DefaultEventLoopPolicy):
    """Event loop policy returning `SandboxSelectorEventLoop` when offline on Windows."""

    def new_event_loop(self) -> asyncio.AbstractEventLoop:
        return SandboxSelectorEventLoop()


def ensure_event_loop_policy() -> None:
    """Install socketless event loop policy on Windows when running in offline/sandbox mode."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    if sys.platform == "win32" and not os.environ.get("OPENAI_API_KEY", "").strip():
        asyncio.set_event_loop_policy(SandboxEventLoopPolicy())
