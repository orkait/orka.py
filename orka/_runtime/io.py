"""Async disk I/O: a single background-writer thread that drains a bounded queue of
(fn, args) write tasks so the pack pipeline never blocks on disk. ``_BG_WRITER`` is the
process-wide singleton; ``.wait()`` flushes and surfaces any deferred write errors.
"""

from __future__ import annotations

import queue
import threading


class BackgroundWriter:
    """Worker thread starts on the first ``submit``. ``_BG_WRITER`` is a module-level
    singleton, so starting in __init__ put a thread in every ``import orka``."""

    def __init__(self):
        self.queue = queue.Queue(maxsize=128)
        self.errors: list[tuple[str, str]] = []
        self.thread: threading.Thread | None = None
        self._start_lock = threading.Lock()

    def _ensure_started(self) -> None:
        if self.thread is not None:
            return
        with self._start_lock:
            if self.thread is None:
                thread = threading.Thread(target=self._worker, daemon=True)
                thread.start()
                self.thread = thread

    def _worker(self):
        while True:
            task = self.queue.get()
            if task is None:
                self.queue.task_done()
                break
            fn, args = task
            try:
                fn(*args)
            except Exception as e:
                self.errors.append((fn.__name__, repr(e)))
            finally:
                self.queue.task_done()

    def submit(self, fn, *args):
        self._ensure_started()
        self.queue.put((fn, args))

    def wait(self):
        self.queue.join()
        if self.errors:
            detail = "; ".join(f"{name}: {err}" for name, err in self.errors)
            raise RuntimeError(
                f"background writes failed ({len(self.errors)} error(s)): {detail}"
            )

    def stop(self):
        if self.thread is None:
            return
        self.queue.put(None)
        if self.thread.is_alive():
            self.thread.join()


_BG_WRITER = BackgroundWriter()
