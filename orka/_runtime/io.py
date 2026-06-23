"""Async disk I/O: a single background-writer thread that drains a bounded queue of
(fn, args) write tasks so the pack pipeline never blocks on disk. ``_BG_WRITER`` is the
process-wide singleton; ``.wait()`` flushes and surfaces any deferred write errors.
"""

from __future__ import annotations

import queue
import threading


class BackgroundWriter:
    def __init__(self):
        self.queue = queue.Queue(maxsize=128)
        self.errors: list[tuple[str, str]] = []
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

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
        self.queue.put((fn, args))

    def wait(self):
        self.queue.join()
        if self.errors:
            detail = "; ".join(f"{name}: {err}" for name, err in self.errors)
            raise RuntimeError(
                f"background writes failed ({len(self.errors)} error(s)): {detail}"
            )

    def stop(self):
        self.queue.put(None)
        if self.thread.is_alive():
            self.thread.join()


_BG_WRITER = BackgroundWriter()
