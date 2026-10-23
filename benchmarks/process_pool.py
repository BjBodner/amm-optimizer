"""A process pool for deterministic tasks that survives a broken pool.

On this Windows machine, under heavy load, a pool worker now and then loses the call
queue's semaphore handle. The parent duplicates the handle into the new worker at
creation; the handle is still valid after unpickling and after multiprocessing's
``_close_stdin``, but it is invalid by the worker's first ``call_queue.get``. No
Python code in the worker closes it, and a profiler or an extra import early in the
worker hides it, so the closer is native and timing-bound. Its rate rises with the
work the worker does at start under load: a 2-worker pool whose initializer imports
SciPy and builds a cocoex Suite (``quick_scan``'s) lost a worker in 7-11 of 30 runs at
100% CPU; SciPy alone, the Suite alone or no initializer, in 0 of 30.

Two failure modes follow (lesson 23):

* ``Queue.get``'s ``self._sem.release()`` raises ``OSError: [WinError 6] The handle
  is invalid``, the worker dies, and the executor fails every pending task with
  ``BrokenProcessPool``. ``RetryingPool.map`` replaces the pool and reruns the tasks
  that had no result. That changes no result: each task here (one bbob run) builds
  its own problem and random streams, so it returns the same value in any worker, as
  ``--jobs 1`` shows.
* When the handle value is reused by another semaphore, the release succeeds on that
  one and the call queue leaks a slot. At shutdown CPython's ``shutdown_workers`` then
  retries ``put_nowait(None)`` in a busy loop while a worker lives, and
  ``shutdown(wait=True)`` never returns (once in about 400 such pools, after every
  result was in). ``RetryingPool.close`` gives the workers ``grace`` seconds to exit,
  then terminates them.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Any, TypeVar

T = TypeVar("T")
R = TypeVar("R")

RETRIES = 3  # fresh pools per ``map`` call before the break is raised
GRACE = 30.0  # seconds the workers get to exit on ``close`` before they are terminated


class RetryingPool:
    """``ProcessPoolExecutor`` reused across ``map`` calls, replaced when it breaks.

    The pool starts on the first ``map``; ``close`` (or leaving a ``with`` block)
    shuts it down. ``restarts`` counts the broken pools replaced so far.
    """

    def __init__(
        self,
        jobs: int,
        initializer: Callable[..., object] | None = None,
        initargs: tuple[Any, ...] = (),
        retries: int = RETRIES,
        grace: float = GRACE,
    ) -> None:
        self.jobs = jobs
        self.initializer = initializer
        self.initargs = initargs
        self.retries = retries
        self.grace = grace
        self.restarts = 0
        self._pool: ProcessPoolExecutor | None = None

    def map(self, fn: Callable[[T], R], tasks: Iterable[T]) -> Iterator[R]:
        """``fn(task)`` for every task, in task order, one task per call."""
        todo = list(tasks)
        done = 0
        for attempt in range(self.retries + 1):
            if self._pool is None:
                self._pool = ProcessPoolExecutor(
                    self.jobs, initializer=self.initializer, initargs=self.initargs
                )
            try:
                for result in self._pool.map(fn, todo[done:], chunksize=1):
                    done += 1
                    yield result
                return
            except BrokenProcessPool as error:
                self.close()
                if attempt == self.retries:
                    raise
                self.restarts += 1
                print(
                    f"process pool broke ({error}); a fresh pool reruns the "
                    f"{len(todo) - done} tasks without a result "
                    f"(retry {attempt + 1} of {self.retries})",
                    file=sys.stderr,
                    flush=True,
                )

    def close(self) -> None:
        """Shut the pool down; terminate workers still alive after ``grace`` s."""
        pool, self._pool = self._pool, None
        if pool is None:
            return
        # ProcessPoolExecutor has no public accessor for these; shutdown drops them
        workers = list((pool._processes or {}).values())
        manager = pool._executor_manager_thread
        pool.shutdown(wait=False, cancel_futures=True)
        deadline = time.monotonic() + self.grace
        for worker in workers:
            worker.join(max(deadline - time.monotonic(), 0.0))
            if worker.exitcode is None:
                worker.terminate()
                worker.join()
        if manager is not None:
            manager.join()

    def __enter__(self) -> RetryingPool:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
