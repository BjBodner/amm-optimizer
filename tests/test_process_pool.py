"""The harness's process pool (benchmarks/process_pool.py) survives a broken pool.

A worker that dies (here ``os._exit``; on this machine, under heavy load, ``WinError
6`` in its call queue) breaks a ``ProcessPoolExecutor`` and fails every pending task.
``RetryingPool.map`` replaces the pool and reruns only the tasks without a result, and
``close`` returns even when CPython's shutdown cannot send its stop sentinels.
"""

import importlib.util
import os
import time
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "process_pool", ROOT / "benchmarks" / "process_pool.py"
)
process_pool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(process_pool)


def square_or_die_once(task):
    """``i * i``; the first worker to get ``i == 3`` dies instead."""
    i, marker = task
    if i == 3 and not os.path.exists(marker):
        Path(marker).touch()
        os._exit(1)
    return i * i


def die(_):
    os._exit(1)


def square(i):
    return i * i


def test_a_broken_pool_is_replaced_and_every_result_comes_back_in_order(tmp_path):
    marker = str(tmp_path / "died")
    with process_pool.RetryingPool(2) as pool:
        got = list(pool.map(square_or_die_once, [(i, marker) for i in range(8)]))
        assert got == [i * i for i in range(8)]
        assert pool.restarts == 1
        first = pool._pool
        assert list(pool.map(square_or_die_once, [(5, marker)])) == [25]
        assert pool._pool is first  # one pool serves every map call
    assert pool._pool is None


def test_a_pool_that_keeps_breaking_raises_after_its_retries(capsys):
    with process_pool.RetryingPool(1, retries=1) as pool:
        with pytest.raises(BrokenProcessPool):
            list(pool.map(die, range(3)))
        assert pool.restarts == 1
    assert "a fresh pool reruns the 3 tasks without a result" in capsys.readouterr().err


def test_close_returns_when_the_shutdown_cannot_send_its_sentinels():
    # a worker whose queue semaphore went bad leaks the call queue's slots; then
    # shutdown_workers busy-loops on put_nowait(None) and shutdown(wait=True) hangs
    pool = process_pool.RetryingPool(1, grace=1.0)
    assert list(pool.map(square, [3])) == [9]
    worker = next(iter(pool._pool._processes.values()))
    sem = pool._pool._call_queue._sem
    while sem.acquire(block=False):  # no slot left for a sentinel
        pass
    started = time.monotonic()
    pool.close()
    assert time.monotonic() - started < 20
    assert worker.exitcode is not None  # terminated after the grace period
