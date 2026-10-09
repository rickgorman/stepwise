"""Concurrent fulfill_watch calls on one external gate.

Engine.fulfill_watch loads the run, checks that it is SUSPENDED, and later
saves it as COMPLETED, with no lock or compare-and-set between the check and
the write. POST /api/runs/{run_id}/fulfill is a sync FastAPI route, so
concurrent requests run fulfill_watch on separate threadpool threads at the
same time. A sequential second fulfill is rejected with already_fulfilled;
a concurrent one must be rejected the same way.
"""

import asyncio
import threading

import pytest

from stepwise.engine import AsyncEngine
from stepwise.events import JOB_COMPLETED, WATCH_FULFILLED
from stepwise.models import (
    ExecutorRef,
    JobStatus,
    StepDefinition,
    StepRunStatus,
    WorkflowDefinition,
)
from stepwise.server import ThreadSafeStore


@pytest.fixture
def server_engine(tmp_path, registry):
    """AsyncEngine on the server's per-thread-connection store.

    The in-memory SQLiteStore shares one connection, which is not safe for
    the cross-thread calls this test makes.
    """
    store = ThreadSafeStore(str(tmp_path / "stepwise.db"))
    yield AsyncEngine(store=store, registry=registry)
    store.close()


def _gate_wf() -> WorkflowDefinition:
    return WorkflowDefinition(steps={
        "approve": StepDefinition(
            name="approve",
            executor=ExecutorRef(type="external", config={"prompt": "Approve?"}),
            outputs=["decision"],
        ),
    })


def test_concurrent_fulfills_accept_exactly_one(server_engine, monkeypatch):
    """Both callers read the gate as SUSPENDED before either one writes.

    The barrier in load_run holds each caller after its read until the other
    caller has also read, so the race fires on every run instead of by luck.
    """
    engine = server_engine
    payloads = [{"decision": "approve"}, {"decision": "reject"}]

    async def scenario():
        engine_task = asyncio.create_task(engine.run())
        try:
            job = engine.create_job(objective="t", workflow=_gate_wf())
            engine.start_job(job.id)
            for _ in range(100):
                suspended = engine.store.suspended_runs(job.id)
                if suspended:
                    break
                await asyncio.sleep(0.01)
            assert len(suspended) == 1
            gate_id = suspended[0].id

            both_read = threading.Barrier(2, timeout=5)
            real_load_run = engine.store.load_run

            def load_run_then_rendezvous(run_id):
                run = real_load_run(run_id)
                if run_id == gate_id and run.status == StepRunStatus.SUSPENDED:
                    try:
                        both_read.wait()
                    except threading.BrokenBarrierError:
                        # A fix that serializes the whole check-and-write
                        # never lets the second caller read SUSPENDED.
                        pass
                return run

            monkeypatch.setattr(engine.store, "load_run", load_run_then_rendezvous)
            results = await asyncio.gather(*(
                asyncio.to_thread(engine.fulfill_watch, gate_id, payload)
                for payload in payloads
            ))

            accepted = [p for p, r in zip(payloads, results) if r is None]
            rejected = [r for r in results if r is not None]
            assert len(accepted) == 1, (
                f"{len(accepted)} concurrent fulfills accepted for one gate"
            )
            assert rejected[0]["error"] == "already_fulfilled"
            assert rejected[0]["run_id"] == gate_id

            gate = real_load_run(gate_id)
            assert gate.status == StepRunStatus.COMPLETED
            assert gate.result.artifact == accepted[0]

            job = await asyncio.wait_for(engine.wait_for_job(job.id), 5)
            assert job.status == JobStatus.COMPLETED

            events = engine.store.load_events(job.id)
            fulfilled = [e for e in events if e.type == WATCH_FULFILLED]
            assert [e.data["payload"] for e in fulfilled] == accepted
            assert [e.type for e in events].count(JOB_COMPLETED) == 1
        finally:
            engine_task.cancel()
            try:
                await engine_task
            except asyncio.CancelledError:
                pass

    asyncio.run(scenario())
