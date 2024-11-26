from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Callable

from .jobstatus import JobStatus

if TYPE_CHECKING:
    from ..backends.backendv2 import Backend
    from . import JobResult
    from .client_api import APIClient


class Job:
    def __init__(self, backend: Backend, job_id: str = "local_sim", api_client: APIClient | None = None) -> None:
        self._backend = backend
        self._job_id = job_id
        self._api_client = api_client
        self.set_status(JobStatus.INITIALIZING)
        self._result = None

    @property
    def job_id(self) -> str:
        return self._job_id

    @property
    def backend(self) -> Backend:
        return self._backend

    async def status(self) -> JobStatus:
        if self._api_client:
            self.set_status(await self._api_client.get_job_status(self._job_id))
        else:
            # For local simulation, we assume the job is done immediately
            self.set_status(JobStatus.DONE)
        return self._status

    def result(self) -> JobResult:
        if self._result is not None:
            return self._result

        # Handle the async operations in a separate method
        async def _get_result():
            await self._wait_for_final_state()
            if self._api_client:
                self._result = await self._api_client.get_job_result(self._job_id)
            else:
                # For local simulation, we get the result directly from the backend
                self._result = await self._backend.run_local_simulation(self._job_id)
            return self._result

        # Run the async operations in the event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're inside an event loop, create a task
                return loop.run_until_complete(_get_result())
            else:
                # No loop is running, use this one
                return loop.run_until_complete(_get_result())
        except RuntimeError:
            # No event loop exists, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(_get_result())
            finally:
                loop.close()

    async def _wait_for_final_state(
            self, timeout: float | None = None, callback: Callable[[str, JobStatus], None] | None = None
    ) -> None:
        if self._api_client:
            try:
                await asyncio.wait_for(self._api_client.wait_for_job_completion(self._job_id, callback), timeout)
            except asyncio.TimeoutError:
                msg = f"Timeout while waiting for job {self._job_id}"
                raise TimeoutError(msg)
        else:
            # For local simulation, we assume the job is done immediately
            self.set_status(JobStatus.DONE)
            if callback:
                callback(self._job_id, self._status)

    def cancelled(self) -> bool:
        return self._status == JobStatus.CANCELLED

    def done(self) -> bool:
        return self._status == JobStatus.DONE

    def running(self) -> bool:
        return self._status == JobStatus.RUNNING

    def in_final_state(self) -> bool:
        return self._status.is_final

    def set_result(self, result: JobResult) -> None:
        self._result = result

    def set_status(self, new_status: JobStatus) -> None:
        self._status = new_status
