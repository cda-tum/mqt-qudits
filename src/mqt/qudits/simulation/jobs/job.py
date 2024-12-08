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
        self._status = JobStatus.INITIALIZING
        self._result = None

    @property
    def job_id(self) -> str:
        return self._job_id

    @property
    def backend(self) -> Backend:
        return self._backend

    async def status(self) -> JobStatus:
        if self._api_client:
            self._status = await self._api_client.get_job_status(self._job_id)
        else:
            # For local simulation, we assume the job is done immediately
            self._status = JobStatus.DONE
        return self._status

    async def result(self) -> JobResult:
        if self._result is None:
            await self.wait_for_final_state()
            if self._api_client:
                await self._api_client.get_job_result(self._job_id)
            else:
                # For local simulation, we get the result directly from the backend
                await self._backend.run_local_simulation(self._job_id)
            # self._result = JobResult(self._job_id, result_data["state_vector"], result_data["counts"])
        return self._result

    async def wait_for_final_state(
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
            self._status = JobStatus.DONE
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
