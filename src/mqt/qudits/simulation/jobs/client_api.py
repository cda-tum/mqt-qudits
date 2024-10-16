import aiohttp
import asyncio

from typing import Any, Callable
from mqt.qudits.simulation.jobs.config_api import (
    BASE_URL, SUBMIT_JOB_ENDPOINT, JOB_STATUS_ENDPOINT, JOB_RESULT_ENDPOINT
)
from mqt.qudits.simulation.jobs.jobstatus import JobStatus


class APIClient:
    def __init__(self):
        self.session = aiohttp.ClientSession()

    async def close(self):
        await self.session.close()

    async def submit_job(self, circuit, shots, energy_level_graphs):
        url = f"{BASE_URL}{SUBMIT_JOB_ENDPOINT}"
        payload = {
            "circuit":             circuit.to_dict(),
            "shots":               shots,
            "energy_level_graphs": [graph.to_dict() for graph in energy_level_graphs]
        }
        async with self.session.post(url, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                return data.get('job_id')
            else:
                raise Exception(f"Job submission failed with status code {response.status}")

    async def get_job_status(self, job_id: str) -> JobStatus:
        url = f"{BASE_URL}{JOB_STATUS_ENDPOINT}/{job_id}"
        async with self.session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                return JobStatus.from_string(data.get('status'))
            else:
                raise Exception(f"Failed to get job status with status code {response.status}")

    async def get_job_result(self, job_id: str):
        url = f"{BASE_URL}{JOB_RESULT_ENDPOINT}/{job_id}"
        async with self.session.get(url) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(f"Failed to get job result with status code {response.status}")

    async def wait_for_job_completion(self, job_id: str, callback: Callable[[str, JobStatus], None] = None,
                                      polling_interval: float = 5):
        while True:
            status = await self.get_job_status(job_id)
            if callback:
                callback(job_id, status)
            if status.is_final:
                return status
            await asyncio.sleep(polling_interval)
