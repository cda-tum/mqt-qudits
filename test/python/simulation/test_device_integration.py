import unittest
import asyncio
import aiohttp
from fastapi.testclient import TestClient
from multiprocessing import Process
import uvicorn
import time

from mqt.qudits.simulation import MQTQuditProvider
from mqt.qudits.simulation.jobs import JobStatus
from mqt.qudits.simulation.jobs.client_api import APIClient
from mqt.qudits.simulation.jobs.device_server import app


def run_server():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="critical")


class TestQuantumSystemIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Start the server in a separate process
        cls.server_process = Process(target=run_server)
        cls.server_process.start()
        time.sleep(1)  # Give the server a moment to start

    @classmethod
    def tearDownClass(cls):
        # Shut down the server
        cls.server_process.terminate()
        cls.server_process.join()

    def setUp(self):
        self.provider = MQTQuditProvider()
        self.backend = self.provider.get_backend("innsbruck01")
        self.api_client = APIClient()

    def tearDown(self):
        asyncio.run(self.api_client.close())

    def test_full_job_lifecycle(self):
        async def run_test():
            # Create a simple quantum circuit
            circuit = {
                "operations": [
                    {"gate": "H", "qubit": 0},
                    {"gate": "CNOT", "control": 0, "target": 1}
                ]
            }

            # Submit the job
            job = await self.backend.run(circuit, shots=1000)
            self.assertIsNotNone(job.job_id)

            # Check initial status
            status = await job.status()
            self.assertIn(status, list(JobStatus))

            # Wait for the job to complete
            await job.wait_for_final_state(timeout=30)

            # Check final status
            final_status = await job.status()
            self.assertEqual(final_status, JobStatus.DONE)

            # Get results
            result = await job.result()
            self.assertIsNotNone(result)
            self.assertTrue(hasattr(result, 'get_counts'))
            self.assertTrue(hasattr(result, 'get_state_vector'))

            counts = result.get_counts()
            self.assertIsInstance(counts, list)
            self.assertEqual(sum(counts), 1000)  # Total should match our shots

            state_vector = result.get_state_vector()
            self.assertIsInstance(state_vector, list)
            self.assertEqual(len(state_vector), 4)  # 2^2 for 2 qubits

        asyncio.run(run_test())

    def test_multiple_concurrent_jobs(self):
        async def run_concurrent_jobs():
            circuit = {"operations": [{"gate": "H", "qubit": 0}]}
            jobs = []
            for _ in range(5):
                job = await self.backend.run(circuit, shots=100)
                jobs.append(job)

            results = await asyncio.gather(*(job.wait_for_final_state(timeout=30) for job in jobs))
            self.assertEqual(len(results), 5)

            for job in jobs:
                status = await job.status()
                self.assertEqual(status, JobStatus.DONE)

                result = await job.result()
                self.assertIsNotNone(result)

        asyncio.run(run_concurrent_jobs())

    def test_error_handling(self):
        async def run_error_test():
            # Try to get status of non-existent job
            with self.assertRaises(Exception):
                await self.api_client.get_job_status("non_existent_job_id")

            # Try to get result of non-existent job
            with self.assertRaises(Exception):
                await self.api_client.get_job_result("non_existent_job_id")

            # Submit invalid job
            invalid_circuit = {"operations": [{"gate": "InvalidGate", "qubit": 0}]}
            with self.assertRaises(Exception):
                await self.backend.run(invalid_circuit, shots=100)

        asyncio.run(run_error_test())


