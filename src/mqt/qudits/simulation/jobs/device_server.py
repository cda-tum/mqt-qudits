from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import asyncio
import uuid
from jobstatus import JobStatus, JobStatusError

app = FastAPI()

# Simulating a database with an in-memory dictionary
job_database = {}


class Circuit(BaseModel):
    operations: List[Dict[str, Any]]


class EnergyLevelGraph(BaseModel):
    edges: List[Dict[str, Any]]
    nodes: List[int]


class JobSubmission(BaseModel):
    circuit: Dict[str, Any]
    shots: int
    energy_level_graphs: List[Dict[str, Any]]


class JobResult(BaseModel):
    state_vector: List[complex]
    counts: List[int]


@app.post("/submit_job")
async def submit_job(job: JobSubmission):
    job_id = str(uuid.uuid4())
    job_database[job_id] = {
        "status":     JobStatus.INITIALIZING,
        "submission": job.dict(),
        "result":     None
    }

    # Start job processing
    asyncio.create_task(process_job(job_id))

    return {"job_id": job_id}


@app.get("/job_status/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in job_database:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": job_database[job_id]["status"].value}


@app.get("/job_result/{job_id}")
async def get_job_result(job_id: str):
    if job_id not in job_database:
        raise HTTPException(status_code=404, detail="Job not found")
    if job_database[job_id]["status"] != JobStatus.DONE:
        raise HTTPException(status_code=400, detail="Job not completed yet")
    return job_database[job_id]["result"]


async def process_job(job_id: str):
    try:
        # Simulate job processing
        job_database[job_id]["status"] = JobStatus.QUEUED
        await asyncio.sleep(2)  # Simulate queueing time

        job_database[job_id]["status"] = JobStatus.VALIDATING
        await asyncio.sleep(1)  # Simulate validation time

        job_database[job_id]["status"] = JobStatus.RUNNING
        await asyncio.sleep(5)  # Simulate running time

        # Generate mock results
        mock_state_vector = [complex(1, 1), complex(0, 1), complex(-1, 0), complex(0, -1)]
        mock_counts = [10, 15, 12, 13]

        job_database[job_id]["result"] = JobResult(state_vector=mock_state_vector, counts=mock_counts).dict()
        job_database[job_id]["status"] = JobStatus.DONE
    except Exception as e:
        job_database[job_id]["status"] = JobStatus.ERROR
        raise JobStatusError(f"Error processing job: {str(e)}", JobStatus.ERROR)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)