from fastapi import APIRouter
from app.utils.flow_simulator import run_flow_simulation

router = APIRouter()

@router.get("/simulate-flow")
async def simulate_flow():
    return run_flow_simulation()