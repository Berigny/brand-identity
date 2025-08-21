from fastapi import APIRouter, Query
from app.utils.flow_simulator import run_flow_simulation
from app.database import db
from datetime import datetime

router = APIRouter()

@router.get("/simulate-flow")
async def simulate_flow(commit: bool = Query(False, description="Persist summary metrics to Neo4j")):
    summary = run_flow_simulation()
    if commit:
        try:
            # Aggregate metrics node
            db.execute_query(
                """
                MERGE (m:SimMetrics {id: 'default'})
                SET m.total_runs = coalesce(m.total_runs, 0) + 1,
                    m.last_completed = $completed,
                    m.last_confidence = $confidence,
                    m.last_steps = $steps,
                    m.last_status = $status
                """,
                {
                    "completed": bool(summary.get("completed", False)),
                    "confidence": float(summary.get("final_confidence", 0.0)),
                    "steps": int(summary.get("total_steps", 0)),
                    "status": summary.get("status", "unknown"),
                },
            )
            # Time-series entry
            db.execute_query(
                """
                CREATE (r:SimRun {
                    ts: $ts,
                    completed: $completed,
                    confidence: $confidence,
                    steps: $steps,
                    status: $status,
                    coherence_score: $coherence
                })
                """,
                {
                    "ts": datetime.utcnow().isoformat(),
                    "completed": bool(summary.get("completed", False)),
                    "confidence": float(summary.get("final_confidence", 0.0)),
                    "steps": int(summary.get("total_steps", 0)),
                    "status": summary.get("status", "unknown"),
                    "coherence": float(summary.get("coherence_score", 0.0)),
                },
            )
        except Exception as e:
            # Non-fatal for the endpoint
            summary["persist_error"] = str(e)
    return summary
