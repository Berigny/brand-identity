from fastapi import APIRouter
from app.database import db

router = APIRouter()

@router.get("/anti-drift")
async def check_drift():
    # Get under-activated nodes
    result = db.execute_query("""
    MATCH (t:BrandToken)-[:COHERES_WITH]->(c:CoreNode)
    WHERE c.activations < 2
    RETURN t.key AS token_key, 
           c.system + '-' + c.node AS node,
           'Under-activated node' AS issue
    """)
    
    # Check resonance mismatches
    resonance_check = db.execute_query("""
    MATCH (t:BrandToken)-[:RESONATES_WITH]->(r:Resonance)
    WHERE NOT r.id IN [2, 12, 24, 42, 480]
    RETURN t.key AS token_key, 
           r.id AS resonance_id,
           'Invalid resonance' AS issue
    """)
    
    return {
        "drift_alerts": result + resonance_check
    }