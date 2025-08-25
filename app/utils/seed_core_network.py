"""
Seed the Core Network topology (S1/S2, IC, EC) into Neo4j.

This mirrors the architecture in app/utils/flow_simulator.py so Cypher can
reason over the same graph (routes, weights, kinds).

Usage:
  python app/utils/seed_core_network.py

Env vars (same as rag_agent):
  - NEO4J_URI (default: bolt://localhost:7687)
  - NEO4J_USER, NEO4J_PASSWORD
  - NEO4J_DATABASE (default: neo4j)
"""

from __future__ import annotations

import os
from typing import List, Dict, Any

from langchain_neo4j import Neo4jGraph


def _make_graph() -> Neo4jGraph:
    uri = os.getenv("NEO4J_URI") or "bolt://localhost:7687"
    user = os.getenv("NEO4J_USER")
    pwd = os.getenv("NEO4J_PASSWORD")
    db = os.getenv("NEO4J_DATABASE", "neo4j")
    try:
        return Neo4jGraph(url=uri, username=user, password=pwd, database=db)
    except Exception as e:
        # Fallback to default database if specific DB not found
        msg = str(e)
        if ("DatabaseNotFound" in msg or "does not exist" in msg) and db != "neo4j":
            return Neo4jGraph(url=uri, username=user, password=pwd, database="neo4j")
        raise


def _nodes_spec() -> List[Dict[str, Any]]:
    # S1 (0..3)
    s1 = [
        {"system": "S1", "node": "0", "idx": 0, "role": "Compression", "prime": 2},
        {"system": "S1", "node": "1", "idx": 1, "role": "Expression", "prime": 3},
        {"system": "S1", "node": "2", "idx": 2, "role": "Stabilisation", "prime": 5},
        {"system": "S1", "node": "3", "idx": 3, "role": "Emission", "prime": 7},
    ]
    # S2 (4..7)
    s2 = [
        {"system": "S2", "node": "4", "idx": 4, "role": "Compression", "prime": 11},
        {"system": "S2", "node": "5", "idx": 5, "role": "Expression", "prime": 13},
        {"system": "S2", "node": "6", "idx": 6, "role": "Stabilisation", "prime": 17},
        {"system": "S2", "node": "7", "idx": 7, "role": "Emission", "prime": 19},
    ]
    # Central channels
    central = [
        {"system": "C", "node": "IC", "idx": None, "role": "InternalC", "prime": 23},
        {"system": "C", "node": "EC", "idx": None, "role": "ExternalC", "prime": 29},
    ]

    all_nodes = s1 + s2 + central
    for n in all_nodes:
        n["key"] = f"{n['system']}-{n['node']}"
    return all_nodes


def _edges_spec() -> List[Dict[str, Any]]:
    def k(sys: str, node: str) -> str:
        return f"{sys}-{node}"

    edges: List[Dict[str, Any]] = []

    # Tetrahedron 1 (0,1,2,3)
    t1 = [("0", "1"), ("0", "2"), ("0", "3"), ("1", "2"), ("1", "3"), ("2", "3")]
    edges += [{"from": k("S1", a), "to": k("S1", b), "kind": "tetrahedral", "weight": 0.7} for a, b in t1]
    edges += [
        {"from": k("S1", "0"), "to": k("C", "IC"), "kind": "cubic", "weight": 1.0},
        {"from": k("C", "IC"), "to": k("S1", "1"), "kind": "cubic", "weight": 0.9},
        {"from": k("S1", "2"), "to": k("C", "IC"), "kind": "cubic", "weight": 0.8},
        {"from": k("C", "IC"), "to": k("S1", "3"), "kind": "cubic", "weight": 1.0},
        {"from": k("S1", "3"), "to": k("S1", "0"), "kind": "cubic", "weight": 0.6},  # loopback
    ]

    # Tetrahedron 2 (4,5,6,7)
    t2 = [("4", "5"), ("4", "6"), ("4", "7"), ("5", "6"), ("5", "7"), ("6", "7")]
    edges += [{"from": k("S2", a), "to": k("S2", b), "kind": "tetrahedral", "weight": 0.7} for a, b in t2]
    edges += [
        {"from": k("S2", "4"), "to": k("C", "IC"), "kind": "cubic", "weight": 1.0},
        {"from": k("C", "IC"), "to": k("S2", "5"), "kind": "cubic", "weight": 0.9},
        {"from": k("S2", "6"), "to": k("C", "IC"), "kind": "cubic", "weight": 0.8},
        {"from": k("C", "IC"), "to": k("S2", "7"), "kind": "cubic", "weight": 1.0},
        {"from": k("S2", "7"), "to": k("S2", "4"), "kind": "cubic", "weight": 0.6},  # loopback
    ]

    # Cross-tetrahedron cubic edges
    cross = [
        (k("S1", "0"), k("S2", "4")), (k("S1", "1"), k("S2", "5")), (k("S1", "2"), k("S2", "6")), (k("S1", "3"), k("S2", "7")),
        (k("S1", "0"), k("S2", "5")), (k("S1", "0"), k("S2", "6")), (k("S1", "1"), k("S2", "4")), (k("S1", "1"), k("S2", "6")),
        (k("S1", "2"), k("S2", "4")), (k("S1", "2"), k("S2", "5")), (k("S1", "3"), k("S2", "4")), (k("S1", "3"), k("S2", "5")),
    ]
    edges += [{"from": a, "to": b, "kind": "cubic", "weight": 0.8} for a, b in cross]

    # External connections
    edges += [
        {"from": k("C", "EC"), "to": k("C", "IC"), "kind": "cubic", "weight": 1.0},
        {"from": k("C", "EC"), "to": k("S1", "0"), "kind": "cubic", "weight": 0.8},
        {"from": k("C", "EC"), "to": k("S2", "4"), "kind": "cubic", "weight": 0.8},
        {"from": k("S1", "0"), "to": k("C", "EC"), "kind": "failsafe", "weight": 0.5},
        {"from": k("S1", "2"), "to": k("C", "EC"), "kind": "failsafe", "weight": 0.5},
        {"from": k("S2", "4"), "to": k("C", "EC"), "kind": "failsafe", "weight": 0.5},
        {"from": k("S2", "6"), "to": k("C", "EC"), "kind": "failsafe", "weight": 0.5},
    ]

    return edges


def seed_core_network(graph: Neo4jGraph | None = None) -> dict:
    """Seed/Upsert the core nodes and edges. Idempotent via MERGE."""
    g = graph or _make_graph()
    nodes = _nodes_spec()
    edges = _edges_spec()

    g.query(
        """
        UNWIND $nodes AS n
        MERGE (c:CoreNode {system: n.system, node: n.node})
        SET c.key = n.key,
            c.idx = n.idx,
            c.role = n.role,
            c.prime = n.prime
        """,
        {"nodes": nodes},
    )

    g.query(
        """
        UNWIND $edges AS e
        MATCH (a:CoreNode {key: e.from}), (b:CoreNode {key: e.to})
        MERGE (a)-[r:CONNECTED_TO]->(b)
        SET r.kind = e.kind, r.weight = e.weight
        """,
        {"edges": edges},
    )

    # Return small summary
    summary = g.query(
        """
        CALL {
          MATCH (c:CoreNode) RETURN count(c) AS nodes
        }
        CALL {
          MATCH ()-[r:CONNECTED_TO]->() RETURN count(r) AS edges
        }
        RETURN nodes, edges
        """
    )
    return summary[0] if summary else {"nodes": None, "edges": None}


if __name__ == "__main__":
    try:
        out = seed_core_network()
        print(f"Seeded core network: nodes={out.get('nodes')} edges={out.get('edges')}")
        print("Hint: provide coherence_links like 'S1-0', 'S2-5', 'C-IC', 'C-EC'.")
    except Exception as e:
        import traceback
        print(f"Seeding failed: {e}\n\n{traceback.format_exc()}")

