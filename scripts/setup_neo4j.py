from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

# Load environment from .env and .env.local (dev)
load_dotenv()
load_dotenv(".env.local")

"""
One-shot Neo4j setup script to create constraints and seed base nodes.

Reads env vars: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE.
"""

CONSTRAINTS = [
    "CREATE CONSTRAINT brandtoken_key IF NOT EXISTS FOR (t:BrandToken) REQUIRE t.key IS UNIQUE",
    "CREATE CONSTRAINT core_node_key IF NOT EXISTS FOR (c:CoreNode) REQUIRE (c.system, c.node) IS UNIQUE",
    "CREATE CONSTRAINT resonance_id IF NOT EXISTS FOR (r:Resonance) REQUIRE r.id IS UNIQUE",
    "CREATE CONSTRAINT day_id IF NOT EXISTS FOR (d:Day) REQUIRE d.id IS UNIQUE",
    # New entities
    "CREATE CONSTRAINT brand_id IF NOT EXISTS FOR (b:Brand) REQUIRE b.id IS UNIQUE",
    "CREATE CONSTRAINT rule_id IF NOT EXISTS FOR (r:DesignRule) REQUIRE r.id IS UNIQUE",
    "CREATE CONSTRAINT palette_name IF NOT EXISTS FOR (p:Palette) REQUIRE p.name IS UNIQUE",
    "CREATE CONSTRAINT article_slug IF NOT EXISTS FOR (a:Article) REQUIRE a.slug IS UNIQUE",
    "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
]

SEEDS = [
    # Core nodes across System 1 and System 2
    """
    UNWIND [['system_1','N0'],['system_1','N1'],['system_1','N2'],['system_1','N3'],
            ['system_2','N0'],['system_2','N1'],['system_2','N2'],['system_2','N3']] AS p
    MERGE (:CoreNode {system:p[0], node:p[1]})
    """,
    # Days 0..12
    "UNWIND range(0,12) AS d MERGE (:Day {id:d})",
    # Resonance whitelist
    "UNWIND [2,12,24,42,480] AS r MERGE (:Resonance {id:r})",
    # Default brand node (id: 'default')
    "MERGE (:Brand {id:'default'})",
]


def run_setup():
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE", "neo4j")

    if not uri or not user or not password:
        raise SystemExit("Missing Neo4j config. Set NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD.")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session(database=database) as session:
        for c in CONSTRAINTS:
            session.run(c)
        for s in SEEDS:
            session.run(s)
    driver.close()
    print("Neo4j constraints and seeds applied.")


if __name__ == "__main__":
    run_setup()
