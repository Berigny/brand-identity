import os
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph

load_dotenv()
load_dotenv(".env.local")

def _make_graph() -> Neo4jGraph:
    uri = os.getenv("NEO4J_URI") or "bolt://localhost:7687"
    user = os.getenv("NEO4J_USER")
    pwd = os.getenv("NEO4J_PASSWORD")
    db = os.getenv("NEO4J_DATABASE", "neo4j")
    try:
        return Neo4jGraph(url=uri, username=user, password=pwd, database=db)
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")
        raise

graph = _make_graph()

# Non-destructive flip to a new property
query = """
MATCH (n:Rule) WHERE n.prime IS NOT NULL
SET n.prime_v2 = 1.0 - toFloat(n.prime)
"""

try:
    graph.query(query)
    print("Successfully added 'prime_v2' property to Rule nodes.")
except Exception as e:
    print(f"An error occurred during Neo4j migration: {e}")
