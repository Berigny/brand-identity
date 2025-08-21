import json
from neo4j import GraphDatabase
from neo4j import exceptions as neo4j_exceptions
from dotenv import load_dotenv
import os

# Load environment variables from .env if present
load_dotenv()

def ingest_brand_data(driver):
    # Use either absolute OR relative path - not both
    # Recommended approach (relative path):
    with open('brand_kernel.json') as f:
        data = json.load(f)
    
    # Alternative absolute path (if needed):
    # with open('/Users/davidberigny/Documents/GitHub/brand-identity/brand_kernel.json') as f:
    #     data = json.load(f)
    
    # Ingest brand palette
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    with driver.session(database=database) as session:
        for token_key, token_data in data['brand_palette'].items():
            session.run("""
                MERGE (t:BrandToken {key: $key})
                SET t += {
                    name: $name,
                    hex: $hex,
                    role: $role,
                    prime_id: $prime_id,
                    modality: $modality,
                    mutability_level: $mutability_level
                }
                WITH t
                UNWIND $coherence_links AS link
                MERGE (c:CoreNode {system: split(link, '-')[0], node: split(link, '-')[1]})
                MERGE (t)-[:COHERES_WITH]->(c)
                """,
                key=token_key,
                name=token_data['name'],
                hex=token_data['hex'],
                role=token_data['role'],
                prime_id=token_data['prime_id'],
                modality=token_data['modality'],
                mutability_level=token_data['mutability_level'],
                coherence_links=token_data['coherence_links']
            )
        
        print(f"Ingested {len(data['brand_palette'])} brand tokens")

        # Anti-drift check
        result = session.run("""
            MATCH (t:BrandToken)
            WHERE NOT (t)-[:COHERES_WITH]->()
            RETURN t.key AS orphaned_token
            """)
        orphans = [record["orphaned_token"] for record in result]
        if orphans:
            print(f"Warning: Orphaned tokens found - {orphans}")

if __name__ == "__main__":
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")

    if not uri or not user or not password:
        raise SystemExit("Missing Neo4j config. Set NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD.")

    driver = None
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        ingest_brand_data(driver)
    except (neo4j_exceptions.Neo4jError, neo4j_exceptions.ConfigurationError, neo4j_exceptions.AuthError) as e:
        raise SystemExit(f"Neo4j error: {e}")
    finally:
        if driver:
            driver.close()
