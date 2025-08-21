from neo4j import GraphDatabase
from neo4j import exceptions as neo4j_exceptions
from dotenv import load_dotenv
import os

load_dotenv()

class Neo4jConnection:
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI")
        self.user = os.getenv("NEO4J_USER")
        self.password = os.getenv("NEO4J_PASSWORD")
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")

        if not self.uri or not self.user or not self.password:
            raise RuntimeError(
                "Missing Neo4j configuration. Ensure NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD are set."
            )

        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        except (neo4j_exceptions.ConfigurationError, neo4j_exceptions.AuthError) as e:
            raise RuntimeError(f"Failed to initialize Neo4j driver: {e}")
    
    def execute_query(self, query, parameters=None):
        params = parameters or {}
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, params)
                return [record.data() for record in result]
        except neo4j_exceptions.Neo4jError as e:
            raise RuntimeError(f"Neo4j query failed: {e}")
    
    def close(self):
        self.driver.close()

db = Neo4jConnection()
