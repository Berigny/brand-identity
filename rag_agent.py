from langchain import LLMChain, PromptTemplate
from langchain_community.graphs import Neo4jGraph
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import StructuredTool
from langchain_openai import OpenAI  # Replace with Grok API if preferred
from dotenv import load_dotenv
import networkx as nx
import random
import os
import json

load_dotenv()

# Neo4j connection
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USER"),
    password=os.getenv("NEO4J_PASSWORD")
)

# Continuous Flow Simulation with Genesis Topology (0-9 days, 12 tribes)
def run_flow_simulation(start_node="system_1-N0"):
    G = nx.DiGraph()
    nodes = {
        'blueprint-Centroid': 'Blueprint / ∞ (Day 0)',
        'system_1-N0': 'Possibility / Novelty (Day 1)',
        'system_1-N1': 'Light / Shadow (Day 2)',
        'system_1-N2': 'Separation / Structure (Day 3)',
        'system_1-N3': 'Emergence / Life (Day 9)',
        'system_2-N0': 'Time / Sequence (Day 8)',
        'system_2-N1': 'Motion / Vitality (Day 4)',
        'system_2-N2': 'Agency / Image (Day 5)',
        'system_2-N3': 'Coherence / Rest (Day 7)',
        'tribal_grid-12': '12 Tribes (VE 12)'
    }
    for node, label in nodes.items():
        G.add_node(node, label=label, activations=0)
    G.add_edges_from([
        ('system_1-N0', 'blueprint-Centroid'), ('system_2-N0', 'blueprint-Centroid'),
        ('blueprint-Centroid', 'system_1-N1'), ('blueprint-Centroid', 'system_2-N1'),
        ('system_1-N1', 'system_1-N2'), ('system_2-N1', 'system_2-N2'),
        ('system_1-N2', 'system_1-N3'), ('system_2-N2', 'system_2-N3'),
        ('system_1-N3', 'system_2-N0'), ('system_2-N3', 'system_1-N0'),
        ('system_1-N0', 'tribal_grid-12'), ('system_2-N2', 'tribal_grid-12')
    ])

    activations = {node: 0 for node in nodes}
    day_activations = {i: 0 for i in range(10)}  # 0-9 days
    day_activations[12] = 0  # Tribal grid

    current_node = start_node
    activations[current_node] += 1
    path = [current_node]

    while not all(v >= 2 for v in activations.values()):
        sys, node = current_node.split('-') if '-' in current_node else ('blueprint', 'Centroid')
        day = {'blueprint-Centroid': 0, 'system_1-N0': 1, 'system_1-N1': 2, 'system_1-N2': 3,
               'system_2-N1': 4, 'system_2-N2': 5, 'system_2-N3': 7, 'system_2-N0': 8,
               'system_1-N3': 9, 'tribal_grid-12': 12}.get(current_node, None)
        if day is not None:
            day_activations[day] += 1
            graph.query("MATCH (d:Day {id: $day}) SET d.activations = coalesce(d.activations, 0) + 1", {'day': day})

        if node == 'N0':
            next_node = 'blueprint-Centroid'
        elif node in ['N1', 'N3']:
            next_node = f"{sys}-N{int(node[1:]) + 1}" if node == 'N1' else ('system_2-N0' if sys == 'system_1' else 'system_1-N0')
        elif current_node == 'blueprint-Centroid':
            next_node = random.choice(['system_1-N1', 'system_2-N1'])
        else:
            next_node = f"{sys}-N3"

        current_node = next_node
        activations[current_node] += 1
        path.append(current_node)

    return {"path": path, "activations": activations, "day_activations": day_activations}

# Tool 1: Retrieve External Rules
def fetch_external_rules(query: str) -> str:
    # Mock; replace with x_keyword_search or web API
    return "Guideline: Tertiary colors should be vibrant, emotional (e.g., #ff6699), mutable for trends. Align with emotional outputs (52, 65)."

external_tool = StructuredTool.from_function(
    func=fetch_external_rules,
    name="FetchExternalRules",
    description="Fetch brand guidelines from external sources."
)

# Tool 2: Write to DB
def write_to_db(content: str) -> str:
    try:
        new_content = json.loads(content)
        if "brand_palette" in new_content:
            for key, token in new_content["brand_palette"].items():
                query = """
                MERGE (t:BrandToken {key: $key})
                SET t += $props
                WITH t
                FOREACH (r_id IN $resonance_outputs |
                    MERGE (r:Resonance {id: r_id})
                    MERGE (t)-[:RESONATES_WITH]->(r))
                FOREACH (c_link IN $coherence_links |
                    MERGE (c:CoreNode {system: split(c_link, '-')[0], node: split(c_link, '-')[1]})
                    MERGE (t)-[:COHERES_WITH]->(c))
                """
                graph.query(query, {
                    "key": key,
                    "props": {k: v for k, v in token.items() if k not in ["relationships", "mutability_level", "resonance_outputs", "coherence_links"]},
                    "mutability_level": token.get("mutability_level", "flexible"),
                    "resonance_outputs": token.get("resonance_outputs", []),
                    "coherence_links": token.get("coherence_links", [])
                })
        return f"Successfully wrote {content} to DB."
    except Exception as e:
        return f"Error: {str(e)}"

write_tool = StructuredTool.from_function(
    func=write_to_db,
    name="WriteToDB",
    description="Write proposed brand content to Neo4j."
)

# Tool 3: Test for Drift
def test_drift(content: str) -> str:
    simulation = run_flow_simulation()
    result = graph.query("""
    MATCH (t:BrandToken)-[:RESONATES_WITH]->(r:Resonance)
    WHERE NOT r.id IN [2, 12, 24, 42, 480]
    RETURN t.key, r.id AS misaligned_resonance
    UNION
    MATCH (t:BrandToken)-[:COHERES_WITH]->(c:CoreNode)
    WHERE coalesce(c.activations, 0) < 2
    RETURN t.key, c.system + '-' + c.node AS under_activated
    """)
    return f"Simulation: {simulation['activations']}\nDrift issues: {result}"

drift_tool = StructuredTool.from_function(
    func=test_drift,
    name="TestDrift",
    description="Test for drift using flow simulation and queries."
)

# Tool 4: Refine Rules
llm = OpenAI(openai_api_key=os.getenv("GROK_API_KEY"), base_url="https://api.x.ai/v1", model="grok-1")

prompt = PromptTemplate(
    input_variables=["test_results", "proposed_content"],
    template="Refine {proposed_content} based on drift test: {test_results}. Ensure alignment with VE phases (2, 12, 24, 42, 480) and modal_domains.emotional.outputs (e.g., 52)."
)

refine_chain = LLMChain(llm=llm, prompt=prompt)

def refine_rules(test_results: str, proposed_content: str) -> str:
    return refine_chain.run({"test_results": test_results, "proposed_content": proposed_content})

refine_tool = StructuredTool.from_function(
    func=refine_rules,
    name="RefineRules",
    description="Refine proposed content for higher consilience."
)

# Agent Setup
tools = [external_tool, write_tool, drift_tool, refine_tool]

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=PromptTemplate.from_template(
        "You are a RAG agent building a brand identity. Use tools to layer external rules, write to DB, test for drift, and refine for consilience with VE phases (2, 12, 24, 42, 480) and core nodes."
    )
)

executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run the agent with a prompt
if __name__ == "__main__":
    import sys
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Add a new tertiary emotional token with high prime, test for drift, and refine for consilience."
    response = executor.run(prompt)
    print(response)
print(graph.query("MATCH (n:BrandToken) RETURN n.key LIMIT 5"))