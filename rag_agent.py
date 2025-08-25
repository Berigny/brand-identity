from langchain_core.prompts import PromptTemplate
from langchain_neo4j import Neo4jGraph
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import StructuredTool
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from dotenv import load_dotenv
from app.utils.flow_simulator import run_flow_simulation  # BAU simulator
from app.design_system import (
    set_brand_details,
    upsert_design_rule,
    upsert_palette,
    upsert_article,
    check_adherence,
)
import os
import json
from langchain_community.document_loaders import WebBaseLoader

# Load environment from .env and .env.local (dev)
load_dotenv()
load_dotenv(".env.local")

# Neo4j connection
# Provide a sensible default for local single-instance use
_neo4j_uri = os.getenv("NEO4J_URI") or "bolt://localhost:7687"
graph = Neo4jGraph(
    url=_neo4j_uri,
    username=os.getenv("NEO4J_USER"),
    password=os.getenv("NEO4J_PASSWORD"),
    database=os.getenv("NEO4J_DATABASE", "neo4j"),
)


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
        new_content = json.loads(content) if content else {}

        # Gating: adherence and coherence thresholds
        import os as _os
        min_coh = float(_os.getenv("MIN_COHERENCE", "0.75"))
        require_adherence = (_os.getenv("REQUIRE_ADHERENCE", "true").lower() == "true")
        force_write = (_os.getenv("FORCE_WRITE", "false").lower() == "true")

        if not force_write:
            if require_adherence:
                report = check_adherence()
                if not report.get("ok", False):
                    return "Blocked: adherence issues detected. See CheckAdherence for details."
            sim = run_flow_simulation(max_steps=80)
            if float(sim.get("coherence_score", 0.0)) < min_coh:
                return f"Blocked: coherence_score {sim.get('coherence_score')} below threshold {min_coh}."

        if "brand_palette" in new_content:
            for key, token in new_content["brand_palette"].items():
                query = """
                MERGE (t:BrandToken {key: $key})
                SET t += $props,
                    t.mutability_level = coalesce($mutability_level, t.mutability_level)
                WITH t
                FOREACH (r_id IN $resonance_outputs |
                    MERGE (r:Resonance {id: r_id})
                    MERGE (t)-[:RESONATES_WITH]->(r))
                FOREACH (c_link IN $coherence_links |
                    MERGE (c:CoreNode {system: split(c_link, "-")[0], node: split(c_link, "-")[1]})
                    MERGE (t)-[:COHERES_WITH]->(c))
                """
                graph.query(query, {
                    "key": key,
                    "props": {k: v for k, v in token.items() if k not in ["relationships", "mutability_level", "resonance_outputs", "coherence_links"]},
                    "mutability_level": token.get("mutability_level", "flexible"),
                    "resonance_outputs": token.get("resonance_outputs", []),
                    "coherence_links": token.get("coherence_links", [])
                })

        # Optionally record a SimRun post-write
        try:
            sim2 = run_flow_simulation(max_steps=80)
            graph.query(
                """
                CREATE (r:SimRun {ts: $ts, completed: $completed, confidence: $conf, steps: $steps, status: $status, coherence_score: $coh, origin: 'WriteToDB'})
                """,
                {
                    "ts": __import__("datetime").datetime.utcnow().isoformat(),
                    "completed": bool(sim2.get("completed", False)),
                    "conf": float(sim2.get("final_confidence", 0.0)),
                    "steps": int(sim2.get("total_steps", 0)),
                    "status": sim2.get("status", "unknown"),
                    "coh": float(sim2.get("coherence_score", 0.0)),
                },
            )
        except Exception:
            pass

        return "Successfully wrote to DB."
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
# LLM selection with fallbacks: Azure -> OpenRouter -> error
def _build_llm():
    # Prefer Azure if configured
    if os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"):
        return AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_version=os.getenv("OPENAI_API_VERSION"),
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        )

    # Fallback to OpenRouter (OpenAI-compatible API)
    if os.getenv("OPENROUTER_API_KEY"):
        # Optional headers recommended by OpenRouter docs
        default_headers = {}
        if os.getenv("OPENROUTER_SITE_URL"):
            default_headers["HTTP-Referer"] = os.getenv("OPENROUTER_SITE_URL")
        if os.getenv("OPENROUTER_APP_NAME"):
            default_headers["X-Title"] = os.getenv("OPENROUTER_APP_NAME")

        return ChatOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            model=os.getenv("OPENROUTER_MODEL", "openrouter/auto"),
            default_headers=default_headers or None,
        )

    # As another option, allow xAI Grok via OpenAI-compatible API if provided
    if os.getenv("GROK_API_KEY"):
        try:
            return ChatOpenAI(
                api_key=os.getenv("GROK_API_KEY"),
                base_url=os.getenv("GROK_BASE_URL", "https://api.x.ai/v1"),
                model=os.getenv("GROK_MODEL", "grok-2-latest"),
            )
        except Exception:
            pass

    raise RuntimeError(
        "No LLM credentials found. Set Azure (AZURE_OPENAI_*) or OpenRouter (OPENROUTER_API_KEY)."
    )

llm = _build_llm()

prompt = PromptTemplate(
    input_variables=["test_results", "proposed_content"],
    template="Refine {proposed_content} based on drift test: {test_results}. Ensure alignment with VE phases (2, 12, 24, 42, 480) and modal_domains.emotional.outputs (e.g., 52)."
)

refine_chain = prompt | llm

def refine_rules(test_results: str, proposed_content: str) -> str:
    return refine_chain.invoke({"test_results": test_results, "proposed_content": proposed_content})

refine_tool = StructuredTool.from_function(
    func=refine_rules,
    name="RefineRules",
    description="Refine proposed content for higher consilience."
)

# Design system and content tools
brand_tool = StructuredTool.from_function(
    func=lambda content: set_brand_details(**json.loads(content)),
    name="SetBrandDetails",
    description="Set brand/company details: name, tagline, mission, values, tone, website, extras(JSON). Input is JSON.",
)

rule_tool = StructuredTool.from_function(
    func=lambda content: upsert_design_rule(**json.loads(content)),
    name="UpsertDesignRule",
    description="Create or update a design rule. Input JSON with keys: rule_id, description, rule_type, params.",
)

palette_tool = StructuredTool.from_function(
    func=lambda content: upsert_palette(**json.loads(content)),
    name="UpsertPalette",
    description="Create or update a palette with a name and token_keys list. Input JSON.",
)

article_tool = StructuredTool.from_function(
    func=lambda content: upsert_article(**json.loads(content)),
    name="UpsertArticle",
    description="Create or update an article: slug, title, summary, url, topics[]. Input JSON.",
)

adherence_tool = StructuredTool.from_function(
    func=lambda _: json.dumps(check_adherence()),
    name="CheckAdherence",
    description="Check current tokens and rules adherence; returns JSON summary.",
)

# Optional Tool 5: Generate tangible artifacts (text + HTML snippet + image spec)
def generate_tangible_outputs(content: str) -> str:
    """Produce tangible outputs (campaign text, HTML snippet, image spec) from proposed content.

    Expects a JSON string; if a `brand_palette` with a hex color exists, incorporates it.
    Returns a JSON string with keys: campaign_text, html_snippet, image_spec.
    """
    try:
        payload = json.loads(content) if content else {}
    except Exception:
        payload = {}

    # Try to find a hex color in brand_palette
    color = None
    token_key = None
    if isinstance(payload, dict) and "brand_palette" in payload:
        for k, v in payload["brand_palette"].items():
            if isinstance(v, dict) and "hex" in v:
                color = v.get("hex")
                token_key = k
                break

    color = color or "#ff6699"
    token_key = token_key or "tertiary-emotional"

    campaign_text = (
        f"Headline: Feel the Pulse of Possibility!\n"
        f"Body: Dive into our vibrant world where bold emotions meet coherence."
        f" Powered by token `{token_key}` in {color}."
    )

    html_snippet = f"""
    <html>
      <head>
        <style>
          body {{ background-color: {color}; color: #102030; font-family: Arial, sans-serif; }}
          .cta {{ background: #102030; color: {color}; padding: 12px 16px; border-radius: 6px; }}
        </style>
      </head>
      <body>
        <h1>Welcome to Your Emotional Journey</h1>
        <p>Explore possibilities with our brand vibe, inspired by novelty and coherence.</p>
        <button class=\"cta\">Get Started</button>
      </body>
    </html>
    """.strip()

    image_spec = {
        "type": "color_swatch",
        "colors": [color, "#006dbb"],
        "description": "Emotional gradient from primary token to deep accent",
    }

    return json.dumps({
        "campaign_text": campaign_text,
        "html_snippet": html_snippet,
        "image_spec": image_spec,
    })

artifact_tool = StructuredTool.from_function(
    func=generate_tangible_outputs,
    name="GenerateTangibleArtifacts",
    description="Generate tangible outputs (campaign text, HTML snippet, image spec) from proposed content JSON.",
)

# Tool to fetch web content
def fetch_web_page(url: str) -> str:
    """Fetch the content of a web page."""
    try:
        loader = WebBaseLoader(url)
        data = loader.load()
        return json.dumps([d.page_content for d in data])
    except Exception as e:
        return f"Error: {str(e)}"

web_tool = StructuredTool.from_function(
    func=fetch_web_page,
    name="FetchWebPage",
    description="Fetch the content of a web page given its URL.",
)


def load_local_directory(path: str) -> str:
    if not os.path.isdir(path):
        return "Invalid directory path."
    allowed_extensions = [".txt", ".md", ".json", ".css"]
    contents = []
    for root, dirs, files in os.walk(path):
        for file in files:
            ext = os.path.splitext(file)[1]
            if ext in allowed_extensions:
                full_path = os.path.join(root, file)
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    contents.append(f"File: {full_path}\n{content}\n")
                except Exception as e:
                    contents.append(f"Error reading {full_path}: {str(e)}\n")
    return "\n---\n".join(contents)

local_directory_tool = StructuredTool.from_function(
    func=load_local_directory,
    name="LoadLocalDirectory",
    description="Load the content of files in a local directory given its path. Only reads .txt, .md, .json, .css files. Use this to enrich brand information from local design system.",
)



tools = [
    external_tool,
    write_tool,
    drift_tool,
    refine_tool,
    artifact_tool,
    brand_tool,
    rule_tool,
    palette_tool,
    article_tool,
    adherence_tool,
    web_tool,
    local_directory_tool,
]

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=PromptTemplate.from_template(
"""You are a RAG agent building a brand identity. Use tools to layer external rules, write to DB, test for drift, and refine for consilience with VE phases (2, 12, 24, 42, 480) and core nodes.
You can also fetch content from web pages to enrich the brand information.
You can also load content from local directories using the LoadLocalDirectory tool to enrich the brand information.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}""")
)

executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# Simple module-level helper for callers that expect a `run(prompt)` API
def run(prompt: str, url: str = None, local_path: str = None) -> str:
    try:
        if url:
            prompt = f"{prompt}\n\nReference URL: {url}"
        if local_path:
            prompt = f"{prompt}\n\nLocal directory path: {local_path}"
        return executor.invoke({"input": prompt})["output"]
    except Exception as e:
        import traceback
        return f"An unexpected error occurred: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"


# Run the agent with a prompt
if __name__ == "__main__":
    import sys
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Add a new tertiary emotional token with high prime, test for drift, and refine for consilience."
    response = executor.invoke({"input": prompt})["output"]
    print(response)
    # Optional sample query
    try:
        print(graph.query("MATCH (n:BrandToken) RETURN n.key LIMIT 5"))
    except Exception as e:
        print(f"Neo4j sample query failed: {e}")
