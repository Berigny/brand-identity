from dotenv import load_dotenv

# Load environment from .env and .env.local (dev)
load_dotenv()
load_dotenv(".env.local")

from langchain_core.prompts import PromptTemplate
from langchain_neo4j import Neo4jGraph
try:
    from neo4j.exceptions import ClientError  # for graceful DB fallback
except Exception:  # pragma: no cover
    ClientError = Exception
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import StructuredTool
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from app.utils.llm_factory import make_llm
from app.utils.rerank import rerank_texts
from app.utils.flow_simulator import run_flow_simulation  # BAU simulator
from app.utils.seed_core_network import seed_core_network
from app.design_system import (
    set_brand_details,
    upsert_design_rule,
    upsert_palette,
    upsert_article,
    check_adherence,
)
import os
import json
import re
from langchain_community.document_loaders import WebBaseLoader

# Neo4j connection with graceful database fallback
def _make_graph() -> Neo4jGraph:
    uri = os.getenv("NEO4J_URI") or "bolt://localhost:7687"
    user = os.getenv("NEO4J_USER")
    pwd = os.getenv("NEO4J_PASSWORD")
    db = os.getenv("NEO4J_DATABASE", "neo4j")
    try:
        return Neo4jGraph(url=uri, username=user, password=pwd, database=db)
    except ClientError as e:
        code = getattr(e, "code", "")
        msg = str(e)
        if ("DatabaseNotFound" in code) or ("does not exist" in msg and "Database" in msg):
            if db != "neo4j":
                print(f"[Neo4j] Database '{db}' not found; falling back to 'neo4j'.")
                return Neo4jGraph(url=uri, username=user, password=pwd, database="neo4j")
        raise

graph = _make_graph()


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
        new_content = _parse_json_like(content)

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
                    MERGE (c:CoreNode {system: split(c_link, \"-\")[0], node: split(c_link, \"-\")[1]})
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
    description="Write proposed brand content to Neo4j. The input must be a valid JSON string."
)

# Tool 3: Test for Drift
def test_drift(content: str) -> str:
    simulation = run_flow_simulation()
    result = graph.query("""
    MATCH (t:BrandToken)-[:RESONATES_WITH]->(r:Resonance)
    WHERE NOT r.id IN [2, 12, 24, 42, 480]
    RETURN t.key AS token, r.id AS issue
    UNION
    MATCH (t:BrandToken)-[:COHERES_WITH]->(c:CoreNode)
    WHERE coalesce(c.activations, 0) < 2
    RETURN t.key AS token, c.system + '-' + c.node AS issue
    """)
    return f"Simulation coherence score: {simulation['coherence_score']}\nDrift issues: {result}"

drift_tool = StructuredTool.from_function(
    func=test_drift,
    name="TestDrift",
    description="Test for drift using flow simulation and queries."
)

# Tool 4: Refine Rules
# LLM selection with fallbacks: Azure -> OpenRouter -> error
def _build_llm():
    provider = (os.getenv("LLM_PROVIDER") or "auto").strip().lower()

    def _openrouter_llm():
        if not os.getenv("OPENROUTER_API_KEY"):
            raise RuntimeError("OPENROUTER_API_KEY is required for provider 'openrouter'.")
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

    def _azure_llm():
        azure_endpoint = os.getenv("OPENAI_API_BASE") or os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_deployment = os.getenv("OPENAI_DEPLOYMENT_NAME") or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        azure_api_version = os.getenv("OPENAI_API_VERSION") or os.getenv("AZURE_OPENAI_API_VERSION")
        azure_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
        if not (azure_endpoint and azure_deployment and azure_api_version and azure_api_key):
            raise RuntimeError("Azure provider requires OPENAI_API_BASE, OPENAI_DEPLOYMENT_NAME, OPENAI_API_VERSION, OPENAI_API_KEY.")
        return AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            openai_api_version=azure_api_version,
            deployment_name=azure_deployment,
            api_key=azure_api_key,
        )

    def _grok_llm():
        if not os.getenv("GROK_API_KEY"):
            raise RuntimeError("GROK_API_KEY is required for provider 'grok'.")
        return ChatOpenAI(
            api_key=os.getenv("GROK_API_KEY"),
            base_url=os.getenv("GROK_BASE_URL", "https://api.x.ai/v1"),
            model=os.getenv("GROK_MODEL", "grok-2-latest"),
        )

    # Explicit provider selection
    if provider == "openrouter":
        return _openrouter_llm()
    if provider == "azure":
        return _azure_llm()
    if provider == "grok":
        return _grok_llm()
    if provider in ("proxy", "openai", "openai_compat"):
        return make_llm()

    # Auto: try OpenRouter -> Azure -> Grok
    # Proxy first if configured
    if os.getenv("OPENAI_API_BASE") or os.getenv("LITELLM_MASTER_KEY"):
        return make_llm()
    if os.getenv("OPENROUTER_API_KEY"):
        return _openrouter_llm()
    try:
        return _azure_llm()
    except Exception:
        pass
    try:
        return _grok_llm()
    except Exception:
        pass

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

def refine_rules_tool(payload: str) -> str:
    """Refine proposed content.

    Accepts either a JSON string with keys {"test_results", "proposed_content"}
    or a plain string treated as proposed_content (empty test_results).
    """
    test_results = ""
    proposed = ""
    if payload:
        try:
            data = json.loads(payload)
            test_results = str(data.get("test_results", ""))
            proposed = str(data.get("proposed_content", ""))
        except Exception:
            proposed = str(payload)
    return refine_chain.invoke({"test_results": test_results, "proposed_content": proposed})

# Helper to parse tolerant JSON inputs
def _parse_json_like(payload: str):
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return payload
    s = str(payload).strip()
    if not s:
        return {}
    # If input is quoted with single quotes, strip them
    if s.startswith("'") and s.endswith("'"):
        s = s[1:-1]
    s = re.sub(r"^\s*json\s*", "", s, flags=re.IGNORECASE)
    if s.startswith("```"):
        m = re.match(r"^```(?:json)?\n(.*?)```\s*$", s, flags=re.DOTALL | re.IGNORECASE)
        if m:
            s = m.group(1).strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        import ast
        v = ast.literal_eval(s)
        if isinstance(v, dict):
            return v
    except Exception:
        pass
    raise ValueError("Invalid JSON input. Provide a JSON object string.")

refine_tool = StructuredTool.from_function(
    func=refine_rules_tool,
    name="RefineRules",
    description="Refine proposed content for higher consilience. Input: JSON {test_results, proposed_content} or plain text.")

# Design system and content tools
def _set_brand_details_tool(content: str) -> str:
    try:
        data = _parse_json_like(content)
        vals = data.get("values")
        if isinstance(vals, str):
            data["values"] = [v.strip() for v in vals.split(",") if v.strip()]
        res = set_brand_details(
            name=data.get("name"),
            tagline=data.get("tagline"),
            mission=data.get("mission"),
            values=data.get("values"),
            tone=data.get("tone"),
            website=data.get("website"),
            extras=data.get("extras"),
        )
        return json.dumps(res)
    except Exception as e:
        return f"Error: {str(e)}"

brand_tool = StructuredTool.from_function(
    func=_set_brand_details_tool,
    name="SetBrandDetails",
    description="Set brand/company details. Input JSON: {name, tagline?, mission?, values?, tone?, website?, extras?}.",
)

def amend_threshold(prime: float, base: float = 0.2, k: float = 0.6) -> float:
    prime = max(0.0, min(1.0, float(prime)))
    return base + k * (1.0 - prime)

def update_rule_with_threshold(content: str) -> str:
    try:
        data = json.loads(content)
        rule_id = data.get("rule_id")
        proposed_text = data.get("description")
        evidence_score = data.get("evidence_score")

        if not all([rule_id, proposed_text, evidence_score]):
            return "Error: rule_id, description, and evidence_score are required."

        # Get the prime value for the rule from the database
        result = graph.query("MATCH (r:DesignRule {id: $id}) RETURN r.prime AS prime", {"id": rule_id})
        if not result or "prime" not in result[0]:
            return f"Error: Rule with id '{rule_id}' not found or has no prime value."

        prime = result[0]["prime"]
        threshold = amend_threshold(prime)

        if evidence_score >= threshold:
            from datetime import datetime
            upsert_design_rule(
                rule_id=rule_id,
                description=proposed_text,
                rule_type=data.get("rule_type"),
                params=data.get("params"),
                lastAmendedAt=datetime.utcnow().isoformat(),
                lastEvidenceScore=evidence_score,
            )
            return f"Rule '{rule_id}' updated successfully. Threshold was {threshold:.2f}."
        else:
            return f"Amendment threshold not met for rule '{rule_id}'. Evidence score {evidence_score} is below threshold {threshold:.2f}."

    except Exception as e:
        return f"Error: {str(e)}"

rule_tool = StructuredTool.from_function(
    func=update_rule_with_threshold,
    name="UpsertDesignRule",
    description="Create or update a design rule, subject to an amendment threshold. Input JSON with keys: rule_id, description, evidence_score, rule_type, params.",
)

def _create_design_rule_tool(content: str) -> str:
    try:
        data = _parse_json_like(content)
        rule_id = data.get("rule_id")
        description = data.get("description")
        if not rule_id or not description:
            return "Error: rule_id and description are required."
        
        from datetime import datetime
        res = upsert_design_rule(
            rule_id=rule_id,
            description=description,
            rule_type=data.get("rule_type"),
            params=data.get("params"),
            createdAt=datetime.utcnow().isoformat(),
            lastAmendedAt=datetime.utcnow().isoformat(),
            lastEvidenceScore=data.get("evidence_score", 0.0),
        )
        return json.dumps(res)
    except Exception as e:
        return f"Error: {str(e)}"

create_rule_tool = StructuredTool.from_function(
    func=_create_design_rule_tool,
    name="CreateDesignRule",
    description="Create a new design rule. Input JSON: {rule_id, description, rule_type?, params?, evidence_score?}.",
)

def _upsert_palette_tool(content: str) -> str:
    try:
        data = _parse_json_like(content)
        name = data.get("name")
        token_keys = data.get("token_keys") or data.get("keys") or data.get("tokens") or data.get("colors")
        if isinstance(token_keys, list) and token_keys and isinstance(token_keys[0], dict):
            token_keys = [d.get("key") for d in token_keys if isinstance(d, dict) and d.get("key")]
        if not name or not isinstance(token_keys, list):
            return "Error: provide name (string) and token_keys (list). Aliases accepted: keys/tokens/colors -> token_keys."
        res = upsert_palette(name=name, token_keys=[str(k) for k in token_keys])
        return json.dumps(res)
    except Exception as e:
        return f"Error: {str(e)}"

palette_tool = StructuredTool.from_function(
    func=_upsert_palette_tool,
    name="UpsertPalette",
    description="Create or update a palette. Input JSON: {name, token_keys:[...]}. Aliases accepted: keys/tokens/colors -> token_keys.",
)

def _upsert_article_tool(content: str) -> str:
    try:
        data = _parse_json_like(content)
        res = upsert_article(
            slug=data.get("slug"),
            title=data.get("title"),
            summary=data.get("summary"),
            url=data.get("url"),
            topics=data.get("topics"),
        )
        return json.dumps(res)
    except Exception as e:
        return f"Error: {str(e)}"

article_tool = StructuredTool.from_function(
    func=_upsert_article_tool,
    name="UpsertArticle",
    description="Create or update an article. Input JSON: {slug, title, summary, url?, topics?}.",
)

def _check_adherence_tool(_: str = "") -> str:
    return json.dumps(check_adherence())

adherence_tool = StructuredTool.from_function(
    func=_check_adherence_tool,
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

    html_snippet = f'''
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
        <button class="cta">Get Started</button>
      </body>
    </html>
    '''.strip()

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
    # Allow JSON payload: {"path": "/dir/or/file", "query": "...", "top_k": 3, "max_tokens": 900}
    user_query = None
    top_k_override = None
    max_tokens_override = None
    if path and path.strip().startswith("{"):
        try:
            payload = json.loads(path)
            user_query = payload.get("query")
            top_k_override = payload.get("top_k")
            max_tokens_override = payload.get("max_tokens")
            path = payload.get("path") or ""
        except Exception:
            pass

    if not os.path.exists(path):
        return "Invalid path."
    allowed_extensions = [".txt", ".md", ".json", ".css"]
    skip_dirs = {".git", ".venv", "node_modules", "dist", "build", "__pycache__"}
    max_files = int(os.getenv("MAX_FILES", "30"))
    max_chars_per_file = int(os.getenv("MAX_CHARS_PER_FILE", "4000"))
    max_total_chars = int(os.getenv("MAX_TOTAL_CHARS", "25000"))

    contents = []
    total_chars = 0
    file_count = 0

    # Single file support
    if os.path.isfile(path):
        ext = os.path.splitext(path)[1]
        if ext not in allowed_extensions:
            return f"Unsupported file extension: {ext}"
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read(max_chars_per_file + 1)
            truncated = " (TRUNCATED)" if len(content) > max_chars_per_file else ""
            content = content[:max_chars_per_file]
            snippet = f"File: {path}{truncated}\n{content}"
            # Consider auto-rerank unnecessary for single file; just return snippet
            return snippet
        except Exception as e:
            return f"Error reading {path}: {str(e)}"

    for root, dirs, files in os.walk(path):
        # prune skipped directories
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for file in files:
            if file_count >= max_files or total_chars >= max_total_chars:
                break
            ext = os.path.splitext(file)[1]
            if ext in allowed_extensions:
                full_path = os.path.join(root, file)
                try:
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read(max_chars_per_file + 1)
                    truncated = " (TRUNCATED)" if len(content) > max_chars_per_file else ""
                    content = content[:max_chars_per_file]
                    snippet = f"File: {full_path}{truncated}\n{content}\n"
                    contents.append(snippet)
                    file_count += 1
                    total_chars += len(content)
                except Exception as e:
                    contents.append(f"Error reading {full_path}: {str(e)}\n")
        if file_count >= max_files or total_chars >= max_total_chars:
            break

    # Auto-rerank/compress if configured or if we hit caps
    auto_rerank = os.getenv("AUTO_RERANK", "1") == "1"
    default_query = os.getenv("DEFAULT_RERANK_QUERY", "brand color rules tokens palette")
    query = user_query or default_query
    top_k = int(top_k_override or os.getenv("RERANK_TOP_K", 3))
    max_tokens = int(max_tokens_override or os.getenv("CONTEXT_MAX_TOKENS", 900))

    selected = contents
    if auto_rerank and (file_count >= max_files or total_chars >= max_total_chars or len(contents) > top_k):
        try:
            selected = rerank_texts(query, contents, top_k=top_k, max_tokens=max_tokens)
        except Exception:
            # If rerank fails for any reason, fall back to first-k trimmed
            selected = contents[: top_k]

    header = (
        f"Loaded {file_count} files (caps: MAX_FILES={max_files}, MAX_CHARS_PER_FILE={max_chars_per_file}, MAX_TOTAL_CHARS={max_total_chars}).\n"
        f"Auto-rerank: {'on' if auto_rerank else 'off'}; query='{query}'; top_k={top_k}; max_tokens={max_tokens}.\n"
    )
    return header + "\n---".join(selected)

local_directory_tool = StructuredTool.from_function(
    func=load_local_directory,
    name="LoadLocalDirectory",
    description="Load file(s) from local path. Input may be a plain path or JSON {path, query?, top_k?, max_tokens?}. Reads .txt/.md/.json/.css; auto-reranks if large.",
)



tools = [
    external_tool,
    write_tool,
    drift_tool,
    refine_tool,
    artifact_tool,
    brand_tool,
    rule_tool,
    create_rule_tool,
    palette_tool,
    article_tool,
    adherence_tool,
    web_tool,
    local_directory_tool,
]

# Seed core S1/S2/IC/EC topology into Neo4j
def seed_core_network_tool(_: str = "") -> str:
    try:
        out = seed_core_network()
        return json.dumps({"ok": True, "nodes": out.get("nodes"), "edges": out.get("edges")})
    except Exception as e:
        return f"Error: {str(e)}"

seed_core_tool = StructuredTool.from_function(
    func=seed_core_network_tool,
    name="SeedCoreNetwork",
    description="Create/refresh S1/S2/IC/EC CoreNode graph and CONNECTED_TO edges in Neo4j (idempotent).",
)

tools.append(seed_core_tool)

# Optional: Rerank and compress contexts for cheaper LLM calls
def rerank_context(payload: str) -> str:
    """Input JSON: {"query": str, "texts": [str], "top_k": 3, "max_tokens": 300}
    Returns a single string of top-k trimmed snippets separated by \n---\n.
    """
    try:
        data = json.loads(payload) if payload else {}
        query = data.get("query", "")
        texts = data.get("texts", [])
        top_k = int(data.get("top_k", os.getenv("RERANK_TOP_K", 3)))
        max_tokens = int(data.get("max_tokens", os.getenv("CONTEXT_MAX_TOKENS", 900)))
        selected = rerank_texts(query, texts, top_k=top_k, max_tokens=max_tokens)
        return "\n---".join(selected)
    except Exception as e:
        return f"Error: {str(e)}"

rerank_tool = StructuredTool.from_function(
    func=rerank_context,
    name="RerankContext",
    description="Rerank and trim candidate texts given a query; returns top-k compressed snippets.",
)

tools.append(rerank_tool)

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=PromptTemplate.from_template(
'''You are a RAG agent building a brand identity. Use tools to layer external rules, write to DB, test for drift, and refine for consilience with VE phases (2, 12, 24, 42, 480) and core nodes.
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
Thought: {agent_scratchpad}''')
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
    # CLI: seed core topology or run the agent
    if len(sys.argv) > 1 and sys.argv[1] in {"seed-core", "--seed-core", "seed"}:
        try:
            out = seed_core_network()
            print(f"Seeded core network: nodes={out.get('nodes')} edges={out.get('edges')}")
            raise SystemExit(0)
        except Exception as e:
            import traceback
            print(f"Seeding failed: {e}\n\n{traceback.format_exc()}")
            raise SystemExit(1)

    prompt = sys.argv[1] if len(sys.argv) > 1 else "Add a new tertiary emotional token with high prime, test for drift, and refine for consilience."
    response = executor.invoke({"input": prompt})["output"]
    print(response)
    # Optional sample query
    try:
        print(graph.query("MATCH (n:BrandToken) RETURN n.key LIMIT 5"))
    except Exception as e:
        print(f"Neo4j sample query failed: {e}")
