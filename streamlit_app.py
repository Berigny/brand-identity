import streamlit as st
import json

# Import helper run() and graph from rag_agent and design_system helpers
try:
    from rag_agent import run as run_agent, graph
    from app.design_system import set_brand_details, upsert_design_rule, upsert_palette, upsert_article, check_adherence, add_rule_presets
    from app.utils.flow_simulator import run_flow_simulation
    from app.utils.seed_core_network import seed_core_network
    from app.database import db as neo_db
except Exception as e:
    run_agent = None
    graph = None
    _import_err = e
    seed_core_network = None
    neo_db = None

st.title("Brand Identity Workbench")

tabs = st.tabs(["Agent", "Brand", "Rules", "Palettes", "Articles", "Adherence", "Simulator", "Database"])

with tabs[0]:
    st.header("Agent")
    if run_agent is None:
        st.error(f"Failed to import rag_agent: {_import_err}")
    else:
        prompt = st.text_input(
            "Enter request:",
            "Add a new tertiary emotional token with high prime, test for drift, and refine for consilience.",
        )
        url = st.text_input("Enter URL (optional):")
        local_path = st.text_input("Enter local directory path (optional):")
        if st.button("Run Agent"):
            with st.spinner("Processing..."):
                try:
                    response = run_agent(prompt, url=url if url else None, local_path=local_path if local_path else None)
                    st.success("Agent Response:")
                    st.write(response)
                except Exception as e:
                    st.error(f"Agent error: {e}")
        if graph is not None:
            try:
                st.write("Sample Brand Tokens:")
                st.json(graph.query("MATCH (t:BrandToken) RETURN t.key AS key, t.hex AS hex LIMIT 5"))
            except Exception as e:
                st.warning(f"Neo4j query failed: {e}")

with tabs[1]:
    st.header("Brand Details")
    with st.form("brand_form"):
        name = st.text_input("Company/Brand Name")
        tagline = st.text_input("Tagline")
        mission = st.text_area("Mission")
        values = st.text_input("Values (comma-separated)")
        tone = st.text_input("Tone (e.g., professional, playful)")
        website = st.text_input("Website")
        submitted = st.form_submit_button("Save Brand Details")
        if submitted:
            try:
                res = set_brand_details(
                    name=name,
                    tagline=tagline,
                    mission=mission,
                    values=[v.strip() for v in values.split(",") if v.strip()],
                    tone=tone,
                    website=website,
                )
                st.success("Saved")
                st.json(res)
            except Exception as e:
                st.error(f"Error: {e}")

with tabs[2]:
    st.header("Design Rules")
    with st.form("rule_form"):
        rid = st.text_input("Rule ID", value="roles-required")
        rdesc = st.text_input("Description", value="Require primary, secondary, tertiary roles")
        rtype = st.selectbox("Type", ["require_roles", "other"], index=0)
        rparams = st.text_area("Params (JSON)", value=json.dumps({"roles": ["primary", "secondary", "tertiary"]}))
        submitted = st.form_submit_button("Save Rule")
        if submitted:
            try:
                params = json.loads(rparams or "{}")
                res = upsert_design_rule(rid, rdesc, rtype, params)
                st.success("Saved")
                st.json(res)
            except Exception as e:
                st.error(f"Error: {e}")
    if st.button("Add Rule Presets"):
        try:
            res = add_rule_presets()
            st.success("Added presets")
            st.json(res)
        except Exception as e:
            st.error(f"Error: {e}")

with tabs[3]:
    st.header("Palettes")
    with st.form("palette_form"):
        pname = st.text_input("Palette Name", value="default")
        keys = st.text_input("Token keys (comma-separated)")
        submitted = st.form_submit_button("Save Palette")
        if submitted:
            try:
                res = upsert_palette(pname, [k.strip() for k in keys.split(",") if k.strip()])
                st.success("Saved")
                st.json(res)
            except Exception as e:
                st.error(f"Error: {e}")

with tabs[4]:
    st.header("Articles")
    with st.form("article_form"):
        slug = st.text_input("Slug", value="on-brand-campaign-1")
        title = st.text_input("Title")
        summary = st.text_area("Summary")
        url = st.text_input("URL")
        topics = st.text_input("Topics (comma-separated)")
        submitted = st.form_submit_button("Save Article")
        if submitted:
            try:
                res = upsert_article(
                    slug=slug,
                    title=title,
                    summary=summary,
                    url=url,
                    topics=[t.strip() for t in topics.split(",") if t.strip()],
                )
                st.success("Saved")
                st.json(res)
            except Exception as e:
                st.error(f"Error: {e}")

with tabs[5]:
    st.header("Adherence Check")
    if st.button("Run Adherence Check"):
        try:
            report = check_adherence()
            st.json(report)
            if not report.get("ok"):
                st.warning("Some adherence issues found. See details above.")
            else:
                st.success("All checks passed.")
        except Exception as e:
            st.error(f"Error: {e}")

with tabs[6]:
    st.header("Simulator")
    col1, col2 = st.columns(2)
    if col1.button("Run BAU Simulation"):
        try:
            result = run_flow_simulation()
            st.json(result)
        except Exception as e:
            st.error(f"Error: {e}")
    if col2.button("Run + Save Metric"):
        try:
            result = run_flow_simulation()
            st.json(result)
            # Persist a SimRun row
            if graph is not None:
                graph.query(
                    """
                    CREATE (r:SimRun {ts: $ts, completed: $completed, confidence: $conf, steps: $steps, status: $status, coherence_score: $coh, origin:'Streamlit'})
                    """,
                    {
                        "ts": __import__("datetime").datetime.utcnow().isoformat(),
                        "completed": bool(result.get("completed", False)),
                        "conf": float(result.get("final_confidence", 0.0)),
                        "steps": int(result.get("total_steps", 0)),
                        "status": result.get("status", "unknown"),
                        "coh": float(result.get("coherence_score", 0.0)),
                    },
                )
                st.success("Saved SimRun")
        except Exception as e:
            st.error(f"Error: {e}")

    st.subheader("Trend")
    try:
        if graph is not None:
            rows = graph.query("MATCH (r:SimRun) RETURN r.ts AS ts, r.coherence_score AS coherence, r.confidence AS confidence, r.steps AS steps ORDER BY r.ts ASC")
            if rows:
                import pandas as pd
                import numpy as np
                df = pd.DataFrame(rows)
                df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
                numeric_cols = ["coherence", "confidence", "steps"]
                for col in numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
                df = df.replace([np.inf, -np.inf], 0)
                df = df.dropna(subset=["ts"]).set_index("ts")
                st.line_chart(df[["coherence", "confidence"]])
                st.bar_chart(df[["steps"]])
            else:
                st.info("No SimRun metrics yet. Run and save a metric.")
        else:
            st.info("Graph not available for trend.")
    except Exception as e:
        st.warning(f"Trend error: {e}")

with tabs[7]:
    st.header("Database Preview")
    if graph is None and neo_db is None:
        st.error("Neo4j connection not available. Configure NEO4J_* env vars.")
    else:
        import os
        st.caption(f"NEO4J_URI={os.getenv('NEO4J_URI','?')} | NEO4J_DATABASE={os.getenv('NEO4J_DATABASE','neo4j')}")
        col1, col2, col3 = st.columns(3)
        try:
            q = lambda cy, p=None: (graph.query(cy, p or {}) if graph is not None else neo_db.execute_query(cy, p or {}))

            # Quick counts
            c_tokens = q("MATCH (t:BrandToken) RETURN count(t) AS n")[0]["n"] if q("MATCH (t:BrandToken) RETURN count(t) AS n") else 0
            c_rules = q("MATCH (r:DesignRule) RETURN count(r) AS n")[0]["n"] if q("MATCH (r:DesignRule) RETURN count(r) AS n") else 0
            c_pal = q("MATCH (p:Palette) RETURN count(p) AS n")[0]["n"] if q("MATCH (p:Palette) RETURN count(p) AS n") else 0
            c_core = q("MATCH (c:CoreNode) RETURN count(c) AS n")[0]["n"] if q("MATCH (c:CoreNode) RETURN count(c) AS n") else 0
            c_edges = q("MATCH ()-[r:CONNECTED_TO]->() RETURN count(r) AS n")[0]["n"] if q("MATCH ()-[r:CONNECTED_TO]->() RETURN count(r) AS n") else 0
            c_runs = q("MATCH (r:SimRun) RETURN count(r) AS n")[0]["n"] if q("MATCH (r:SimRun) RETURN count(r) AS n") else 0

            col1.metric("BrandTokens", c_tokens)
            col2.metric("DesignRules", c_rules)
            col3.metric("Palettes", c_pal)
            col1.metric("CoreNodes", c_core)
            col2.metric("CONNECTED_TO edges", c_edges)
            col3.metric("SimRuns", c_runs)

            st.subheader("Brand Tokens")
            rows = q(
                """
                MATCH (t:BrandToken)
                OPTIONAL MATCH (t)-[:RESONATES_WITH]->(r:Resonance)
                WITH t, count(DISTINCT r) AS resonance
                OPTIONAL MATCH (t)-[:COHERES_WITH]->(c:CoreNode)
                RETURN t.key AS key, t.hex AS hex, t.role AS role, t.mutability_level AS mutability,
                       resonance, count(DISTINCT c) AS coherence
                ORDER BY key LIMIT 200
                """
            )
            if rows:
                import pandas as pd
                st.dataframe(pd.DataFrame(rows))
            else:
                st.info("No BrandTokens yet.")

            st.subheader("Core Nodes")
            rows = q(
                """
                MATCH (c:CoreNode)
                OPTIONAL MATCH (c)-[r:CONNECTED_TO]->(:CoreNode)
                RETURN c.key AS key, c.system AS system, c.node AS node, c.idx AS idx, c.role AS role, c.prime AS prime,
                       c.activations AS activations, count(r) AS out_edges
                ORDER BY system, toInteger(coalesce(c.idx, 9999)) LIMIT 200
                """
            )
            if rows:
                import pandas as pd
                st.dataframe(pd.DataFrame(rows))
            else:
                st.info("No CoreNode records.")

            st.subheader("Core Edges")
            rows = q(
                """
                MATCH (a:CoreNode)-[r:CONNECTED_TO]->(b:CoreNode)
                RETURN a.key AS from, b.key AS to, r.kind AS kind, r.weight AS weight, r.count AS count
                ORDER BY from, to LIMIT 300
                """
            )
            if rows:
                import pandas as pd
                st.dataframe(pd.DataFrame(rows))
            else:
                st.info("No CONNECTED_TO edges.")

            # Visualizations
            st.subheader("Visualizations")
            try:
                nodes_rows = q("""
                    MATCH (c:CoreNode)
                    RETURN c.key AS key, c.system AS system, c.prime AS prime, coalesce(c.activations,0) AS activations
                """)
                edges_rows = q("""
                    MATCH (a:CoreNode)-[r:CONNECTED_TO]->(b:CoreNode)
                    RETURN a.key AS src, b.key AS dst, coalesce(r.kind,'cubic') AS kind, coalesce(r.weight,0.8) AS weight, coalesce(r.count,0) AS count
                """)

                if nodes_rows:
                    import networkx as nx
                    import matplotlib.pyplot as plt
                    import numpy as np

                    size_by_activ = st.checkbox("Size nodes by activations", value=True)
                    width_by_count = st.checkbox("Edge width by count (else weight)", value=True)
                    layout_choice = st.selectbox("Layout", ["spring", "kamada-kawai", "circular"], index=0)

                    G = nx.DiGraph()
                    for r in nodes_rows:
                        G.add_node(r["key"], system=r.get("system"), prime=float(r.get("prime") or 0), activations=int(r.get("activations") or 0))
                    for r in edges_rows or []:
                        G.add_edge(r["src"], r["dst"], kind=r.get("kind"), weight=float(r.get("weight") or 0.8), count=int(r.get("count") or 0))

                    if layout_choice == "spring":
                        pos = nx.spring_layout(G, seed=42)
                    elif layout_choice == "kamada-kawai":
                        pos = nx.kamada_kawai_layout(G)
                    else:
                        pos = nx.circular_layout(G)

                    def color_for_system(sys):
                        return "skyblue" if sys == "S1" else ("lightgreen" if sys == "S2" else "orange")

                    node_colors = [color_for_system(G.nodes[n].get("system")) for n in G.nodes]
                    base_size = 800
                    node_sizes = [base_size + (G.nodes[n].get("activations", 0) * 80 if size_by_activ else 0) for n in G.nodes]
                    # Avoid zero-size nodes
                    node_sizes = [max(400, s) for s in node_sizes]

                    def edge_color_for_kind(k):
                        return "green" if k == "tetrahedral" else ("red" if k == "failsafe" else "gray")

                    edge_colors = [edge_color_for_kind(G.edges[e].get("kind")) for e in G.edges]
                    edge_widths = [
                        (1.0 + (G.edges[e].get("count", 0) / 2.0)) if width_by_count else (1.0 + 2.0 * float(G.edges[e].get("weight", 0.8)))
                        for e in G.edges
                    ]

                    fig = plt.figure(figsize=(12, 8))
                    nx.draw(
                        G,
                        pos,
                        with_labels=True,
                        node_color=node_colors,
                        node_size=node_sizes,
                        edge_color=edge_colors,
                        width=edge_widths,
                        font_size=8,
                        arrows=True,
                    )
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info("No CoreNode data to visualize.")
            except Exception as e:
                st.warning(f"Visualization error: {e}")

            st.subheader("Palettes")
            rows = q(
                """
                MATCH (p:Palette)
                OPTIONAL MATCH (p)-[:CONTAINS]->(t:BrandToken)
                WITH p, collect(t.key) AS keys
                RETURN p.name AS name, keys, size(keys) AS size
                ORDER BY name LIMIT 200
                """
            )
            if rows:
                import pandas as pd
                st.dataframe(pd.DataFrame(rows))
            else:
                st.info("No Palettes.")

            st.subheader("Design Rules")
            rows = q(
                """
                MATCH (r:DesignRule)
                RETURN r.id AS id, r.type AS type, r.description AS description,
                       r.lastEvidenceScore AS lastEvidenceScore, toString(r.lastAmendedAt) AS lastAmendedAt
                ORDER BY id LIMIT 200
                """
            )
            if rows:
                import pandas as pd
                st.dataframe(pd.DataFrame(rows))
            else:
                st.info("No DesignRule records.")

            st.subheader("Articles")
            rows = q(
                """
                MATCH (a:Article)
                OPTIONAL MATCH (a)-[:ABOUT]->(t:Topic)
                WITH a, collect(t.name) AS topics
                RETURN a.slug AS slug, a.title AS title, a.url AS url, topics
                ORDER BY slug LIMIT 200
                """
            )
            if rows:
                import pandas as pd
                st.dataframe(pd.DataFrame(rows))
            else:
                st.info("No Articles.")

            st.subheader("Sim Runs")
            rows = q(
                """
                MATCH (r:SimRun)
                RETURN r.ts AS ts, r.origin AS origin, r.coherence_score AS coherence, r.confidence AS confidence, r.steps AS steps, r.status AS status
                ORDER BY r.ts DESC LIMIT 200
                """
            )
            if rows:
                import pandas as pd
                st.dataframe(pd.DataFrame(rows))
            else:
                st.info("No SimRun records.")

        except Exception as e:
            st.error(f"Query error: {e}")

        st.subheader("Actions")
        c1, c2, c3 = st.columns(3)
        if c1.button("Seed Core Network"):
            if seed_core_network is None:
                st.error("Seeder not available.")
            else:
                try:
                    out = seed_core_network()
                    st.success(f"Seeded: nodes={out.get('nodes')} edges={out.get('edges')}")
                except Exception as e:
                    st.error(f"Seeding failed: {e}")

        with c2.form("custom_query"):
            cy = st.text_area("Run custom Cypher (read-only recommended)", value="MATCH (n) RETURN labels(n) AS labels, count(n) AS cnt LIMIT 10")
            submitted = st.form_submit_button("Run")
            if submitted:
                try:
                    rows = q(cy)
                    import pandas as pd
                    st.dataframe(pd.DataFrame(rows))
                except Exception as e:
                    st.error(f"Query failed: {e}")

        sim_steps = c3.slider("Simulation max steps", min_value=10, max_value=300, value=120, step=10)
        if c3.button("Run Simulation + Record Activations"):
            try:
                # Ensure topology exists
                if seed_core_network is not None:
                    seed_core_network()
                # Run simulation
                sim = run_flow_simulation(max_steps=int(sim_steps))
                trans = sim.get("transitions") or []
                # Map simulator nodes to CoreNode.key
                def to_key(x):
                    s = str(x)
                    if s == "IC":
                        return "C-IC"
                    if s == "EC":
                        return "C-EC"
                    try:
                        n = int(s)
                    except ValueError:
                        return None
                    if 0 <= n <= 3:
                        return f"S1-{n}"
                    if 4 <= n <= 7:
                        return f"S2-{n}"
                    return None
                nodes = set()
                pairs = []
                for t in trans:
                    kf = to_key(t.get("from"))
                    kt = to_key(t.get("to"))
                    if kf:
                        nodes.add(kf)
                    if kt:
                        nodes.add(kt)
                    if kf and kt:
                        pairs.append({"from": kf, "to": kt})
                # Update DB
                n_updated = q("""
                    UNWIND $nodes AS k
                    MATCH (c:CoreNode {key:k})
                    SET c.activations = coalesce(c.activations,0) + 1
                    RETURN count(c) AS n
                """, {"nodes": list(nodes)})
                e_updated = q("""
                    UNWIND $pairs AS p
                    MATCH (a:CoreNode {key:p.from}),(b:CoreNode {key:p.to})
                    MERGE (a)-[r:CONNECTED_TO]->(b)
                    SET r.count = coalesce(r.count,0) + 1, r.lastSeen = timestamp()
                    RETURN count(r) AS n
                """, {"pairs": pairs})
                n_cnt = (n_updated[0]["n"] if n_updated else 0)
                e_cnt = (e_updated[0]["n"] if e_updated else 0)
                st.success(f"Recorded activations: nodes updated={n_cnt}, edges updated={e_cnt}")
                with st.expander("Activation details"):
                    st.write({"nodes": list(nodes)[:50], "pairs": pairs[:50]})
            except Exception as e:
                st.error(f"Activation recording failed: {e}")

        st.subheader("Sample Data")
        help_expander = st.expander("What does this do?", expanded=False)
        help_expander.write("Creates a demo BrandToken and links to CoreNodes. Uses current NEO4J_* env.")
        if st.button("Create Sample BrandToken"):
            try:
                if seed_core_network is not None:
                    seed_core_network()
                cy = """
                MERGE (t:BrandToken {key:$key})
                SET t.hex=$hex, t.role=$role, t.mutability_level=$mut
                WITH t
                FOREACH (rid IN $res |
                  MERGE (r:Resonance {id:rid}) MERGE (t)-[:RESONATES_WITH]->(r))
                FOREACH (ck IN $links |
                  MATCH (c:CoreNode {key:ck}) MERGE (t)-[:COHERES_WITH]->(c))
                RETURN t.key AS key
                """
                params = {
                    "key": "tertiary-emotional",
                    "hex": "#ff6699",
                    "role": "tertiary",
                    "mut": "flexible",
                    "res": [52, 65],
                    "links": ["C-IC", "S1-1", "S1-2", "S2-5"],
                }
                out = (graph.query(cy, params) if graph is not None else neo_db.execute_query(cy, params))
                st.success(f"Created token: {out[0]['key'] if out else 'ok'}")
            except Exception as e:
                st.error(f"Creation failed: {e}")
