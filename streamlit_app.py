import streamlit as st
import json

# Import helper run() and graph from rag_agent and design_system helpers
try:
    from rag_agent import run as run_agent, graph
    from app.design_system import set_brand_details, upsert_design_rule, upsert_palette, upsert_article, check_adherence, add_rule_presets
    from app.utils.flow_simulator import run_flow_simulation
except Exception as e:
    run_agent = None
    graph = None
    _import_err = e

st.title("Brand Identity Workbench")

tabs = st.tabs(["Agent", "Brand", "Rules", "Palettes", "Articles", "Adherence", "Simulator"])

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
