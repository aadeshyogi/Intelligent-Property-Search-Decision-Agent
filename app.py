"""
🏠 Intelligent Property Search & Decision Agent — Streamlit UI
----------------------------------------------------------------
Streamlit wrapper around the LangGraph multi-agent workflow:
Requirement -> Planner -> Property Search -> (parallel: Location & Commute,
Market Price, Neighborhood, Verification) -> Risk (fan-in) -> Ranking ->
Decision -> Human Review (interrupt) -> approve / replan loop.

Run:
    streamlit run app.py

Needs (same folder):
    Indian_housing_Delhi_data.csv

Needs in environment / sidebar:
    OPENAI_API_KEY   (required)
    TAVILY_API_KEY   (optional — Neighborhood Agent degrades gracefully without it)
"""

import os
import re
import math
import json
import uuid
from typing import TypedDict, List, Dict, Any, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # picks up OPENAI_API_KEY / TAVILY_API_KEY from a .env file if present

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Intelligent Property Search & Decision Agent",
    page_icon="🏠",
    layout="wide",
)

CSV_PATH = os.path.join(os.path.dirname(__file__), "Indian_housing_Delhi_data.csv")

# ---------------------------------------------------------------------------
# 1. Data loading & cleaning (same logic as the notebook)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(csv_bytes: bytes = None):
    if csv_bytes is not None:
        import io
        df_raw = pd.read_csv(io.BytesIO(csv_bytes))
    else:
        df_raw = pd.read_csv(CSV_PATH)

    def parse_bhk(house_type):
        if not isinstance(house_type, str):
            return None
        m = re.match(r"(\d+)\s*(BHK|RK)", house_type.strip(), re.IGNORECASE)
        return int(m.group(1)) if m else None

    def parse_size(size_str):
        if not isinstance(size_str, str):
            return None
        try:
            return float(size_str.replace(",", "").lower().replace("sq ft", "").strip())
        except ValueError:
            return None

    def parse_deposit(dep):
        if not isinstance(dep, str):
            return 0.0
        dep = dep.strip()
        if "no deposit" in dep.lower():
            return 0.0
        try:
            return float(dep.replace(",", "").strip())
        except ValueError:
            return None

    def parse_days_ago(v):
        if not isinstance(v, str):
            return None
        m = re.search(r"(a|an|\d+)\s*(day|month|year)", v.lower())
        if not m:
            return None
        num = 1 if m.group(1) in ("a", "an") else int(m.group(1))
        mult = {"day": 1, "month": 30, "year": 365}[m.group(2)]
        return num * mult

    df = df_raw.copy()
    df["bhk"] = df["house_type"].apply(parse_bhk)
    df["size_sqft"] = df["house_size"].apply(parse_size)
    df["deposit_amt"] = df["SecurityDeposit"].apply(parse_deposit)
    df["days_since_posted"] = df["verificationDate"].apply(parse_days_ago)
    df["location_clean"] = df["location"].astype(str).str.strip()
    df["price_per_sqft"] = df["price"] / df["size_sqft"]
    df["is_negotiable"] = df["isNegotiable"].notna()
    df = df.dropna(subset=["price_per_sqft", "bhk", "latitude", "longitude"])

    locality_avg_psf = df.groupby("location_clean")["price_per_sqft"].mean().to_dict()
    return df, locality_avg_psf


# ---------------------------------------------------------------------------
# 2. Shared tools (stand-ins for Maps / Web-search MCP servers)
# ---------------------------------------------------------------------------
LANDMARKS = {
    "connaught place": (28.6315, 77.2167),
    "cyber city gurgaon": (28.4950, 77.0890),
    "cyber hub gurgaon": (28.4950, 77.0890),
    "gurgaon": (28.4595, 77.0266),
    "noida sector 62": (28.6272, 77.3716),
    "noida": (28.5355, 77.3910),
    "aiims": (28.5672, 77.2100),
    "delhi airport": (28.5562, 77.1000),
    "dwarka": (28.5921, 77.0460),
    "rohini": (28.7041, 77.1025),
    "gurugram": (28.4595, 77.0266),
}


def find_landmark(name: Optional[str]):
    if not name:
        return None
    name_l = name.lower().strip()
    for key, coord in LANDMARKS.items():
        if name_l in key or key in name_l:
            return coord
    return None


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dlmb = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def make_web_search(tavily_api_key: Optional[str]):
    """Returns a safe web_search(query) -> str function. Degrades gracefully
    if no Tavily key is configured, or if langchain-tavily isn't installed
    (Neighborhood Agent just returns '' in that case)."""
    if not tavily_api_key:
        def _no_search(query: str) -> str:
            return ""
        return _no_search

    try:
        from langchain_tavily import TavilySearch
    except ImportError:
        st.sidebar.warning(
            "⚠️ `langchain-tavily` not installed — run `pip install langchain-tavily` "
            "to enable neighborhood web search. Skipping for now."
        )
        def _no_search(query: str) -> str:
            return ""
        return _no_search

    tavily = TavilySearch(max_results=3, tavily_api_key=tavily_api_key)

    def web_search(query: str) -> str:
        try:
            res = tavily.invoke({"query": query})
            results = res.get("results", []) if isinstance(res, dict) else res
            snippets = [r.get("content", "")[:300] for r in results[:2]]
            return " | ".join(s for s in snippets if s)
        except Exception as e:
            return f"(web search unavailable: {e})"

    return web_search


# ---------------------------------------------------------------------------
# 3. Agent state
# ---------------------------------------------------------------------------
class PropertyState(TypedDict):
    user_query: str
    requirements: dict
    previous_requirements: Optional[dict]
    changed_fields: List[str]
    agents_to_run: Dict[str, bool]
    candidates: List[dict]
    location_report: Dict[str, Any]
    price_report: Dict[str, Any]
    neighborhood_report: Dict[str, Any]
    verification_report: Dict[str, Any]
    risk_report: Dict[str, Any]
    ranked: List[dict]
    decision: str
    human_feedback: str
    iteration: int


LOCATION_SENSITIVE = {"locations", "commute_landmark", "max_commute_km"}
PRICE_SENSITIVE = {"budget_min", "budget_max"}
RISK_WEIGHT = {"low": 1.5, "medium": 1.0, "high": 0.6}

REQUIREMENT_SYSTEM_PROMPT = """You are a real-estate requirement extraction agent.
Read the user's message (which may add to or modify earlier requirements) and return ONLY a
JSON object (no markdown fences, no extra text) with these keys:
  budget_min (int or null), budget_max (int or null), bhk (int or null),
  locations (list of strings, empty list if unspecified),
  furnishing (one of "Furnished","Semi-Furnished","Unfurnished" or null),
  commute_landmark (string or null), max_commute_km (int or null),
  amenities (list of strings), risk_tolerance (one of "low","medium","high", default "medium")

If the user is only tweaking one thing (e.g. "increase budget to 40000"), carry over the rest of
the previous requirements unchanged."""

DECISION_SYSTEM_PROMPT = """You are the final decision agent of a property search assistant.
Given the user's requirements and a ranked shortlist with scores, write a short final
recommendation (max ~120 words): name the top pick, justify it in 2-3 bullet points using the
numbers given, then briefly mention 1-2 alternatives and their trade-offs."""


# ---------------------------------------------------------------------------
# 4. Build the graph (nodes are closures over df / llm / web_search so they
#    can be rebuilt per-session with the user's own API keys / uploaded data)
# ---------------------------------------------------------------------------
def build_graph(df, locality_avg_psf, llm, web_search):
    from langchain_core.messages import HumanMessage, SystemMessage
    from langgraph.graph import StateGraph, START, END
    from langgraph.types import interrupt
    from langgraph.checkpoint.memory import MemorySaver

    def requirement_agent(state: PropertyState):
        prev = state.get("requirements")
        incoming_message = state.get("human_feedback") or state["user_query"]
        context = f"\nPrevious requirements: {json.dumps(prev)}" if prev else ""
        resp = llm.invoke([
            SystemMessage(content=REQUIREMENT_SYSTEM_PROMPT),
            HumanMessage(content=f"User message: {incoming_message}{context}")
        ])
        raw = resp.content.strip().strip("`").replace("json\n", "", 1)
        try:
            new_req = json.loads(raw)
        except json.JSONDecodeError:
            new_req = prev or {}
        return {
            "requirements": new_req,
            "previous_requirements": prev,
            "iteration": state.get("iteration", 0) + 1,
        }

    def planner_agent(state: PropertyState):
        req, prev = state["requirements"], state.get("previous_requirements")
        if not prev:
            changed = list(req.keys())
        else:
            changed = [k for k in req if req.get(k) != prev.get(k)]

        plan = {
            "location_commute": (not prev) or bool(LOCATION_SENSITIVE & set(changed)),
            "market_price": (not prev) or bool(PRICE_SENSITIVE & set(changed)),
            "neighborhood": (not prev) or bool(LOCATION_SENSITIVE & set(changed)),
            "verification": True,
        }
        return {"changed_fields": changed, "agents_to_run": plan}

    def property_search_agent(state: PropertyState):
        req = state["requirements"]
        sub = df.copy()
        if req.get("bhk"):
            sub = sub[sub["bhk"] == req["bhk"]]
        if req.get("budget_min"):
            sub = sub[sub["price"] >= req["budget_min"]]
        if req.get("budget_max"):
            sub = sub[sub["price"] <= req["budget_max"]]
        if req.get("locations"):
            pattern = "|".join(re.escape(l) for l in req["locations"])
            sub = sub[sub["location_clean"].str.contains(pattern, case=False, na=False)]
        if req.get("furnishing"):
            sub = sub[sub["Status"].str.lower() == req["furnishing"].lower()]

        sub = sub.sort_values("price").head(8).reset_index()
        candidates = [{"id": str(r["index"]), **r.to_dict()} for _, r in sub.iterrows()]
        return {"candidates": candidates}

    def location_commute_agent(state: PropertyState):
        if not state["agents_to_run"].get("location_commute", True):
            return {}
        landmark = find_landmark(state["requirements"].get("commute_landmark"))
        max_km = state["requirements"].get("max_commute_km")
        report = {}
        for c in state["candidates"]:
            if landmark:
                d = round(haversine_km(c["latitude"], c["longitude"], *landmark), 1)
                report[c["id"]] = {"distance_km": d, "within_limit": d <= max_km if max_km else None}
            else:
                report[c["id"]] = {"distance_km": None, "within_limit": None}
        return {"location_report": report}

    def market_price_agent(state: PropertyState):
        if not state["agents_to_run"].get("market_price", True):
            return {}
        report = {}
        for c in state["candidates"]:
            avg = locality_avg_psf.get(c["location_clean"])
            diff_pct = round((c["price_per_sqft"] - avg) / avg * 100, 1) if avg else None
            report[c["id"]] = {"locality_avg_psf": round(avg, 1) if avg else None, "diff_pct": diff_pct}
        return {"price_report": report}

    def neighborhood_agent(state: PropertyState):
        if not state["agents_to_run"].get("neighborhood", True):
            return {}
        report, cache = {}, {}
        for c in state["candidates"]:
            loc = c["location_clean"]
            if loc not in cache:
                cache[loc] = web_search(f"{loc} Delhi neighborhood safety connectivity amenities review")
            report[c["id"]] = {"summary": cache[loc]}
        return {"neighborhood_report": report}

    def verification_agent(state: PropertyState):
        report = {}
        for c in state["candidates"]:
            flags = []
            avg = locality_avg_psf.get(c["location_clean"])
            if avg and (c["price_per_sqft"] < avg * 0.5 or c["price_per_sqft"] > avg * 2):
                flags.append("price_outlier")
            if (c.get("days_since_posted") or 0) > 365:
                flags.append("stale_listing")
            if pd.isna(c.get("deposit_amt")):
                flags.append("missing_deposit_info")
            report[c["id"]] = {"flags": flags, "status": "Flagged" if flags else "OK"}
        return {"verification_report": report}

    def risk_agent(state: PropertyState):
        weight = RISK_WEIGHT.get(state["requirements"].get("risk_tolerance", "medium"), 1.0)
        report = {}
        for c in state["candidates"]:
            cid = c["id"]
            score = 0
            if state.get("verification_report", {}).get(cid, {}).get("status") == "Flagged":
                score += 30
            diff_pct = state.get("price_report", {}).get(cid, {}).get("diff_pct")
            if diff_pct is not None and abs(diff_pct) > 40:
                score += 20
            if state.get("location_report", {}).get(cid, {}).get("within_limit") is False:
                score += 15
            report[cid] = {"risk_score": round(score * weight, 1)}
        return {"risk_report": report}

    def ranking_agent(state: PropertyState):
        ranked = []
        for c in state["candidates"]:
            cid = c["id"]
            risk = state.get("risk_report", {}).get(cid, {}).get("risk_score", 0)
            diff_pct = state.get("price_report", {}).get(cid, {}).get("diff_pct")
            within_limit = state.get("location_report", {}).get(cid, {}).get("within_limit")

            score = 100 - risk
            if diff_pct is not None:
                score += max(0, -diff_pct)
            if within_limit:
                score += 15

            ranked.append({**c, "final_score": round(score, 1)})

        ranked.sort(key=lambda x: x["final_score"], reverse=True)
        return {"ranked": ranked[:5]}

    def _summarize_for_llm(state: PropertyState) -> str:
        lines = []
        for c in state["ranked"][:5]:
            cid = c["id"]
            loc_r = state.get("location_report", {}).get(cid, {})
            price_r = state.get("price_report", {}).get(cid, {})
            risk_r = state.get("risk_report", {}).get(cid, {})
            verif = state.get("verification_report", {}).get(cid, {})
            lines.append(
                f"- [{cid}] {c['bhk']}BHK in {c['location_clean']}, Rs.{c['price']}/mo, "
                f"{c['size_sqft']}sqft, {c.get('Status')}, "
                f"commute={loc_r.get('distance_km')}km, "
                f"price_vs_locality={price_r.get('diff_pct')}%, "
                f"risk_score={risk_r.get('risk_score')}, "
                f"verification={verif.get('status')} {verif.get('flags')}"
            )
        return "\n".join(lines)

    def decision_agent(state: PropertyState):
        resp = llm.invoke([
            SystemMessage(content=DECISION_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"User requirements: {json.dumps(state['requirements'])}\n\n"
                f"Ranked shortlist:\n{_summarize_for_llm(state)}"
            )),
        ])
        return {"decision": resp.content}

    def human_review(state: PropertyState):
        feedback = interrupt({
            "question": "Approve this recommendation? Reply 'yes' to accept, or describe what to change.",
            "decision": state["decision"],
        })
        return {"human_feedback": feedback}

    def route_after_review(state: PropertyState):
        fb = (state.get("human_feedback") or "").lower()
        if "yes" in fb or "approve" in fb:
            return "END"
        return "requirement_agent"

    graph = StateGraph(PropertyState)
    graph.add_node("requirement_agent", requirement_agent)
    graph.add_node("planner_agent", planner_agent)
    graph.add_node("property_search_agent", property_search_agent)
    graph.add_node("location_commute_agent", location_commute_agent)
    graph.add_node("market_price_agent", market_price_agent)
    graph.add_node("neighborhood_agent", neighborhood_agent)
    graph.add_node("verification_agent", verification_agent)
    graph.add_node("risk_agent", risk_agent)
    graph.add_node("ranking_agent", ranking_agent)
    graph.add_node("decision_agent", decision_agent)
    graph.add_node("human_review", human_review)

    graph.add_edge(START, "requirement_agent")
    graph.add_edge("requirement_agent", "planner_agent")
    graph.add_edge("planner_agent", "property_search_agent")

    graph.add_edge("property_search_agent", "location_commute_agent")
    graph.add_edge("property_search_agent", "market_price_agent")
    graph.add_edge("property_search_agent", "neighborhood_agent")
    graph.add_edge("property_search_agent", "verification_agent")

    graph.add_edge("location_commute_agent", "risk_agent")
    graph.add_edge("market_price_agent", "risk_agent")
    graph.add_edge("neighborhood_agent", "risk_agent")
    graph.add_edge("verification_agent", "risk_agent")

    graph.add_edge("risk_agent", "ranking_agent")
    graph.add_edge("ranking_agent", "decision_agent")
    graph.add_edge("decision_agent", "human_review")

    graph.add_conditional_edges(
        "human_review",
        route_after_review,
        {"END": END, "requirement_agent": "requirement_agent"},
    )

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


# ---------------------------------------------------------------------------
# 5. Streamlit session state helpers
# ---------------------------------------------------------------------------
def init_session():
    defaults = {
        "agent": None,
        "thread_id": str(uuid.uuid4()),
        "chat": [],           # list of {"role": ..., "content": ...} for display
        "awaiting_approval": False,
        "last_result": None,
        "graph_ready": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_session():
    st.session_state["thread_id"] = str(uuid.uuid4())
    st.session_state["chat"] = []
    st.session_state["awaiting_approval"] = False
    st.session_state["last_result"] = None


def render_result_panel(result: dict):
    """Renders requirements / agents-run / candidates / ranked table / decision."""
    req = result.get("requirements", {})
    with st.expander("🧾 Extracted requirements", expanded=False):
        st.json(req)

    plan = result.get("agents_to_run")
    if plan:
        cols = st.columns(4)
        labels = {
            "location_commute": "📍 Location & Commute",
            "market_price": "💰 Market Price",
            "neighborhood": "🏘️ Neighborhood",
            "verification": "✅ Verification",
        }
        for i, (key, label) in enumerate(labels.items()):
            ran = plan.get(key, True)
            cols[i].metric(label, "Ran ✅" if ran else "Skipped ⏭️")
        if result.get("changed_fields") is not None and result.get("previous_requirements"):
            st.caption(f"Changed fields since last turn: {result.get('changed_fields') or 'none'}")

    ranked = result.get("ranked") or []
    if ranked:
        st.subheader("🏆 Ranked shortlist")
        rows = []
        for c in ranked:
            cid = c["id"]
            loc_r = result.get("location_report", {}).get(cid, {})
            price_r = result.get("price_report", {}).get(cid, {})
            risk_r = result.get("risk_report", {}).get(cid, {})
            verif = result.get("verification_report", {}).get(cid, {})
            rows.append({
                "Score": c.get("final_score"),
                "BHK": c.get("bhk"),
                "Location": c.get("location_clean"),
                "Rent (₹/mo)": c.get("price"),
                "Size (sqft)": c.get("size_sqft"),
                "Status": c.get("Status"),
                "Commute (km)": loc_r.get("distance_km"),
                "Price vs Locality": f"{price_r.get('diff_pct')}%" if price_r.get("diff_pct") is not None else "—",
                "Risk Score": risk_r.get("risk_score"),
                "Verification": verif.get("status"),
                "Flags": ", ".join(verif.get("flags", []) or []) or "—",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    decision = result.get("decision")
    if decision:
        st.subheader("🤖 Decision Agent recommendation")
        st.markdown(decision)


# ---------------------------------------------------------------------------
# 6. Sidebar — configuration
# ---------------------------------------------------------------------------
def sidebar():
    st.sidebar.title("⚙️ Configuration")

    env_openai = os.getenv("OPENAI_API_KEY", "")
    env_tavily = os.getenv("TAVILY_API_KEY", "")

    if env_openai:
        st.sidebar.success("✅ OpenAI key loaded")
    else:
        st.sidebar.error("❌ OPENAI_API_KEY not found in .env")

    if env_tavily:
        st.sidebar.success("✅ Tavily key loaded")
    else:
        st.sidebar.caption("ℹ️ TAVILY_API_KEY not set — neighborhood web search disabled")

    openai_key = env_openai
    tavily_key = env_tavily

    st.sidebar.divider()
    st.sidebar.subheader("📊 Listings data")
    uploaded = st.sidebar.file_uploader(
        "Upload a different CSV (optional)", type=["csv"],
        help=f"Defaults to the bundled {os.path.basename(CSV_PATH)} if you don't upload one.",
    )

    st.sidebar.divider()
    if st.sidebar.button("🔄 New session / reset conversation", use_container_width=True):
        reset_session()
        st.rerun()

    st.sidebar.caption(f"Thread ID: `{st.session_state['thread_id'][:8]}…`")

    with st.sidebar.expander("ℹ️ How it works"):
        st.markdown(
            "Multi-agent LangGraph workflow:\n\n"
            "**Requirement → Planner → Property Search → "
            "(Location & Commute ∥ Market Price ∥ Neighborhood ∥ Verification) "
            "→ Risk (fan-in) → Ranking → Decision → Human Review**\n\n"
            "Describe what you're looking for, review the recommendation, and either "
            "**approve** it or reply with changes (e.g. *\"raise budget to 45000\"*) "
            "to trigger a replan — only the affected agents re-run."
        )

    return openai_key, tavily_key, uploaded


# ---------------------------------------------------------------------------
# 7. Main app
# ---------------------------------------------------------------------------
def main():
    init_session()
    st.title("🏠 Intelligent Property Search & Decision Agent")
    st.caption(
        "Natural-language property search over Delhi rental listings, powered by a "
        "stateful multi-agent LangGraph workflow."
    )

    openai_key, tavily_key, uploaded_csv = sidebar()

    if not openai_key:
        st.error(
            "❌ **OPENAI_API_KEY** not found. Add it to a `.env` file next to `app.py` "
            "(e.g. `OPENAI_API_KEY=sk-...`) and restart the app."
        )
        return

    os.environ["OPENAI_API_KEY"] = openai_key
    if tavily_key:
        os.environ["TAVILY_API_KEY"] = tavily_key

    # Load data
    try:
        csv_bytes = uploaded_csv.getvalue() if uploaded_csv is not None else None
        df, locality_avg_psf = load_data(csv_bytes)
    except FileNotFoundError:
        st.error(
            f"Couldn't find `{os.path.basename(CSV_PATH)}` next to app.py. "
            "Upload a CSV from the sidebar instead."
        )
        return
    except Exception as e:
        st.error(f"Failed to load/clean data: {e}")
        return

    st.caption(f"📦 {len(df):,} usable listings loaded.")

    # Build (or rebuild) the graph once per session — cheap, cached in session_state
    if st.session_state["agent"] is None:
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_key)
            web_search = make_web_search(tavily_key)
            st.session_state["agent"] = build_graph(df, locality_avg_psf, llm, web_search)
        except Exception as e:
            st.error(f"Failed to initialize the agent graph: {e}")
            return

    agent = st.session_state["agent"]
    config = {"configurable": {"thread_id": st.session_state["thread_id"]}}

    # --- Render chat history --------------------------------------------------
    for msg in st.session_state["chat"]:
        with st.chat_message(msg["role"]):
            if msg.get("type") == "result":
                render_result_panel(msg["content"])
            else:
                st.markdown(msg["content"])

    # --- Chat input -------------------------------------------------------
    placeholder = (
        "Reply 'yes' to approve, or describe what to change…"
        if st.session_state["awaiting_approval"]
        else "e.g. 2BHK in Rohini or Dwarka under 35000 rent, low risk, within 25km of Cyber City Gurgaon"
    )
    user_input = st.chat_input(placeholder)

    if user_input:
        st.session_state["chat"].append({"role": "user", "content": user_input, "type": "text"})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Agents at work… (Requirement → Planner → Search → parallel specialists → Risk → Ranking → Decision)"):
                try:
                    from langgraph.types import Command
                    if st.session_state["awaiting_approval"]:
                        result = agent.invoke(Command(resume=user_input), config=config)
                    else:
                        result = agent.invoke(
                            {"user_query": user_input, "iteration": 0}, config=config
                        )
                except Exception as e:
                    st.error(f"Agent run failed: {e}")
                    return

            interrupt_info = result.get("__interrupt__")
            if interrupt_info:
                st.session_state["awaiting_approval"] = True
                render_result_panel(result)
                question = interrupt_info[0].value["question"]
                st.info(f"⏸️ {question}")
                st.session_state["chat"].append({"role": "assistant", "content": result, "type": "result"})
                st.session_state["chat"].append({"role": "assistant", "content": f"⏸️ {question}", "type": "text"})
            else:
                st.session_state["awaiting_approval"] = False
                st.success("✅ Approved — final recommendation below.")
                render_result_panel(result)
                st.session_state["chat"].append({"role": "assistant", "content": result, "type": "result"})
                st.session_state["chat"].append({"role": "assistant", "content": "✅ Approved — final recommendation above.", "type": "text"})

            st.session_state["last_result"] = result


if __name__ == "__main__":
    main()