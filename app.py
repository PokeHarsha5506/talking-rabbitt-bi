"""
Talking Rabbitt - AI Powered Business Intelligence Dashboard
Single-file Streamlit app with Hybrid Structured Analysis & Semantic RAG Text Matching.
"""

import hashlib
import io
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
CHAT_MODEL = "openai/gpt-oss-120b"  # fast + strong reasoning on Groq.

st.set_page_config(page_title="Talking Rabbitt", page_icon="🐇", layout="wide")

REFUSAL = (
    "I couldn't find anything in the uploaded data or knowledge documents related to that question. "
    "Please ask something about the columns in your file: {cols} or your text references."
)

# ----------------------------------------------------------------------------
# THEME INJECTOR
# ----------------------------------------------------------------------------
def inject_theme(theme: str):
    if theme == "Dark":
        vals = dict(
            bg="#0e1117", fg="#e8e8ea", sidebar_bg="#161a23",
            card_bg="#1b1f2b", border="#2a2f3d", chip_bg="rgba(124,92,252,0.18)",
        )
    else:
        vals = dict(
            bg="#fafafa", fg="#1f2430", sidebar_bg="#f2f0fb",
            card_bg="#ffffff", border="#e6e3f5", chip_bg="rgba(124,92,252,0.10)",
        )

    st.markdown(
        f"""
        <style>
        :root {{ --accent: #7C5CFC; }}
        .stApp {{ background-color: {vals['bg']}; color: {vals['fg']}; }}
        section[data-testid="stSidebar"] {{ background-color: {vals['sidebar_bg']}; }}
        div[data-testid="stMetric"] {{
            background: {vals['card_bg']};
            border: 1px solid {vals['border']};
            border-radius: 14px;
            padding: 10px 16px;
        }}
        div[data-testid="stChatMessage"] {{
            border-radius: 16px;
            border: 1px solid {vals['border']};
        }}
        button[kind="secondary"] {{
            border-radius: 999px !important;
        }}
        h1, h2, h3 {{ font-family: 'Trebuchet MS', 'Segoe UI', sans-serif; }}
        .rabbitt-caption {{ opacity: 0.75; font-size: 0.9rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    return "plotly_dark" if theme == "Dark" else "plotly_white"


# ----------------------------------------------------------------------------
# GROQ CLIENT & EMBEDDING RESOURCE LOADERS
# ----------------------------------------------------------------------------
def get_client():
    key = st.session_state.get("groq_key") or os.environ.get("GROQ_API_KEY")
    if not key:
        return None
    return Groq(api_key=key)


@st.cache_resource
def load_embedding_model():
    """Loads and caches a compact, high-speed open-source semantic embedding model."""
    return SentenceTransformer("all-MiniLM-L6-v2")


# ----------------------------------------------------------------------------
# RAG ENGINE EXTRACTION & ALIGNMENT PIPELINE
# ----------------------------------------------------------------------------
def process_rag_text(raw_text: str, chunk_size=500, overlap=100) -> list:
    """Splits uploaded raw textual materials into smaller overlapping context vectors."""
    chunks = []
    start = 0
    while start < len(raw_text):
        end = start + chunk_size
        chunks.append(raw_text[start:end].strip())
        start += (chunk_size - overlap)
    return chunks


def get_relevant_context(query: str, text_chunks: list, top_k=2) -> str:
    """Performs lightning-fast serverless cosine similarity matching."""
    if not text_chunks:
        return ""
    
    model = load_embedding_model()
    chunk_embeddings = model.encode(text_chunks, convert_to_numpy=True)
    query_embedding = model.encode([query], convert_to_numpy=True)[0]
    
    # Mathematical dot product vector alignments
    scores = np.dot(chunk_embeddings, query_embedding) / (
        np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    
    top_indices = np.argsort(scores)[::-1][:top_k]
    relevant_passages = [text_chunks[idx] for idx in top_indices if scores[idx] > 0.25]
    return "\n---\n".join(relevant_passages)


# ----------------------------------------------------------------------------
# DATA TABULAR CONTEXT BUILDER
# ----------------------------------------------------------------------------
def build_context(df: pd.DataFrame) -> str:
    numeric_cols = df.select_dtypes("number").columns.tolist()
    cat_cols = [c for c in df.columns if c not in numeric_cols]

    lines = [f"Rows: {len(df)}", f"Columns: {list(df.columns)}", ""]

    if numeric_cols:
        lines.append("Numeric column stats:")
        lines.append(df[numeric_cols].describe().round(2).to_string())

    for c in cat_cols:
        uniq = df[c].dropna().unique()
        if len(uniq) <= 25:
            lines.append(f"\nValues in '{c}': {list(uniq)}")
        else:
            lines.append(f"\n'{c}' has {len(uniq)} unique values, e.g. {list(uniq[:8])}")

    lines.append("\nSample rows:")
    lines.append(df.head(3).to_string())
    
    tax_rate = st.session_state.get("var_tax_rate", 25)
    growth_rate = st.session_state.get("var_growth_rate", 12)
    lines.append(f"\nGLOBAL MODEL PARAMETERS (Use for financial projections if requested):")
    lines.append(f"- Baseline Corporate Tax Rate: {tax_rate}%")
    lines.append(f"- User Forecast Target Growth Rate: {growth_rate}%")
    
    return "\n".join(lines)


SYSTEM_TEMPLATE = """You are the analytics brain of "Talking Rabbitt", a BI dashboard.
You are allowed to use the structured dataset described below and qualitative knowledge documents to compute derived metrics, financial performance, and future outlooks.

FINANCIAL DERIVATION RULES:
- If columns representing revenue/sales/income and cost/expenses/spend are present, you CAN compute:
  * Profit/Loss = Revenue - Cost (Mapped to the virtual column: 'Derived_Profit')
  * Tax Deduction = Profit * Tax Rate (Mapped to the virtual column: 'Derived_Tax')
- Never invent base numbers out of nowhere, but you are explicitly encouraged to mathematically derive these values.

FUTURE FORECASTING & UNSTRUCTURED TEXT COUPLING RULES:
- If asked to predict or forecast trajectories, analyze historical data paths or draw contextual answers directly from the qualitative context snippets.
- Set `chart_type` to 'line' or 'bar' and map the target metric to 'Derived_Forecast' to visualize trends. If the inquiry is purely text-based and doesn't require a plot, return `chart_type` as 'none'.

DATASET CONTEXT:
{context}

Respond to the user's question with ONLY a single JSON object (no markdown, no prose outside the JSON) matching this schema:
{{
  "in_scope": true or false,
  "answer": "conversational answer containing computations, text match clarifications, or predictions, 2-4 sentences",
  "chart_type": "bar" | "line" | "pie" | "heatmap" | "bubble" | "none",
  "x": "column name to use on x-axis, or null",
  "y": "column name or virtual column ('Derived_Profit', 'Derived_Tax', 'Derived_Forecast') on y-axis, or null",
  "value": "numeric column for heatmap cell value / bubble size, or null",
  "color": "column name for grouping/color, or null",
  "agg": "sum" | "mean" | "count",
  "insight": "one short actionable business recommendation or data realization based on your calculations/notes, or null"
}}

Rules for Qualitative/RAG questions:
- If the question is answered by the qualitative document context and does NOT require drawing a chart, set "chart_type" to "none", set "x": null, "y": null, and set "in_scope": true.
- Only reference column names that literally appear in the dataset context, or the specified virtual columns ('Derived_Profit', 'Derived_Tax', 'Derived_Forecast').
"""


def call_groq(client, context: str, question: str, history: list):
    messages = [{"role": "system", "content": SYSTEM_TEMPLATE.format(context=context)}]
    messages.extend(history[-6:])
    messages.append({"role": "user", "content": question})

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def generate_suggested_questions(client, context: str) -> list:
    prompt = (
        "Suggest 5 short, specific business user questions about "
        "this dataset and available company notes (max ~8 words each). Mix analytics, financial profit/tax margins, and strategy queries. Return ONLY a JSON object: "
        '{"questions": ["...", "..."]}'
    )
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": f"Dataset context:\n{context}"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        return [q for q in data.get("questions", []) if isinstance(q, str)][:5]
    except Exception:
        return []


# ----------------------------------------------------------------------------
# VALIDATION LAYER
# ----------------------------------------------------------------------------
def validate(resp: dict, df: pd.DataFrame) -> dict:
    # If the AI marked it out of scope entirely, reject it early
    if not resp.get("in_scope"):
        resp["answer"] = REFUSAL.format(cols=", ".join(df.columns))
        resp["chart_type"] = "none"
        resp["insight"] = None
        return resp

    # Only enforce strict CSV column validation if the AI actually wants to build a visual plot
    if resp.get("chart_type") != "none":
        VIRTUAL_COLUMNS = {"Derived_Profit", "Derived_Tax", "Derived_Forecast"}
        cols = set(df.columns).union(VIRTUAL_COLUMNS)
        referenced = [resp.get(k) for k in ("x", "y", "value", "color") if resp.get(k)]
        
        if any(r not in cols for r in referenced):
            resp["in_scope"] = False
            resp["answer"] = REFUSAL.format(cols=", ".join(df.columns))
            resp["chart_type"] = "none"
            resp["insight"] = None

    return resp


# ----------------------------------------------------------------------------
# DETERMINISTIC CALCULATION & PLOT ENGINE
# ----------------------------------------------------------------------------
def make_chart(resp: dict, df: pd.DataFrame, template: str):
    ct, x, y, val, color, agg = (
        resp.get("chart_type"), resp.get("x"), resp.get("y"),
        resp.get("value"), resp.get("color"), resp.get("agg") or "sum",
    )
    if ct == "none" or not ct:
        return None
    
    working_df = df.copy()
    rev_col = next((c for c in working_df.columns if any(k in c.lower() for k in ["sale", "revenue", "income", "total"])), None)
    cost_col = next((c for c in working_df.columns if any(k in c.lower() for k in ["cost", "expense", "spend", "fee"])), None)
    
    tax_factor = st.session_state.get("var_tax_rate", 25) / 100.0
    growth_factor = 1 + (st.session_state.get("var_growth_rate", 12) / 100.0)

    if rev_col and cost_col:
        working_df["Derived_Profit"] = working_df[rev_col] - working_df[cost_col]
        working_df["Derived_Tax"] = working_df["Derived_Profit"].apply(lambda v: v * tax_factor if v > 0 else 0)
    elif rev_col:
        working_df["Derived_Profit"] = working_df[rev_col] * 0.30 
        working_df["Derived_Tax"] = working_df["Derived_Profit"] * tax_factor
    else:
        working_df["Derived_Profit"] = 0
        working_df["Derived_Tax"] = 0

    if rev_col:
        working_df["Derived_Forecast"] = working_df[rev_col] * growth_factor
    else:
        numeric_cols = working_df.select_dtypes("number").columns.tolist()
        working_df["Derived_Forecast"] = working_df[numeric_cols[0]] * growth_factor if numeric_cols else 0

    try:
        fig = None
        if ct == "bar":
            if y and x:
                grp = working_df.groupby(x, dropna=False)[y].agg(agg).reset_index()
                fig = px.bar(grp, x=x, y=y, color=color if color in working_df.columns else None,
                             title=f"{agg.title()} of {y.replace('Derived_', '')} by {x}")
            elif x:
                grp = working_df[x].value_counts().reset_index()
                grp.columns = [x, "count"]
                fig = px.bar(grp, x=x, y="count", title=f"Count by {x}")

        elif ct == "line":
            if x and y:
                d = working_df[[x, y]].dropna().sort_values(x)
                try:
                    import statsmodels
                    fig = px.line(d, x=x, y=y, title=f"{y.replace('Derived_', '')} Outlook over {x}")
                    trend_fig = px.scatter(d, x=x, y=y, trendline="ols")
                    trend_line = trend_fig.data[1]
                    trend_line.line.dash = "dash"
                    trend_line.name = "Trendline"
                    fig.add_trace(trend_line)
                except Exception:
                    fig = px.line(d, x=x, y=y, title=f"{y.replace('Derived_', '')} Outlook over {x}")

        elif ct == "pie":
            if x:
                if y:
                    grp = working_df.groupby(x, dropna=False)[y].agg(agg).reset_index()
                    fig = px.pie(grp, names=x, values=y, title=f"{y.replace('Derived_', '')} share by {x}")
                else:
                    grp = working_df[x].value_counts().reset_index()
                    grp.columns = [x, "count"]
                    fig = px.pie(grp, names=x, values="count", title=f"Share by {x}")

        elif ct == "heatmap":
            if x and y and val:
                pivot = working_df.pivot_table(index=y, columns=x, values=val, aggfunc=agg)
                fig = px.imshow(pivot, text_auto=".1f", aspect="auto", title=f"{agg.title()} of {val} by {y} & {x}")
            else:
                numeric = working_df.select_dtypes("number")
                if numeric.shape[1] >= 2:
                    fig = px.imshow(numeric.corr(), text_auto=".2f", title="Correlation heatmap")

        elif ct == "bubble":
            if x and y:
                fig = px.scatter(
                    working_df, x=x, y=y, size=val if val in working_df.columns else None,
                    color=color if color in working_df.columns else None,
                    title=f"{y} vs {x}" + (f" (size: {val})" if val in working_df.columns else ""),
                )

        if fig is not None:
            fig.update_layout(template=template)
        return fig
    except Exception as e:
        st.warning(f"Couldn't render chart: {e}")
        return None


# ----------------------------------------------------------------------------
# AUTO STRATEGY GENERATOR
# ----------------------------------------------------------------------------
def auto_insights(client, df: pd.DataFrame, context: str) -> str:
    prompt = (
        "Based on the data context and configuration parameters, provide: "
        "(1) estimated general profit/loss health, (2) expected financial trajectory "
        "given the growth variables provided, (3) one concrete strategic recommendation. "
        "Keep it under 120 words, plain text, no markdown headers."
    )
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": f"Dataset context:\n{context}"}, {"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content


# ----------------------------------------------------------------------------
# DYNAMIC FRONTEND SLIDER FILTERS
# ----------------------------------------------------------------------------
def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df.copy()
    with st.sidebar.expander("🔎 Filters", expanded=False):
        date_cols = []
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                date_cols.append(c)
            elif df[c].dtype == object:
                try:
                    parsed = pd.to_datetime(df[c], errors="coerce")
                    if parsed.notna().mean() > 0.8:
                        date_cols.append(c)
                except Exception:
                    pass

        if date_cols:
            dc = st.selectbox("Date column", date_cols, key="filter_date_col")
            parsed_dates = pd.to_datetime(filtered[dc], errors="coerce")
            valid = parsed_dates.dropna()
            if not valid.empty and valid.min() < valid.max():
                min_d, max_d = valid.min().to_pydatetime(), valid.max().to_pydatetime()
                start, end = st.slider("Date range", min_value=min_d, max_value=max_d, value=(min_d, max_d), key="filter_date_range")
                filtered = filtered[parsed_dates.between(start, end)]

        cat_cols = [c for c in df.select_dtypes(exclude="number").columns if df[c].nunique() <= 50]
        for c in cat_cols[:3]:
            options = sorted(df[c].dropna().unique().tolist(), key=str)
            chosen = st.multiselect(c, options, default=options, key=f"filter_cat_{c}")
            if chosen and len(chosen) < len(options):
                filtered = filtered[filtered[c].isin(chosen)]

        num_cols = df.select_dtypes("number").columns.tolist()
        for c in num_cols[:2]:
            lo, hi = float(df[c].min()), float(df[c].max())
            if lo < hi:
                sel = st.slider(c, min_value=lo, max_value=hi, value=(lo, hi), key=f"filter_num_{c}")
                filtered = filtered[filtered[c].between(sel[0], sel[1])]

        if st.button("Reset filters"):
            for k in list(st.session_state.keys()):
                if k.startswith("filter_"):
                    del st.session_state[k]
            st.rerun()

    return filtered


# ----------------------------------------------------------------------------
# EXPORT HELPERS
# ----------------------------------------------------------------------------
def chat_history_csv(history: list) -> bytes:
    rows = [{"role": t["role"], "message": t["content"]} for t in history]
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


# ----------------------------------------------------------------------------
# APPLICATION ENTRY CONTROL
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Setup")
    key_input = st.text_input("Groq API Key", type="password", value=os.environ.get("GROQ_API_KEY", ""))
    if key_input:
        st.session_state["groq_key"] = key_input

    theme_choice = st.radio("Theme", ["Light", "Dark"], horizontal=True, key="theme")
    file = st.file_uploader("Upload CSV Data", type=["csv"])
    rag_file = st.file_uploader("Upload Qualitative Context (TXT/MD)", type=["txt", "md"])
    
    st.markdown("---")
    st.header("🔮 Simulation Models")
    st.slider("Corporate Tax Rate (%)", min_value=0, max_value=50, value=25, step=1, key="var_tax_rate")
    st.slider("Target Growth Projection (%)", min_value=-50, max_value=100, value=12, step=1, key="var_growth_rate")
    st.markdown("---")
    st.caption(f"Engine: {CHAT_MODEL}")

plotly_template = inject_theme(st.session_state.get("theme", "Light"))

if "history" not in st.session_state:
    st.session_state.history = []
if "insights_text" not in st.session_state:
    st.session_state.insights_text = None

st.title("🐇 Talking Rabbitt — AI Powered BI Dashboard")
st.caption("Upload raw tables and reference docs, then execute multi-modal simulation forecasts.")

if not file:
    st.info("Upload a CSV file from the sidebar to activate the analysis window.")
    st.stop()

raw_df = pd.read_csv(file)
raw_df.columns = [c.strip() for c in raw_df.columns]

df = apply_filters(raw_df)
context = build_context(df)

client = get_client()
if client is None:
    st.warning("Provide a valid Groq API Key within the sidebar to unlock AI computation.")

data_signature = hashlib.md5((file.name + str(raw_df.shape) + ",".join(raw_df.columns)).encode()).hexdigest()
if st.session_state.get("data_signature") != data_signature:
    st.session_state.data_signature = data_signature
    st.session_state.suggested_questions = []
    st.session_state.insights_text = None

# --- KPI Section ------------------------------------------------------------
numeric_cols = df.select_dtypes("number").columns.tolist()
st.subheader("Overview")
if len(df) != len(raw_df):
    st.caption(f"🔎 Filters active — showing {len(df)} of {len(raw_df)} matrix vectors")
kpi_cols = st.columns(min(4, max(1, len(numeric_cols) + 1)))
kpi_cols[0].metric("Rows Loaded", len(df))
for i, c in enumerate(numeric_cols[:3]):
    kpi_cols[i + 1].metric(c, round(df[c].sum(), 2))

with st.expander("Preview data grid"):
    st.dataframe(df.head(20), use_container_width=True)

if rag_file:
    st.success(f"📖 Reference Context Loaded: '{rag_file.name}' is actively paired for matching.")

# --- Action Hub -------------------------------------------------------------
st.subheader("AI Insights & Projections")
insight_cols = st.columns([1, 1, 3])
if insight_cols[0].button("Generate insights") and client:
    with st.spinner("Analyzing data vectors..."):
        st.session_state.insights_text = auto_insights(client, df, context)

if client and not st.session_state.get("suggested_questions"):
    st.session_state.suggested_questions = generate_suggested_questions(client, context)

if insight_cols[1].button("🔄 New suggestions") and client:
    with st.spinner("Re-indexing intent paths..."):
        st.session_state.suggested_questions = generate_suggested_questions(client, context)

if st.session_state.insights_text:
    st.info(st.session_state.insights_text)

# --- Export ---------------------------------------------------------------
st.download_button(
    "⬇️ Chat as CSV",
    data=chat_history_csv(st.session_state.history) if st.session_state.history else b"role,message\n",
    file_name="talking_rabbitt_history.csv",
    mime="text/csv",
    disabled=not st.session_state.history,
)

# --- Suggested Questions Chips ----------------------------------------------
if st.session_state.get("suggested_questions"):
    st.markdown('<p class="rabbitt-caption">💡 Try asking:</p>', unsafe_allow_html=True)
    chip_cols = st.columns(len(st.session_state.suggested_questions))
    chip_clicked = None
    for i, q in enumerate(st.session_state.suggested_questions):
        if chip_cols[i].button(q, key=f"chip_{i}"):
            chip_clicked = q
else:
    chip_clicked = None

# --- Chat Stream ------------------------------------------------------------
st.subheader("Ask Talking Rabbitt")
for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.write(turn["content"])
        if turn.get("fig"):
            st.plotly_chart(turn["fig"], use_container_width=True)

typed_question = st.chat_input("Ask anything about numbers, margins, or uploaded reference files...")
question = typed_question or chip_clicked

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    if client is None:
        answer, fig = "Please add your Groq API key in the sidebar first.", None
    else:
        with st.spinner("Processing analysis pipeline..."):
            try:
                # 1. Qualitative RAG Extraction Layer Pass
                extended_context = context
                if rag_file:
                    raw_text = rag_file.getvalue().decode("utf-8")
                    chunks = process_rag_text(raw_text)
                    relevant_passages = get_relevant_context(question, chunks, top_k=2)
                    if relevant_passages:
                        extended_context += f"\n\nRELEVANT QUALITATIVE DOC CONTEXT:\n{relevant_passages}"

                # 2. Complete Prompt Synthesis Execution Call
                raw = call_groq(
                    client, extended_context, question,
                    [{"role": t["role"], "content": t["content"]} for t in st.session_state.history[:-1]],
                )
                raw = validate(raw, df)
                answer = raw["answer"]
                fig = make_chart(raw, df, plotly_template) if raw.get("in_scope") else None
                if raw.get("insight") and raw.get("in_scope"):
                    answer += f"\n\n💡 {raw['insight']}"
            except Exception as e:
                error_message = str(e).lower()
                if "api_key" in error_message or "unauthorized" in error_message or "401" in error_message:
                    answer = (
                        "⚠️ **API Authentication Session Expired or Invalid**\n\n"
                        "Your underlying data modeling state, filters, and chat history have been successfully "
                        "preserved in-memory. Please update your token credential in the sidebar panel to "
                        "automatically resume this analytical sequence exactly where you left off."
                    )
                else:
                    answer = f"Something went wrong inside execution layers: {e}"
                fig = None

    with st.chat_message("assistant"):
        st.write(answer)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    st.session_state.history.append({"role": "assistant", "content": answer, "fig": fig})