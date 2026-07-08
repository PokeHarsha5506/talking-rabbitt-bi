"""
Talking Rabbitt - AI Powered Business Intelligence Dashboard
Single-file Streamlit app with Account Chat Persistence, Auto-RAG, & Simulated Real-Time Streaming Ingestion.
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
CHAT_MODEL = "openai/gpt-oss-120b"  
DB_FILE = "users.json"

st.set_page_config(page_title="Talking Rabbitt", page_icon="🐇", layout="wide")

REFUSAL = (
    "I couldn't find anything in the uploaded data, live stream, or knowledge documents related to that question. "
    "Please ask something about the columns in your active data view: {cols}."
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
# DATABASE LAYER (ACCOUNT & ACCOUNT CHAT PERSISTENCE)
# ----------------------------------------------------------------------------
def load_user_db() -> dict:
    if not os.path.exists(DB_FILE):
        initial_db = {
            "admin": {
                "password": hashlib.sha256("password123".encode()).hexdigest(),
                "history": []
            }
        }
        with open(DB_FILE, "w") as f:
            json.dump(initial_db, f)
        return initial_db
    try:
        with open(DB_FILE, "r") as f:
            db = json.load(f)
        
        migrated = False
        for username, data in list(db.items()):
            if isinstance(data, str):
                db[username] = {"password": data, "history": []}
                migrated = True
        if migrated:
            save_user_db(db)
        return db
    except Exception:
        return {}


def save_user_db(db: dict):
    with open(DB_FILE, "w") as f:
        json.dump(db, f, indent=2)


def sync_user_chat_to_disk():
    if st.session_state.get("authenticated") and st.session_state.get("username"):
        db = load_user_db()
        username = st.session_state["username"]
        if username in db and isinstance(db[username], dict):
            serializable_history = []
            if "history" in st.session_state and isinstance(st.session_state.history, list):
                for msg in st.session_state.history:
                    if isinstance(msg, dict) and "role" in msg and "content" in msg:
                        serializable_history.append({"role": msg["role"], "content": msg["content"]})
            db[username]["history"] = serializable_history
            save_user_db(db)


def hash_string(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ----------------------------------------------------------------------------
# AUTHENTICATION LAYER
# ----------------------------------------------------------------------------
def check_authentication():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        st.title("🐇 Welcome to Talking Rabbitt")
        auth_mode = st.tabs(["🔒 Account Login", "✨ Create Account"])
        user_db = load_user_db()

        with auth_mode[0]:
            with st.form("login_form"):
                username = st.text_input("Username", key="login_user").strip()
                password = st.text_input("Password", type="password", key="login_pass")
                submit_login = st.form_submit_button("Sign In")
                
                if submit_login:
                    if not username or not password:
                        st.error("Fields cannot be blank.")
                    elif username in user_db and isinstance(user_db[username], dict) and user_db[username].get("password") == hash_string(password):
                        st.session_state["authenticated"] = True
                        st.session_state["username"] = username
                        saved_history = user_db[username].get("history", [])
                        st.session_state["history"] = [{"role": m["role"], "content": m["content"], "fig": None} for m in saved_history]
                        st.success(f"Access Authorized! Restoring your workspace...")
                        st.rerun()
                    else:
                        st.error("Invalid username or matching password.")

        with auth_mode[1]:
            with st.form("signup_form"):
                new_user = st.text_input("Choose Username", key="reg_user").strip()
                new_pass = st.text_input("Choose Password", type="password", key="reg_pass")
                confirm_pass = st.text_input("Confirm Password", type="password", key="reg_confirm")
                submit_signup = st.form_submit_button("Register Securely")
                
                if submit_signup:
                    if len(new_user) < 3:
                        st.error("Username must be at least 3 characters long.")
                    elif len(new_pass) < 6:
                        st.error("Password must be at least 6 characters long.")
                    elif new_pass != confirm_pass:
                        st.error("Password keys do not match.")
                    elif new_user in user_db:
                        st.error("Username already exists.")
                    else:
                        user_db[new_user] = {"password": hash_string(new_pass), "history": []}
                        save_user_db(user_db)
                        st.success("Registration complete! Switch to the login tab to connect.")
        st.stop()


# ----------------------------------------------------------------------------
# AUTOMATED STRUCTURAL CORPUS GENERATOR (AUTO-RAG)
# ----------------------------------------------------------------------------
def generate_auto_rag_text(df: pd.DataFrame) -> list:
    text_chunks = []
    numeric_cols = df.select_dtypes("number").columns.tolist()
    
    for col in numeric_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        col_mean = df[col].mean()
        chunk = (
            f"Regarding the column '{col}': The values range from a minimum of {col_min:.2f} "
            f"to a maximum of {col_max:.2f}. The overall average value observed across the entire "
            f"dataset is {col_mean:.2f}."
        )
        text_chunks.append(chunk)

    cat_cols = [c for c in df.columns if c not in numeric_cols]
    for col in cat_cols:
        value_counts = df[col].value_counts()
        total_unique = len(value_counts)
        top_values = value_counts.head(5).to_dict()
        summary = f"The categorical column '{col}' contains {total_unique} distinct classifications. "
        distribution_str = ", ".join([f"'{k}' appearing {v} times" for k, v in top_values.items()])
        summary += f"The most frequent profiles include: {distribution_str}."
        text_chunks.append(summary)

    return text_chunks


def get_relevant_context(query: str, text_chunks: list, top_k=2) -> str:
    if not text_chunks:
        return ""
    model = load_embedding_model()
    chunk_embeddings = model.encode(text_chunks, convert_to_numpy=True)
    query_embedding = model.encode([query], convert_to_numpy=True)[0]
    scores = np.dot(chunk_embeddings, query_embedding) / (
        np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    top_indices = np.argsort(scores)[::-1][:top_k]
    relevant_passages = [text_chunks[idx] for idx in top_indices if scores[idx] > 0.20]
    return "\n---\n".join(relevant_passages)


# ----------------------------------------------------------------------------
# LOGIC & SYSTEM TEMPLATE SETUP
# ----------------------------------------------------------------------------
def get_client():
    key = st.session_state.get("groq_key") or os.environ.get("GROQ_API_KEY")
    return Groq(api_key=key) if key else None

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def build_context(df: pd.DataFrame) -> str:
    numeric_cols = df.select_dtypes("number").columns.tolist()
    cat_cols = [c for c in df.columns if c not in numeric_cols]
    lines = [f"Total Cumulative Rows (CSV + Streaming): {len(df)}", f"Columns: {list(df.columns)}", ""]
    if numeric_cols:
        lines.append(df[numeric_cols].describe().round(2).to_string())
    lines.append("\nSample rows:")
    lines.append(df.tail(3).to_string())
    return "\n".join(lines)


SYSTEM_TEMPLATE = """You are the analytics brain of "Talking Rabbitt", a BI dashboard.
You have access to a dataset containing integrated historical files and real-time streaming data metrics.

Respond to the user's question with ONLY a single JSON object matching this schema:
{{
  "in_scope": true or false,
  "answer": "conversational answer parsing computational findings or streaming trends, 2-4 sentences",
  "chart_type": "bar" | "line" | "pie" | "heatmap" | "bubble" | "none",
  "x": "column name to use on x-axis, or null",
  "y": "column name on y-axis, or null",
  "value": "numeric column for sizing, or null",
  "color": "column name for grouping, or null",
  "agg": "sum" | "mean" | "count",
  "insight": "one short actionable live strategy note, or null"
}}
"""

def call_groq(client, context: str, question: str, history: list):
    messages = [{"role": "system", "content": SYSTEM_TEMPLATE.format(context=context)}]
    messages.extend(history[-6:])
    messages.append({"role": "user", "content": question})
    resp = client.chat.completions.create(model=CHAT_MODEL, messages=messages, temperature=0.2, response_format={"type": "json_object"})
    return json.loads(resp.choices[0].message.content)


def validate(resp: dict, df: pd.DataFrame) -> dict:
    if not resp.get("in_scope"):
        resp["answer"] = REFUSAL.format(cols=", ".join(df.columns))
        resp["chart_type"] = "none"
        return resp
    return resp


def make_chart(resp: dict, df: pd.DataFrame, template: str):
    ct, x, y, agg = resp.get("chart_type"), resp.get("x"), resp.get("y"), resp.get("agg") or "sum"
    if ct == "none" or not ct or x not in df.columns:
        return None
    try:
        if ct == "bar" and y in df.columns:
            grp = df.groupby(x, dropna=False)[y].agg(agg).reset_index()
            return px.bar(grp, x=x, y=y, title=f"{agg.title()} of {y} by {x}").update_layout(template=template)
        elif ct == "line" and y in df.columns:
            return px.line(df.sort_values(x), x=x, y=y, title=f"{y} over Time").update_layout(template=template)
    except:
        pass
    return None

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df.copy()
    with st.sidebar.expander("🔎 Filters", expanded=False):
        num_cols = df.select_dtypes("number").columns.tolist()
        for c in num_cols[:1]:
            lo, hi = float(df[c].min()), float(df[c].max())
            if lo < hi:
                sel = st.slider(c, min_value=lo, max_value=hi, value=(lo, hi))
                filtered = filtered[filtered[c].between(sel[0], sel[1])]
    return filtered


# ----------------------------------------------------------------------------
# RUNTIME ENGINE CONTROL
# ----------------------------------------------------------------------------
if "history" not in st.session_state: st.session_state.history = []
if "stream_buffer" not in st.session_state: st.session_state.stream_buffer = []

check_authentication()  

with st.sidebar:
    st.header(f"Welcome, {st.session_state.get('username')}! 👋")
    if st.button("Log Out"):
        sync_user_chat_to_disk()
        st.session_state["authenticated"] = False
        st.session_state["history"] = []
        st.session_state["stream_buffer"] = []
        st.rerun()
        
    st.markdown("---")
    st.header("Setup")
    key_input = st.text_input("Groq API Key", type="password", placeholder="Paste API Key here...")
    if key_input: st.session_state["groq_key"] = key_input

    theme_choice = st.radio("Theme Layout", ["Light", "Dark"], horizontal=True)
    
    st.markdown("---")
    st.header("📥 Ingestion Vectors")
    ingest_mode = st.radio("Primary Source Mode", ["Static CSV Upload", "Live Stream Ingest"])
    
    file = st.file_uploader("Upload Base CSV Schema", type=["csv"])

    # ---- LIVE STREAM SIMULATED INGESTION HUB ----
    st.markdown("### ⚡ Live Stream Buffer")
    with st.form("streaming_event_form", clear_on_submit=True):
        st.caption("Inject a real-time event payload directly into the analytical processing architecture.")
        
        # Build dynamic element generators depending on loaded column structure
        stream_payload = st.text_area("JSON Event Data", placeholder='{"Metric": 140, "Label": "Region-D"}')
        submit_event = st.form_submit_button("Push Live Event 🚀")
        
        if submit_event and stream_payload:
            try:
                parsed_event = json.loads(stream_payload)
                st.session_state.stream_buffer.append(parsed_event)
                st.toast("Streaming packet appended cleanly to active pipeline buffer!", icon="⚡")
            except Exception as e:
                st.error(f"Invalid Stream Event Structure: {e}")

    if st.button("Clear Streaming Buffer 🗑️"):
        st.session_state.stream_buffer = []
        st.rerun()

plotly_template = inject_theme(theme_choice)

st.title("🐇 Talking Rabbitt — Dashboard Engine")
st.caption("Hybrid data matrix view handling synchronous file tables and asynchronous streaming inputs concurrently.")

# Compile combined internal tracking dataframe matrices
base_df = None
if file:
    base_df = pd.read_csv(file)
    base_df.columns = [c.strip() for c in base_df.columns]

# Resolve and merge dataframes if live items are present in buffer
if st.session_state.stream_buffer:
    stream_df = pd.DataFrame(st.session_state.stream_buffer)
    if base_df is not None:
        # Align columns safely
        for col in base_df.columns:
            if col not in stream_df.columns: stream_df[col] = np.nan
        df = pd.concat([base_df, stream_df], ignore_index=True)
    else:
        df = stream_df
elif base_df is not None:
    df = base_df
else:
    st.info("Awaiting active metric feeds. Upload a CSV template or fire simulated packets in the sidebar stream form.")
    if st.session_state.history:
        for turn in st.session_state.history:
            with st.chat_message(turn["role"]): st.write(turn["content"])
    st.stop()

# Complete dynamic dashboard transformations
df = apply_filters(df)
context = build_context(df)
client = get_client()

# --- Monitor Grid Panel -----------------------------------------------------
st.subheader("Unified Analytics Pipeline Monitor")
m_cols = st.columns(3)
m_cols[0].metric("Base Storage Elements", len(base_df) if base_df is not None else 0)
m_cols[1].metric("Live Stream Packets Ingested", len(st.session_state.stream_buffer))
m_cols[2].metric("Aggregated Frame Volume", len(df))

with st.expander("Inspect Current Merged Data Frame View"):
    st.dataframe(df.tail(20), use_container_width=True)

with st.spinner("Processing real-time Auto-RAG indexes..."):
    auto_rag_chunks = generate_auto_rag_text(df)

# --- Chat Stream Interface --------------------------------------------------
st.subheader("Ask Talking Rabbitt")
for turn in st.session_state.history:
    with st.chat_message(turn["role"]): st.write(turn["content"])

question = st.chat_input("Ask about active aggregations or live updates...")
if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"): st.write(question)

    if client is None:
        answer, fig = "Please add your Groq API key in the sidebar first.", None
    else:
        with st.spinner("Evaluating combined framework vectors..."):
            try:
                extended_context = context + f"\n\nLIVE EVENTS ATTACHED:\n{get_relevant_context(question, auto_rag_chunks)}"
                raw = call_groq(client, extended_context, question, [{"role": t["role"], "content": t["content"]} for t in st.session_state.history[:-1]])
                raw = validate(raw, df)
                answer = raw["answer"]
                fig = make_chart(raw, df, plotly_template) if raw.get("in_scope") else None
                if raw.get("insight") and raw.get("in_scope"): answer += f"\n\n💡 Live Alert: {raw['insight']}"
            except Exception as e:
                answer, fig = f"Internal processing anomaly: {e}", None

    with st.chat_message("assistant"):
        st.write(answer)
        if fig: st.plotly_chart(fig, use_container_width=True)

    st.session_state.history.append({"role": "assistant", "content": answer, "fig": None})
    sync_user_chat_to_disk()
