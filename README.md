# 🐇 Talking Rabbitt — AI-Powered Hybrid BI Dashboard

Talking Rabbitt is an advanced, next-generation Business Intelligence (BI) dashboard built for hackathons. It bridges the gap between structured tabular metrics (`.csv`) and unstructured corporate strategy documentation (`.txt`/`.md`) using a dual-engine pipeline that combines deterministic analytical plotting with a semantic RAG (Retrieval-Augmented Generation) text matching framework.

👉 **Deployed App:** [Insert your Streamlit Share link here]

## 🚀 The Core Problem It Solves
Traditional BI tools like Tableau or PowerBI are excellent at showing **what** happened (e.g., *sales dropped by 20% in March*), but they cannot tell you **why** (e.g., *a 10-day warehouse automation system failure halted shipments*). 

Talking Rabbitt solves this by allowing users to pair a traditional data spreadsheet with qualitative text notes. The application dynamically figures out whether a user's question requires mathematical calculation, data visualization, or textual context lookup, delivering complete executive answers in real-time.

---

## ✨ Key Features

* **Dual Context Pipeline (Hybrid RAG):** Simultaneously processes a tabular data grid via automated context injection and utilizes a local sentence-transformer vector embedding pipeline (`all-MiniLM-L6-v2`) to chunk and fetch semantic text references.
* **Smart Schema Validation & Fallback:** Includes a strict validation layer that dynamically separates textual RAG queries from visual plotting queries—preventing data-refusal loops and ensuring zero-crash performance during live demos.
* **Deterministic Chart Engine:** Automatically interprets the AI's structural JSON output to render high-impact interactive Plotly charts (`bar`, `line` with OLS trendlines, `pie`, `heatmap`, or `bubble`).
* **🔮 What-If Financial Projections:** Interactive sidebar sliders allow users to tweak variable parameters (such as Corporate Tax Rates and Target Growth Rates) to run dynamic simulation models.
* **Auto-Generated Strategic Insights:** One-click business health analysis and dynamic intent-path questions generated directly from your specific data matrix.

---

## 🛠️ Tech Stack & Architecture

* **Frontend & UI:** Streamlit (Custom Light/Dark responsive grid styling)
* **Orchestration & Reasoning LLM:** Groq API Cloud
* **Vector Embeddings (RAG Engine):** `sentence-transformers/all-MiniLM-L6-v2` (Local execution)
* **Data Engine & Visualizations:** Pandas, NumPy, Plotly Express, Statsmodels

---

## 💻 Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/talking-rabbitt-bi.git](https://github.com/your-username/talking-rabbitt-bi.git)
   cd talking-rabbitt-bi