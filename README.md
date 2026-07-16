# Insight Autopilot

An agentic Business Intelligence assistant that turns structured data into actionable insights. Upload any dataset (CSV, Excel, or SQL) and ask a plain-English question — Insight Autopilot will profile the data, create an analysis plan using an AI agent, automatically select and train the right ML model, explain predictions, score the confidence of every insight, and generate a stakeholder-ready report.

**Tech Stack:** Python · Claude (Anthropic via Kintio) · scikit-learn · SciPy · Streamlit · Plotly · SQLite

This project uses the Claude API for the planning and reporting agent tasks. It also supports running completely offline (using a deterministic heuristic planner and template report writer) if no API key is provided.

---

## Features

1. **Robust Ingestion**: Loads data from CSV, Excel, or SQL via SQLAlchemy into a Pandas DataFrame.
2. **Automated Profiling**: Extracts column types, missingness, distributions, correlations, and infers likely prediction targets.
3. **Agentic Planning**: A Claude-powered AI agent reads the dataset profile and the user's question, then structures an execution plan in JSON.
4. **Statistical Analysis**: Performs correlation analysis, outlier detection (Isolation Forest), distribution analysis, trend decomposition, and segment comparison (ANOVA).
5. **AutoML Pipeline**: Automatically selects regression, classification, or clustering models, builds leak-free scikit-learn pipelines, and cross-validates across algorithms.
6. **Plain English Explainability**: Translates complex ML concepts into business logic using permutation feature importance.
7. **Confidence Scoring (0-1)**: Assigns a confidence score to every insight based on data quality, statistical significance, and bootstrap stability.
8. **Automated Reporting**: A second Claude call summarizes the findings into a business narrative (downloadable as HTML).
9. **Session History**: Persists past runs in a local SQLite database for easy retrieval.

---

## Quickstart

### 1. Environment Setup

Ensure you have Python 3.9 - 3.13 installed. Create a virtual environment and install dependencies:

```bash
cd insight_autopilot

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. API Configuration

To use the Claude-powered AI agent, you will need an Anthropic API key.

```bash
cp .env.example .env
```
Open `.env` and configure your API key:
```env
ANTHROPIC_API_KEY="your-api-key-here"
CLAUDE_MODEL="your-preferred-model-here"
```
*(If no API key is provided, the app falls back to a completely offline, heuristic-based pipeline.)*

### 3. Generate Demo Data (Optional)

We include a lightweight script to generate sample datasets:
```bash
python scripts/generate_samples.py
```

### 4. Run the Dashboard

```bash
streamlit run app.py
```

---

## Testing

The test suite runs fully offline (no API key required) and covers ingestion, profiling, model selection, confidence scoring, storage, and the end-to-end pipeline.

```bash
pip install pytest
pytest -q
```

---

## Deployment (Streamlit Cloud)

To deploy for free on Streamlit Cloud:
1. Push this repository to GitHub.
2. Go to **share.streamlit.io** → **New app** and point it to `app.py`.
3. Go to **Advanced settings → Secrets** and add:
   ```toml
   ANTHROPIC_API_KEY = "your-api-key-here"
   CLAUDE_MODEL = "your-preferred-model-here"
   ```
*(Note: Since Streamlit Cloud's free tier has an ephemeral filesystem, the SQLite session history will reset on redeployment, which the app handles gracefully.)*

---

## License

This project is open-source under the MIT License. Built as an advanced portfolio project for Data Science and BI engineering roles.
