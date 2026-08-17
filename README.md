# Survey CSV Analyzer v5

POC delivery build of Survey CSV Analyzer.

Upload a CSV or Excel file, inspect preflight audit warnings, let the LLM create an analysis plan, run statistical processing in Python, and interpret results with the Toya Analysis Ontology.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Set `OPENAI_API_KEY` in your environment or Streamlit secrets before running LLM analysis.
