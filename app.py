import io, os, json, re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import chi2_contingency, ttest_ind, pearsonr, spearmanr
from openai import OpenAI

st.set_page_config(
    page_title="Survey CSV Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE = Path(__file__).parent
ONTOLOGY = json.loads((BASE / "analysis_ontology.json").read_text(encoding="utf-8"))

# ---------- Visual system ----------
st.markdown("""
<style>
:root {
  --bg: #f6f7f9;
  --card: #ffffff;
  --text: #17191c;
  --muted: #6b7280;
  --border: #e6e8ec;
  --accent: #2d5bff;
  --accent-soft: #eef3ff;
  --warn-soft: #fff8e7;
  --stop-soft: #fff0f0;
  --go-soft: #effaf4;
}
.stApp { background: var(--bg); }
.block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 4rem; }
h1, h2, h3 { letter-spacing: -0.02em; }
[data-testid="stHeader"] { background: rgba(0,0,0,0); }
.hero {
  background: linear-gradient(135deg, #141821 0%, #232a3a 100%);
  color: white;
  border-radius: 22px;
  padding: 30px 34px;
  margin-bottom: 22px;
  box-shadow: 0 14px 40px rgba(20,24,33,.10);
}
.hero-kicker {
  font-size: 12px; letter-spacing: .12em; text-transform: uppercase;
  opacity: .65; font-weight: 700; margin-bottom: 8px;
}
.hero-title { font-size: 34px; font-weight: 750; margin: 0 0 8px 0; }
.hero-sub { font-size: 15px; line-height: 1.7; opacity: .78; max-width: 760px; }
.section-label {
  font-size: 12px; letter-spacing: .09em; text-transform: uppercase;
  color: var(--muted); font-weight: 700; margin: 10px 0 8px;
}
.panel {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 18px; padding: 20px 22px; margin: 10px 0 18px;
  box-shadow: 0 3px 14px rgba(0,0,0,.035);
}
.status {
  display: inline-block; border-radius: 999px; padding: 6px 11px;
  font-weight: 700; font-size: 12px; letter-spacing: .04em;
}
.status-go { background: var(--go-soft); color: #197447; }
.status-warn { background: var(--warn-soft); color: #956400; }
.status-stop { background: var(--stop-soft); color: #b42318; }
.flag {
  border: 1px solid var(--border); background: white; border-radius: 14px;
  padding: 13px 15px; margin: 8px 0;
}
.flag strong { font-size: 13px; }
.flag-stop { border-left: 4px solid #e24a4a; }
.flag-warn { border-left: 4px solid #e7a42b; }
.flag-info { border-left: 4px solid #7b8aa0; }
.answer-shell {
  background: #ffffff; border: 1px solid var(--border); border-radius: 18px;
  padding: 26px 28px; box-shadow: 0 8px 30px rgba(0,0,0,.045);
}
.answer-shell h2 { margin-top: 1.25rem; font-size: 1.25rem; }
.answer-shell h3 { margin-top: 1.1rem; font-size: 1.05rem; }
.small-note { color: var(--muted); font-size: 12px; line-height: 1.6; }
div[data-testid="stMetric"] {
  background: white; border: 1px solid var(--border); border-radius: 16px;
  padding: 14px 16px;
}
div[data-testid="stFileUploader"] {
  background: white; border: 1px dashed #cfd5df; border-radius: 18px; padding: 10px;
}
div[data-testid="stTextArea"] textarea {
  border-radius: 14px;
}
.stButton > button[kind="primary"] {
  border-radius: 12px; font-weight: 700; padding: .65rem 1rem;
}
</style>
""", unsafe_allow_html=True)


def secret(name, default=""):
    if os.getenv(name):
        return os.getenv(name)
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


def infer_type(s):
    x = s.dropna()
    if len(x) == 0:
        return "empty"
    num = pd.to_numeric(x, errors="coerce")
    if num.notna().mean() >= .95:
        if x.nunique() <= 12:
            return "categorical_numeric"
        return "numeric"
    if x.nunique() <= 20:
        return "categorical"
    return "text" if x.astype(str).str.len().mean() >= 25 else "categorical"


def profile(df):
    return pd.DataFrame([
        {
            "variable": str(c),
            "type": infer_type(df[c]),
            "valid_n": int(df[c].notna().sum()),
            "missing_pct": round(float(df[c].isna().mean() * 100), 2),
            "unique_n": int(df[c].nunique(dropna=True)),
        }
        for c in df.columns
    ])


def read_file(upload, sheet=None):
    raw = upload.getvalue()
    name = upload.name.lower()
    if name.endswith((".xlsx", ".xls")):
        xls = pd.ExcelFile(io.BytesIO(raw))
        sh = sheet or xls.sheet_names[0]
        return pd.read_excel(io.BytesIO(raw), sheet_name=sh), xls.sheet_names
    for enc in ["utf-8-sig", "utf-8", "cp932", "shift_jis"]:
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc), []
        except Exception:
            pass
    raise ValueError("CSV/Excelを読み込めませんでした")


def cronbach_alpha(frame):
    x = frame.apply(pd.to_numeric, errors="coerce").dropna()
    if x.shape[0] < 5 or x.shape[1] < 2:
        return None
    k = x.shape[1]
    total = x.sum(axis=1).var(ddof=1)
    if total <= 0:
        return None
    return float(k / (k - 1) * (1 - x.var(axis=0, ddof=1).sum() / total))


def batteries(df):
    g = {}
    for c in map(str, df.columns):
        m = re.match(r"^([A-Za-z_]+?)[_ -]?(\d+)$", c.strip())
        if m:
            g.setdefault(m.group(1).lower(), []).append(c)
    return {k: v for k, v in g.items() if len(v) >= 3}


def audit(df):
    flags = []
    insights = []

    for c in df.columns:
        vals = df[c].astype(str)
        n = vals.str.match(
            r"^#(NAME|VALUE|REF|DIV/0|N/A|NUM|NULL|SPILL|CALC)",
            case=False,
            na=False,
        ).sum()
        if n:
            flags.append({
                "severity": "STOP",
                "rule": "G_SCHEMA_VALIDATION",
                "message": f"{c}: セルエラー候補 {int(n)}件",
            })

        num = pd.to_numeric(df[c], errors="coerce")
        if num.notna().mean() > .7:
            neg = num[num.isin(range(-9, 0))].value_counts().to_dict()
            if neg:
                flags.append({
                    "severity": "WARN",
                    "rule": "G_SCHEMA_VALIDATION",
                    "message": f"{c}: 特殊欠損コード候補 {neg}",
                })

        nc = str(c).lower()
        if any(h in nc for h in ["date", "time", "day"]):
            x = num.dropna()
            if len(x) and ((x > 20000) & (x < 70000)).mean() > .8:
                flags.append({
                    "severity": "WARN",
                    "rule": "G_SCHEMA_VALIDATION",
                    "message": f"{c}: Excel/Sheets日付シリアルの可能性",
                })

    for c in df.columns:
        x = pd.to_numeric(df[c], errors="coerce").dropna()
        u = set(x.unique().tolist())
        if len(u) == 2 and u.issubset({0, 1}):
            rare = int(x.value_counts().min())
            if rare < 20:
                flags.append({
                    "severity": "WARN",
                    "rule": "G_STOP_RULE",
                    "message": f"{c}: 二値変数の少数側={rare}。多変量Logitは不安定になりやすい",
                })

    repeat = []
    for c in df.columns:
        nc = str(c).lower()
        s = df[c].dropna()
        if (
            any(h in nc for h in ["id", "series", "respondent", "participant", "country", "wave", "period", "stage"])
            and 1 < s.nunique() < len(s)
            and s.duplicated().any()
        ):
            repeat.append({
                "variable": str(c),
                "unique_n": int(s.nunique()),
                "rows": int(len(s)),
            })

    if repeat:
        flags.append({
            "severity": "WARN",
            "rule": "G_ROWS_NOT_UNITS",
            "message": "反復/クラスタ候補あり。行数と独立unit数を分けて確認",
            "details": repeat[:8],
        })

    bs = []
    for pfx, cols in batteries(df).items():
        a = cronbach_alpha(df[cols])
        bs.append({
            "prefix": pfx,
            "items": cols,
            "alpha": None if a is None else round(a, 3),
        })
        if a is not None and a < .6:
            flags.append({
                "severity": "WARN",
                "rule": "G_SCALE_BEFORE_MODEL",
                "message": f"{pfx}: Cronbach α={a:.3f} と低め",
            })

    if bs:
        insights.append({"scale_candidates": bs})

    if len(df) >= 10000:
        flags.append({
            "severity": "WARN",
            "rule": "G_LARGE_N",
            "message": f"大標本 N={len(df):,}。p値より効果量・検証を優先",
        })

    status = "STOP" if any(f["severity"] == "STOP" for f in flags) else ("WARN" if flags else "GO")
    return {"status": status, "flags": flags, "insights": insights, "row_n": len(df)}


def compact_context(df, p, a):
    vars_ = []
    for _, r in p.iterrows():
        c = r["variable"]
        s = df[c].dropna()
        examples = [] if r["type"] == "text" else [
            str(x)[:60] for x in s.astype(str).value_counts().head(4).index
        ]
        vars_.append({
            "name": c,
            "type": r["type"],
            "missing_pct": r["missing_pct"],
            "unique_n": int(r["unique_n"]),
            "examples": examples,
        })
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "variables": vars_,
        "audit": a,
        "ontology": ONTOLOGY["core_rules"],
    }


def ask_llm(client, model, question, context):
    instructions = """You are an analysis-planning assistant. Return concise Japanese.
Use only metadata and audit information provided; the full dataset is not sent.
Follow Toya Analysis Ontology: find pattern, search exceptions, reconstruct the data-generating process, only then model.
Do not claim causality.

Use EXACTLY these markdown sections, in this order:
## まず押さえること
Give 3-5 concise bullets with the most decision-relevant facts.

## 最優先リスク
Give only the highest-priority risks. Each bullet should start with **High**, **Medium**, or **Low**.

## 次にやること
Give a numbered sequence of 3-6 actions. Prefer concrete checks over generic advice.

## 分析の進め方
Give a compact analysis plan: pattern → exception → DGP → model. Only propose models after prerequisites.

## 補足
Put secondary cautions here. Keep this section short.

If rare outcomes, repeated units, weak scales, structural missingness, or schema issues matter, make them first-class findings.
Do not ask the user to send the full dataset again."""
    payload = {"question": question, "dataset_context": context}
    r = client.responses.create(
        model=model,
        instructions=instructions,
        input=json.dumps(payload, ensure_ascii=False),
    )
    return r.output_text


def status_html(status):
    cls = {"GO": "status-go", "WARN": "status-warn", "STOP": "status-stop"}[status]
    label = {"GO": "GO · 分析可能", "WARN": "WARN · 要確認", "STOP": "STOP · 先に修正"}[status]
    return f'<span class="status {cls}">{label}</span>'


def flag_html(flag):
    sev = flag["severity"].lower()
    cls = "flag-stop" if sev == "stop" else "flag-warn"
    return (
        f'<div class="flag {cls}">'
        f'<strong>{flag["severity"]} · {flag["rule"]}</strong><br>'
        f'<span style="color:#4b5563">{flag["message"]}</span>'
        f'</div>'
    )


# ---------- Header ----------
st.markdown("""
<div class="hero">
  <div class="hero-kicker">Survey CSV Analyzer · Beta</div>
  <div class="hero-title">データを見る前に、分析を疑う。</div>
  <div class="hero-sub">
    CSV / Excel を入れると、まずデータ構造・欠損・反復・尺度・rare event を監査し、
    そのうえで「何を見るべきか」「どこが危ないか」「次に何をするか」を整理します。
  </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 設定")
    api_key = secret("OPENAI_API_KEY", "")
    if api_key:
        st.success("API接続済み")
    else:
        api_key = st.text_input("OpenAI API Key", type="password")
    model = st.text_input("Model", value=secret("OPENAI_MODEL", "gpt-5-mini"))
    st.markdown("---")
    st.caption("CSV/Excel全文はLLMへ送信しません。列名・型・欠損率・少数のカテゴリ例・監査結果のみ送信します。")

st.markdown('<div class="section-label">01 · Upload</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "分析するCSV / Excelをアップロード",
    type=["csv", "xlsx", "xls"],
    help="CSV / XLSX / XLS に対応しています。",
)

if not uploaded:
    st.markdown(
        '<div class="panel"><b>まずデータを1つアップロードしてください。</b>'
        '<div class="small-note" style="margin-top:8px">'
        'アップロード後、Preflight Audit → 分析相談の順に進みます。'
        '</div></div>',
        unsafe_allow_html=True,
    )
    st.stop()

sheet = None
if uploaded.name.lower().endswith((".xlsx", ".xls")):
    xls = pd.ExcelFile(io.BytesIO(uploaded.getvalue()))
    if len(xls.sheet_names) > 1:
        sheet = st.selectbox("分析するシート", xls.sheet_names)
    else:
        sheet = xls.sheet_names[0]

df, _ = read_file(uploaded, sheet)
p = profile(df)
a = audit(df)

st.markdown('<div class="section-label">02 · Data health</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns([1, 1, 1, 1.15])
c1.metric("行数", f"{len(df):,}")
c2.metric("変数数", f"{len(df.columns):,}")
c3.metric("欠損あり変数", int((p["missing_pct"] > 0).sum()))
with c4:
    st.markdown(
        f'<div style="padding:10px 0 0 4px">'
        f'<div style="font-size:12px;color:#6b7280;margin-bottom:8px">PREFLIGHT</div>'
        f'{status_html(a["status"])}</div>',
        unsafe_allow_html=True,
    )

tab1, tab2, tab3 = st.tabs(["概要", "データ品質", "変数プロファイル"])

with tab1:
    if a["status"] == "GO":
        st.markdown(
            '<div class="panel"><b>重大な構造問題は自動検出されませんでした。</b>'
            '<div class="small-note" style="margin-top:8px">'
            'ただし「検出されない＝問題がない」ではありません。分析前に問いと観測単位を確認してください。'
            '</div></div>',
            unsafe_allow_html=True,
        )
    else:
        n_stop = sum(f["severity"] == "STOP" for f in a["flags"])
        n_warn = sum(f["severity"] == "WARN" for f in a["flags"])
        st.markdown(
            f'<div class="panel"><b>分析前に確認したいポイントがあります。</b>'
            f'<div class="small-note" style="margin-top:8px">'
            f'STOP {n_stop}件 / WARN {n_warn}件。まず重要度の高いものから確認してください。'
            f'</div></div>',
            unsafe_allow_html=True,
        )

with tab2:
    if not a["flags"]:
        st.success("自動監査で重大な警告は検出されませんでした。")
    else:
        for f in a["flags"]:
            st.markdown(flag_html(f), unsafe_allow_html=True)

    st.download_button(
        "監査結果JSONを保存",
        json.dumps(
            {"audit": a, "profile": p.to_dict("records")},
            ensure_ascii=False,
            indent=2,
        ),
        "survey_analyzer_preflight.json",
        "application/json",
    )

with tab3:
    st.dataframe(
        p,
        use_container_width=True,
        hide_index=True,
        column_config={
            "variable": "変数",
            "type": "推定型",
            "valid_n": st.column_config.NumberColumn("有効N"),
            "missing_pct": st.column_config.ProgressColumn(
                "欠損率",
                min_value=0,
                max_value=100,
                format="%.1f%%",
            ),
            "unique_n": st.column_config.NumberColumn("ユニーク数"),
        },
    )

st.markdown('<div class="section-label">03 · Ask the analyzer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="panel"><b>このデータで何を知りたいですか？</b>'
    '<div class="small-note" style="margin-top:6px">'
    '迷ったら「このデータを自分なら最初にどう見るべき？」でOKです。'
    '</div></div>',
    unsafe_allow_html=True,
)

q = st.text_area(
    "分析したいこと",
    label_visibility="collapsed",
    placeholder="例：このデータを自分なら最初にどう見るべき？",
    height=110,
)

run = st.button("分析計画を作る →", type="primary", use_container_width=True)

if run:
    if not q.strip():
        st.warning("質問を入力してください")
    elif not api_key:
        st.warning("OpenAI API Keyを設定してください")
    else:
        with st.spinner("データ構造を読み、分析計画を組み立てています…"):
            try:
                ans = ask_llm(
                    OpenAI(api_key=api_key),
                    model,
                    q,
                    compact_context(df, p, a),
                )
                st.markdown('<div class="section-label">04 · Analysis brief</div>', unsafe_allow_html=True)
                st.markdown('<div class="answer-shell">', unsafe_allow_html=True)
                st.markdown(ans)
                st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                msg = str(e)
                if "insufficient_quota" in msg or "429" in msg:
                    st.error("OpenAI APIの利用枠が不足しています。Platform側のBilling / Usageを確認してください。")
                else:
                    st.error(f"エラー: {e}")

st.markdown("---")
st.markdown(
    '<div class="small-note" style="text-align:center">'
    'Survey CSV Analyzer v5 · Toya Analysis Ontology v2<br>'
    'Find the pattern. Search for what breaks it. Reconstruct why the data look that way. Only then model it.'
    '</div>',
    unsafe_allow_html=True,
)
