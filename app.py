import io, os, json, re
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import chi2_contingency, ttest_ind, pearsonr, spearmanr
from openai import OpenAI

st.set_page_config(page_title="Survey CSV Analyzer v5", page_icon="📊", layout="wide")
BASE = Path(__file__).parent
ONTOLOGY = json.loads((BASE / "analysis_ontology.json").read_text(encoding="utf-8"))


def secret(name, default=""):
    if os.getenv(name): return os.getenv(name)
    try: return st.secrets.get(name, default)
    except Exception: return default


def infer_type(s):
    x=s.dropna()
    if len(x)==0: return "empty"
    num=pd.to_numeric(x, errors="coerce")
    if num.notna().mean()>=.95:
        if x.nunique()<=12: return "categorical_numeric"
        return "numeric"
    if x.nunique()<=20: return "categorical"
    return "text" if x.astype(str).str.len().mean()>=25 else "categorical"


def profile(df):
    return pd.DataFrame([{"variable":str(c),"type":infer_type(df[c]),"valid_n":int(df[c].notna().sum()),"missing_pct":round(float(df[c].isna().mean()*100),2),"unique_n":int(df[c].nunique(dropna=True))} for c in df.columns])


def read_file(upload, sheet=None):
    raw=upload.getvalue(); name=upload.name.lower()
    if name.endswith((".xlsx",".xls")):
        xls=pd.ExcelFile(io.BytesIO(raw)); sh=sheet or xls.sheet_names[0]
        return pd.read_excel(io.BytesIO(raw), sheet_name=sh), xls.sheet_names
    for enc in ["utf-8-sig","utf-8","cp932","shift_jis"]:
        try: return pd.read_csv(io.BytesIO(raw), encoding=enc), []
        except Exception: pass
    raise ValueError("CSV/Excelを読み込めませんでした")


def cronbach_alpha(frame):
    x=frame.apply(pd.to_numeric,errors="coerce").dropna()
    if x.shape[0]<5 or x.shape[1]<2: return None
    k=x.shape[1]; total=x.sum(axis=1).var(ddof=1)
    if total<=0:return None
    return float(k/(k-1)*(1-x.var(axis=0,ddof=1).sum()/total))


def batteries(df):
    g={}
    for c in map(str,df.columns):
        m=re.match(r"^([A-Za-z_]+?)[_ -]?(\d+)$",c.strip())
        if m:g.setdefault(m.group(1).lower(),[]).append(c)
    return {k:v for k,v in g.items() if len(v)>=3}


def audit(df):
    flags=[]; insights=[]
    for c in df.columns:
        vals=df[c].astype(str)
        n=vals.str.match(r"^#(NAME|VALUE|REF|DIV/0|N/A|NUM|NULL|SPILL|CALC)",case=False,na=False).sum()
        if n: flags.append({"severity":"STOP","rule":"G_SCHEMA_VALIDATION","message":f"{c}: セルエラー候補 {int(n)}件"})
        num=pd.to_numeric(df[c],errors="coerce")
        if num.notna().mean()>.7:
            neg=num[num.isin(range(-9,0))].value_counts().to_dict()
            if neg: flags.append({"severity":"WARN","rule":"G_SCHEMA_VALIDATION","message":f"{c}: 特殊欠損コード候補 {neg}"})
        nc=str(c).lower()
        if any(h in nc for h in ["date","time","day"]):
            x=num.dropna()
            if len(x) and ((x>20000)&(x<70000)).mean()>.8:
                flags.append({"severity":"WARN","rule":"G_SCHEMA_VALIDATION","message":f"{c}: Excel/Sheets日付シリアルの可能性"})
    for c in df.columns:
        x=pd.to_numeric(df[c],errors="coerce").dropna(); u=set(x.unique().tolist())
        if len(u)==2 and u.issubset({0,1}):
            rare=int(x.value_counts().min())
            if rare<20: flags.append({"severity":"WARN","rule":"G_STOP_RULE","message":f"{c}: 二値変数の少数側={rare}。多変量Logitは不安定になりやすい"})
    repeat=[]
    for c in df.columns:
        nc=str(c).lower(); s=df[c].dropna()
        if any(h in nc for h in ["id","series","respondent","participant","country","wave","period","stage"]) and 1<s.nunique()<len(s) and s.duplicated().any():
            repeat.append({"variable":str(c),"unique_n":int(s.nunique()),"rows":int(len(s))})
    if repeat: flags.append({"severity":"WARN","rule":"G_ROWS_NOT_UNITS","message":"反復/クラスタ候補あり。行数と独立unit数を分けて確認","details":repeat[:8]})
    bs=[]
    for p,cols in batteries(df).items():
        a=cronbach_alpha(df[cols]); bs.append({"prefix":p,"items":cols,"alpha":None if a is None else round(a,3)})
        if a is not None and a<.6: flags.append({"severity":"WARN","rule":"G_SCALE_BEFORE_MODEL","message":f"{p}: Cronbach α={a:.3f} と低め"})
    if bs: insights.append({"scale_candidates":bs})
    if len(df)>=10000: flags.append({"severity":"WARN","rule":"G_LARGE_N","message":f"大標本 N={len(df):,}。p値より効果量・検証を優先"})
    status="STOP" if any(f["severity"]=="STOP" for f in flags) else ("WARN" if flags else "GO")
    return {"status":status,"flags":flags,"insights":insights,"row_n":len(df)}


def compact_context(df,p,a):
    vars=[]
    for _,r in p.iterrows():
        c=r["variable"]; s=df[c].dropna()
        examples=[] if r["type"]=="text" else [str(x)[:60] for x in s.astype(str).value_counts().head(4).index]
        vars.append({"name":c,"type":r["type"],"missing_pct":r["missing_pct"],"unique_n":int(r["unique_n"]),"examples":examples})
    return {"rows":len(df),"columns":len(df.columns),"variables":vars,"audit":a,"ontology":ONTOLOGY["core_rules"]}


def ask_llm(client,model,question,context):
    instructions="""You are an analysis-planning assistant. Return concise Japanese. Use only metadata and audit information provided; the full dataset is not sent. First identify what can safely be concluded, what must be checked, and a small analysis plan. Follow Toya Analysis Ontology: find pattern, search exceptions, reconstruct data-generating process, only then model. Do not claim causality. If rare outcomes, repeated units, weak scales, structural missingness, or schema issues matter, make them first-class findings."""
    payload={"question":question,"dataset_context":context}
    r=client.responses.create(model=model,instructions=instructions,input=json.dumps(payload,ensure_ascii=False))
    return r.output_text


st.title("📊 Survey CSV Analyzer v5")
st.caption("Preflight Audit + Toya Analysis Ontology v2")

with st.sidebar:
    api_key=secret("OPENAI_API_KEY","")
    if api_key: st.success("APIキー設定済み")
    else: api_key=st.text_input("OpenAI API Key",type="password")
    model=st.text_input("Model",value=secret("OPENAI_MODEL","gpt-5-mini"))
    st.caption("CSV/Excel全文はLLMへ送信しません。列名・型・欠損率・少数のカテゴリ例・監査結果のみ送信します。")

uploaded=st.file_uploader("CSV / Excelをアップロード",type=["csv","xlsx","xls"])
if not uploaded:
    st.info("CSV / Excelをアップロードしてください。")
    st.stop()

sheet=None
if uploaded.name.lower().endswith((".xlsx",".xls")):
    xls=pd.ExcelFile(io.BytesIO(uploaded.getvalue()))
    sheet=st.selectbox("分析するシート",xls.sheet_names)

df,_=read_file(uploaded,sheet)
p=profile(df); a=audit(df)

c1,c2,c3=st.columns(3)
c1.metric("行数",f"{len(df):,}"); c2.metric("変数数",len(df.columns)); c3.metric("Audit",a["status"])

with st.expander("🛡️ Preflight Audit",expanded=True):
    if a["status"]=="STOP": st.error("STOP: 分析前に確認すべき構造問題があります")
    elif a["status"]=="WARN": st.warning("WARN: 分析設計に反映すべき注意点があります")
    else: st.success("GO: 重大な問題は自動検出されませんでした")
    for f in a["flags"]: st.markdown(f"- **{f['severity']} / {f['rule']}**: {f['message']}")
    if a["insights"]: st.json(a["insights"])
    st.download_button("監査結果JSONを保存",json.dumps({"audit":a,"profile":p.to_dict('records')},ensure_ascii=False,indent=2),"survey_analyzer_preflight.json","application/json")

with st.expander("変数プロファイル"):
    st.dataframe(p,use_container_width=True,hide_index=True)

st.subheader("分析相談")
q=st.text_area("このデータで何を知りたいですか？",placeholder="例：このデータを自分なら最初にどう見るべき？")
if st.button("分析計画を作る",type="primary"):
    if not q.strip(): st.warning("質問を入力してください")
    elif not api_key: st.warning("OpenAI API Keyを設定してください")
    else:
        with st.spinner("分析計画を作成中…"):
            try:
                ans=ask_llm(OpenAI(api_key=api_key),model,q,compact_context(df,p,a))
                st.markdown(ans)
            except Exception as e: st.error(f"エラー: {e}")

st.divider()
st.caption("v5 POC build — Find the pattern. Search for what breaks it. Reconstruct why the data look that way. Only then model it.")
