import io, os, json, re
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, ttest_ind, f_oneway, pearsonr, spearmanr
from openai import OpenAI

st.set_page_config(page_title="Survey CSV Analyzer v5", page_icon="📊", layout="wide")

BASE_DIR = Path(__file__).parent
ONTOLOGY = json.loads((BASE_DIR/"analysis_ontology.json").read_text(encoding="utf-8"))
def get_secret(name, default=""):
    if os.getenv(name):
        return os.getenv(name)
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default

DEFAULT_MODEL = get_secret("OPENAI_MODEL", "gpt-5-mini")
APP_VERSION = "5.1-evidence-gate"
MAX_CATEGORY_UNIQUE = 20
LIKERT_HINTS = {"1","2","3","4","5","6","7"}

def read_tabular_robust(uploaded_file, sheet_name=None):
    raw = uploaded_file.getvalue()
    name = uploaded_file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        xls = pd.ExcelFile(io.BytesIO(raw))
        sheet = sheet_name or xls.sheet_names[0]
        df = pd.read_excel(io.BytesIO(raw), sheet_name=sheet)
        return df, f"excel:{sheet}", len(raw), xls.sheet_names
    for enc in ["utf-8-sig","utf-8","cp932","shift_jis"]:
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc), enc, len(raw), []
        except Exception:
            pass
    raise ValueError("CSV/Excelを読み込めませんでした。")

def infer_type(series):
    s = series.dropna()
    if len(s)==0: return "empty"
    un = s.nunique()
    ratio = un/max(len(s),1)
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() >= .95:
        vals = set(num.dropna().astype(float).tolist())
        if vals and all(float(v).is_integer() for v in vals) and un<=7 and all(str(int(v)) in LIKERT_HINTS for v in vals):
            return "likert"
        if un<=MAX_CATEGORY_UNIQUE and ratio<.15:
            return "categorical_numeric"
        return "numeric"
    avg_len = s.astype(str).str.len().mean()
    if un<=MAX_CATEGORY_UNIQUE and ratio<=.5: return "categorical"
    if avg_len>=25 or ratio>.7: return "text"
    return "categorical"

def profile_df(df):
    rows=[]
    for c in df.columns:
        s=df[c]
        rows.append({
            "variable":str(c),
            "type":infer_type(s),
            "valid_n":int(s.notna().sum()),
            "missing_n":int(s.isna().sum()),
            "missing_pct":round(float(s.isna().mean()*100),2),
            "unique_n":int(s.nunique(dropna=True)),
        })
    return pd.DataFrame(rows)

def safe_examples(series, n=5):
    if infer_type(series)=="text": return []
    s=series.dropna()
    return [str(x)[:80] for x in s.astype(str).value_counts().head(n).index.tolist()]

def dataset_context(df, profile):
    return {
        "rows":len(df),
        "columns":len(df.columns),
        "variables":[
            {
                "name":r["variable"],
                "type":r["type"],
                "missing_pct":float(r["missing_pct"]),
                "unique_n":int(r["unique_n"]),
                "examples":safe_examples(df[r["variable"]])
            }
            for _,r in profile.iterrows()
        ]
    }

def norm(s):
    return re.sub(r"[^a-z0-9_]+","_",str(s).lower())

def detect_ontology_features(df):
    names=[norm(c) for c in df.columns]
    detected=[]
    for rule in ONTOLOGY.get("feature_rules", ONTOLOGY.get("data_feature_rules", [])):
        hits=[]
        for hint in rule["detect_hints"]:
            h=norm(hint)
            for original,n in zip(df.columns,names):
                if h and h in n:
                    hits.append(str(original))
        if hits:
            detected.append({
                "feature":rule["feature"],
                "matched_variables":sorted(set(hits)),
                "questions":rule["questions"],
                "preferred_analyses":rule["preferred_analyses"],
                "cautions":rule["cautions"]
            })
    return detected

def _is_binary_numeric(s):
    x=pd.to_numeric(s,errors="coerce").dropna()
    u=set(x.unique().tolist())
    return len(u)==2 and u.issubset({0,1})

def cronbach_alpha(items):
    x=items.apply(pd.to_numeric,errors="coerce").dropna()
    if x.shape[0]<5 or x.shape[1]<2: return None
    vars_=x.var(axis=0,ddof=1); total=x.sum(axis=1).var(ddof=1)
    if total<=0: return None
    k=x.shape[1]
    return float(k/(k-1)*(1-vars_.sum()/total))

def infer_item_batteries(df):
    groups={}
    for c in map(str,df.columns):
        m=re.match(r"^([A-Za-z_]+?)[_ -]?(\\d+)$", c.strip())
        if not m: continue
        prefix=m.group(1).lower()
        groups.setdefault(prefix,[]).append(c)
    return {k:v for k,v in groups.items() if len(v)>=3}

def raw_evidence_snapshot(df, n_head=3, n_tail=2):
    return {
        "shape":{"rows":int(len(df)),"columns":int(len(df.columns))},
        "column_names":[str(c) for c in df.columns],
        "unnamed_columns":[str(c) for c in df.columns if norm(c).startswith("unnamed")],
        "duplicate_column_names":int(pd.Index(df.columns).duplicated().sum()),
        "head":df.head(n_head).astype(object).where(df.head(n_head).notna(), None).to_dict("records"),
        "tail":df.tail(n_tail).astype(object).where(df.tail(n_tail).notna(), None).to_dict("records")
    }

def id_unit_evidence(df):
    tokens={"id","number","no","record","respondent","respondent_id","participant_id","subject_id","caseid","case_id"}
    candidates=[]
    for c in df.columns:
        nc=norm(c).strip("_")
        token_hit=(nc in tokens) or nc.endswith("_id") or nc.startswith("id_") or nc.endswith("_number")
        if not token_hit: continue
        x=df[c].dropna()
        if len(x)==0: continue
        candidates.append({"variable":str(c),"valid_n":int(len(x)),"unique_n":int(x.nunique()),"duplicate_n":int(x.duplicated().sum()),"uniqueness_ratio":round(float(x.nunique()/len(x)),4)})
    return candidates

def missingness_signatures(df, min_col_missing=.03, max_col_missing=.97, max_cols=80):
    def _conditional_text_name(c):
        low=str(c).lower()
        return low.endswith("fa") or re.search(r"(_fa|_text|_specify|_other_text)$", low) is not None
    cols=[c for c in df.columns if min_col_missing <= df[c].isna().mean() <= max_col_missing and not _conditional_text_name(c)]
    if not cols: return []
    cols=sorted(cols,key=lambda c:df[c].isna().mean(),reverse=True)[:max_cols]
    mat=df[cols].isna().astype(np.uint8)
    sig=mat.astype(str).agg(''.join,axis=1)
    vc=sig.value_counts(); out=[]
    for key,n in vc.head(8).items():
        missing=[str(c) for c,b in zip(cols,key) if b=='1']; observed=[str(c) for c,b in zip(cols,key) if b=='0']
        out.append({"n":int(n),"pct":round(float(n/len(df)*100),2),"missing_columns":missing[:25],"observed_columns_sample":observed[:10]})
    return out

def paired_fa_evidence(df):
    cols={str(c):c for c in df.columns}; pairs=[]
    for name,c in cols.items():
        low=name.lower(); candidates=[]
        if low.endswith('fa'): candidates += [name[:-2], name[:-2].rstrip('_')]
        if re.search(r'(_fa|_text|_specify|_other_text)$',low): candidates += [re.sub(r'(_fa|_text|_specify|_other_text)$','',name,flags=re.I)]
        base=next((b for b in candidates if b in cols and b!=name),None)
        if not base: continue
        b=df[cols[base]]; f=df[c]
        f_non=f.notna() & f.astype(str).str.strip().ne('')
        vals=pd.to_numeric(b,errors='coerce')
        if vals.notna().mean()>=.8 and vals.nunique(dropna=True)<=10:
            rates=[]
            for v,g in pd.DataFrame({'b':vals,'f':f_non}).dropna(subset=['b']).groupby('b'):
                rates.append({"base_value":float(v),"n":int(len(g)),"fa_nonmissing_rate":round(float(g['f'].mean()),3)})
            max_rate=max((x['fa_nonmissing_rate'] for x in rates),default=0); min_rate=min((x['fa_nonmissing_rate'] for x in rates),default=0)
            pairs.append({"fa_variable":name,"base_variable":base,"fa_nonmissing_n":int(f_non.sum()),"rates":rates,"conditional_alignment":bool(max_rate>=.8 and min_rate<=.1)})
    return pairs

def candidate_effective_n(df, signatures):
    if not signatures: return None
    zero=[x for x in signatures if len(x['missing_columns'])==0]
    best=max(zero,key=lambda x:x['n']) if zero else max(signatures,key=lambda x:x['n'])
    return {"candidate_n":int(best['n']),"basis":"largest complete routing signature among non-conditional columns" if zero else "largest repeated routing signature","missing_columns_in_signature":len(best['missing_columns']),"status":"unresolved"}

def structural_audit(df, profile):
    flags=[]; insights=[]; hypotheses=[]
    raw=raw_evidence_snapshot(df)
    insights.append({"type":"raw_evidence","shape":raw["shape"],"unnamed_columns":raw["unnamed_columns"],"duplicate_column_names":raw["duplicate_column_names"]})

    if raw["unnamed_columns"] or raw["duplicate_column_names"]:
        hypotheses.append({"id":"H_SCHEMA_COLLAPSE","status":"unresolved","evidence":{"unnamed_columns":raw["unnamed_columns"],"duplicate_column_names":raw["duplicate_column_names"]}})
    else:
        hypotheses.append({"id":"H_SCHEMA_COLLAPSE","status":"rejected","evidence":"raw dataframe has named, non-duplicated columns"})

    for c in df.columns:
        vals=df[c].astype(str)
        n=vals.str.match(r"^#(NAME|VALUE|REF|DIV/0|N/A|NUM|NULL|SPILL|CALC)[!?]?",case=False,na=False).sum()
        if n:
            flags.append({"severity":"STOP","rule":"G_SCHEMA_VALIDATION","variable":str(c),"message":f"数式/セルエラーらしき値 {int(n)}件"})
    for c in df.columns:
        x=pd.to_numeric(df[c],errors="coerce")
        if x.notna().mean()>.7:
            neg=x[x.isin(list(range(-9,0)))].value_counts().to_dict()
            if neg:
                flags.append({"severity":"WARN","rule":"G_SCHEMA_VALIDATION","variable":str(c),"message":f"特殊欠損コード候補: {neg}"})
    for c in df.columns:
        if any(h in norm(c) for h in ["date","day","time"]):
            x=pd.to_numeric(df[c],errors="coerce").dropna()
            if len(x) and ((x>20000)&(x<70000)).mean()>.8:
                flags.append({"severity":"WARN","rule":"G_SCHEMA_VALIDATION","variable":str(c),"message":"Excel/Google Sheets日付シリアル値の可能性"})

    id_evidence=id_unit_evidence(df)
    repeats=[]
    repeat_tokens={"id","series","subject","respondent","respondent_id","participant_id","period","survperiod","wave","country","stage","part","market_day","day"}
    for c in df.columns:
        nc=norm(c).strip("_"); ss=df[c].dropna()
        token_hit=(nc in repeat_tokens) or nc.endswith("_id") or nc.startswith("id_")
        if token_hit and 1 < ss.nunique() < len(ss) and ss.duplicated().any():
            repeats.append({"variable":str(c),"unique_n":int(ss.nunique()),"rows":int(len(ss))})
    if repeats:
        flags.append({"severity":"WARN","rule":"G_ROWS_NOT_UNITS","message":"反復/クラスタ候補あり。行数と独立unit数を分けて確認", "candidates":repeats[:8]})
        hypotheses.append({"id":"H_ROWS_EQUAL_UNITS","status":"unresolved","evidence":{"repeat_candidates":repeats[:8],"id_candidates":id_evidence[:8]}})
    elif any(x["uniqueness_ratio"]==1.0 and x["valid_n"]==len(df) for x in id_evidence):
        good=next(x for x in id_evidence if x["uniqueness_ratio"]==1.0 and x["valid_n"]==len(df))
        hypotheses.append({"id":"H_ROWS_EQUAL_UNITS","status":"confirmed","evidence":good,"scope":"provisional; based on available identifiers"})
        insights.append({"type":"unit_validation","verdict":"one-row-per-unit provisionally supported","evidence":good})
    else:
        hypotheses.append({"id":"H_ROWS_EQUAL_UNITS","status":"unresolved","evidence":{"id_candidates":id_evidence[:8]}})

    outcome_hints=["buy","purchase","outcome","selected","choice","target","conversion","response","success","fail","adopt"]
    for c in df.columns:
        if _is_binary_numeric(df[c]):
            vc=pd.to_numeric(df[c],errors="coerce").value_counts(); rare=int(vc.min())
            if rare<20:
                likely_outcome=any(h in norm(c) for h in outcome_hints)
                severity="WARN" if likely_outcome or rare<=3 else "INFO"
                flags.append({"severity":severity,"rule":"G_RARE_OUTCOME","variable":str(c),"message":f"二値変数の少数側={rare}。{'目的変数なら' if likely_outcome else ''}高次元Logitは不安定になりやすい"})

    batteries=[]
    for prefix,cols in infer_item_batteries(df).items():
        a=cronbach_alpha(df[cols]); batteries.append({"prefix":prefix,"items":cols,"alpha":None if a is None else round(a,3)})
        if a is not None and a<.6:
            flags.append({"severity":"WARN","rule":"G_SCALE_BEFORE_MODEL","variable":prefix,"message":f"項目群 {cols} のCronbach α={a:.3f} と低め"})
    if batteries: insights.append({"type":"scale_candidates","groups":batteries})

    signatures=missingness_signatures(df)
    if signatures and len(signatures)>=2:
        top_share=sum(x['n'] for x in signatures[:3])/max(len(df),1)
        if top_share>=.6:
            flags.append({"severity":"WARN","rule":"G_ROUTING_MISSINGNESS","message":"繰り返し現れる欠損署名があります。設問ルーティング/適格条件の可能性を確認", "signatures":signatures[:5]})
            hypotheses.append({"id":"H_STRUCTURAL_MISSINGNESS","status":"unresolved","evidence":{"top_signatures":signatures[:5]}})
            insights.append({"type":"missingness_signatures","signatures":signatures[:8],"candidate_effective_n":candidate_effective_n(df,signatures)})

    fa_pairs=paired_fa_evidence(df)
    aligned=[x for x in fa_pairs if x['conditional_alignment']]
    if aligned:
        flags.append({"severity":"INFO","rule":"G_FA_CONDITIONAL","message":f"条件付き自由記述とみられるFA/テキスト列を {len(aligned)} 組検出。高欠損でも通常のデータ欠損とは限りません", "pairs":aligned[:10]})
        insights.append({"type":"conditional_free_text","pairs":aligned[:20]})

    group_cols=[]
    for c in df.columns:
        nc=norm(c); u=df[c].nunique(dropna=True)
        if any(h in nc for h in ["wave","period","group","trial","treatment","stage","country","screen","sc"] ) and 1<u<=50:
            group_cols.append(c)
    miss_patterns=[]; miss_cols=[c for c in df.columns if .05 <= df[c].isna().mean() <= .95]
    for g in group_cols[:8]:
        for c in miss_cols[:40]:
            rates=df.groupby(g,dropna=False)[c].apply(lambda x: x.isna().mean())
            if len(rates)>=2 and rates.max()-rates.min()>=.5:
                miss_patterns.append({"variable":str(c),"by":str(g),"min_missing":round(float(rates.min()),2),"max_missing":round(float(rates.max()),2)})
    if miss_patterns:
        flags.append({"severity":"WARN","rule":"G_STRUCTURAL_MISSINGNESS","message":"欠損率が群/時期/スクリーナ候補で大きく異なる変数あり", "patterns":miss_patterns[:15]})
        for h in hypotheses:
            if h['id']=='H_STRUCTURAL_MISSINGNESS': h['status']='confirmed'; h['group_evidence']=miss_patterns[:15]
        if not any(h['id']=='H_STRUCTURAL_MISSINGNESS' for h in hypotheses):
            hypotheses.append({"id":"H_STRUCTURAL_MISSINGNESS","status":"confirmed","evidence":miss_patterns[:15]})

    if len(df)>=10000:
        flags.append({"severity":"WARN","rule":"G_LARGE_N","message":f"大標本 N={len(df):,}。p値より効果量・一般化・検証を優先"})

    return {
        "status":"STOP" if any(f["severity"]=="STOP" for f in flags) else ("WARN" if any(f["severity"]=="WARN" for f in flags) else "GO"),
        "row_n":int(len(df)),"flags":flags,"insights":insights,"hypotheses":hypotheses,
        "raw_evidence":{"shape":raw['shape'],"unnamed_columns":raw['unnamed_columns'],"duplicate_column_names":raw['duplicate_column_names'],"column_names":raw['column_names']}
    }

FILTER_SCHEMA={
    "type":"object",
    "properties":{
        "variable":{"type":"string"},
        "operator":{"type":"string","enum":["eq","neq","in","not_in","gt","gte","lt","lte"]},
        "values":{"type":"array","items":{"type":["string","number"]}}
    },
    "required":["variable","operator","values"],
    "additionalProperties":False
}

PLAN_SCHEMA={
    "type":"object",
    "properties":{
        "goal":{"type":"string"},
        "ontology_lenses_used":{"type":"array","items":{"type":"string"}},
        "needs_clarification":{"type":"boolean"},
        "clarifying_question":{"type":"string"},
        "filters":{"type":"array","items":FILTER_SCHEMA},
        "analyses":{
            "type":"array",
            "items":{
                "type":"object",
                "properties":{
                    "kind":{"type":"string","enum":["quality","distribution","cross","group_compare","correlation","text_overview","scale_reliability","rare_event_check","repeated_structure","exception_scan","missingness_pattern","sequence_check"]},
                    "variables":{"type":"array","items":{"type":"string"}},
                    "reason":{"type":"string"},
                    "chart":{"type":"string","enum":["none","bar","histogram","boxplot","scatter"]}
                },
                "required":["kind","variables","reason","chart"],
                "additionalProperties":False
            }
        }
    },
    "required":["goal","ontology_lenses_used","needs_clarification","clarifying_question","filters","analyses"],
    "additionalProperties":False
}

def recent_context(messages,n=6):
    return [{"role":m["role"],"content":m["content"][:800]} for m in messages[-n:]]

def make_plan(client, model, question, df, profile, messages, detected, audit, use_ontology):
    instructions = """
You are a survey/tabular-data analysis planner.

Convert the user's question into the smallest executable plan.

Rules:
- Use only real column names and category values visible in dataset_context.
- Never invent variables.
- filters are row restrictions such as women only, age >= 30, purchasers only.
- If a filter value cannot be safely inferred from examples, ask one clarification question.
- Use prior_chat for follow-up references.
- Prefer descriptive analysis before more complex analysis.
- Never make causal claims from observational associations.
- Treat preflight_audit STOP flags as blockers: do not jump to inferential modeling around them.
- Treat audit hypotheses as hypotheses: confirmed may guide preprocessing; rejected must not be repeated as problems; unresolved must be stated as unresolved.
- When repeated missingness signatures or paired FA fields are detected, reconstruct routing/effective N before generic missing-data treatment.
- Prefer analyses that test data structure and contradictions before inferential comparisons.
- If a likely scale/item battery exists, validate it before using a composite.
- If a binary event is rare, explicitly inspect event counts before proposing a model.
- If repeated/nested structure is plausible, distinguish rows from analytical units.
- If time/stage variables exist, check sequence consistency and structural missingness.
- Actively search for observations that contradict the dominant pattern.

Kinds:
quality = data quality/missingness
distribution = one variable
cross = categorical x categorical
group_compare = categorical x numeric/Likert
correlation = numeric x numeric
text_overview = free-text column detection only

When analyst_ontology is supplied:
- treat it as a prior for how this analyst tends to reason;
- only use ontology lenses supported by dataset features or the user's question;
- do not force every lens;
- put actual used lenses into ontology_lenses_used.
"""
    payload={
        "current_question":question,
        "prior_chat":recent_context(messages),
        "dataset_context":dataset_context(df,profile),
        "detected_dataset_features":detected,
        "preflight_audit": audit
    }
    if use_ontology:
        payload["analyst_ontology"]={
            "principles":ONTOLOGY.get("core_principles", ONTOLOGY.get("principles", [])),
            "guardrails":ONTOLOGY.get("guardrails", []),
            "cross_cutting_reasoning":ONTOLOGY.get("cross_cutting_reasoning", []),
            "analysis_escalation":ONTOLOGY["analysis_escalation"],
            "interpretation_rules":ONTOLOGY["interpretation_rules"]
        }
    r=client.responses.create(
        model=model,
        instructions=instructions,
        input=json.dumps(payload,ensure_ascii=False),
        text={"format":{"type":"json_schema","name":"analysis_plan","schema":PLAN_SCHEMA,"strict":True}}
    )
    return json.loads(r.output_text)

def clean_plan(plan,df):
    cols=set(map(str,df.columns))
    p=dict(plan)
    p["filters"]=[f for f in p.get("filters",[]) if f.get("variable") in cols]
    cleaned=[]
    for a in p.get("analyses",[]):
        aa=dict(a)
        aa["variables"]=[v for v in aa.get("variables",[]) if v in cols]
        cleaned.append(aa)
    p["analyses"]=cleaned
    return p

def apply_filters(df, filters):
    out=df.copy()
    notes=[]
    for f in filters:
        col,op,vals=f["variable"],f["operator"],f["values"]
        if not vals: continue
        s=out[col]
        if op in ("gt","gte","lt","lte"):
            sn=pd.to_numeric(s,errors="coerce")
            v=float(vals[0])
            mask={"gt":sn>v,"gte":sn>=v,"lt":sn<v,"lte":sn<=v}[op]
        else:
            ss=s.astype(str); vv=[str(v) for v in vals]
            if op=="eq": mask=ss==vv[0]
            elif op=="neq": mask=ss!=vv[0]
            elif op=="in": mask=ss.isin(vv)
            else: mask=~ss.isin(vv)
        before=len(out)
        out=out.loc[mask].copy()
        notes.append(f"{col} {op} {vals}: {before} → {len(out)}")
    return out,notes

def cat_summary(s):
    x=s.fillna("(欠損)").astype(str)
    cnt=x.value_counts()
    return pd.DataFrame({"件数":cnt,"割合(%)":(cnt/len(x)*100).round(2)})

def num_summary(s):
    x=pd.to_numeric(s,errors="coerce")
    return {
        "n":int(x.notna().sum()),
        "mean":None if pd.isna(x.mean()) else float(x.mean()),
        "sd":None if pd.isna(x.std()) else float(x.std()),
        "median":None if pd.isna(x.median()) else float(x.median()),
        "min":None if pd.isna(x.min()) else float(x.min()),
        "max":None if pd.isna(x.max()) else float(x.max())
    }

def bar(df,c):
    t=cat_summary(df[c]).reset_index()
    t.columns=[c,"件数","割合(%)"]
    fig,ax=plt.subplots()
    ax.bar(t[c].astype(str),t["件数"])
    ax.set_title(c); ax.tick_params(axis="x",labelrotation=45)
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

def hist(df,c):
    x=pd.to_numeric(df[c],errors="coerce").dropna()
    fig,ax=plt.subplots()
    ax.hist(x,bins=min(20,max(5,int(np.sqrt(max(len(x),1))))))
    ax.set_xlabel(c); fig.tight_layout(); st.pyplot(fig); plt.close(fig)

def box(df,g,v):
    tmp=df[[g,v]].copy()
    tmp[v]=pd.to_numeric(tmp[v],errors="coerce")
    tmp=tmp.dropna()
    labels,vals=[],[]
    for k,d in tmp.groupby(g):
        if len(d):
            labels.append(str(k)); vals.append(d[v].values)
    if len(vals)>=2:
        fig,ax=plt.subplots()
        ax.boxplot(vals,tick_labels=labels)
        ax.set_xlabel(g); ax.set_ylabel(v); ax.tick_params(axis="x",labelrotation=45)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

def scatter(df,x,y):
    tmp=df[[x,y]].apply(pd.to_numeric,errors="coerce").dropna()
    fig,ax=plt.subplots()
    ax.scatter(tmp[x],tmp[y],alpha=.6)
    ax.set_xlabel(x); ax.set_ylabel(y)
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

def execute(a,df,profile):
    if len(df)==0: return {"kind":"error","message":"フィルタ後0件"}
    kind=a["kind"]; vars_=a["variables"]
    types=dict(zip(profile["variable"],profile["type"]))

    if kind=="quality":
        st.dataframe(profile.sort_values("missing_pct",ascending=False),use_container_width=True,hide_index=True)
        return {"kind":"quality","top_missing":profile.sort_values("missing_pct",ascending=False).head(10).to_dict("records")}

    if kind=="distribution":
        if not vars_: return {"kind":"error","message":"対象変数なし"}
        c=vars_[0]
        if types.get(c) in ("categorical","categorical_numeric","likert"):
            t=cat_summary(df[c]); st.dataframe(t,use_container_width=True); bar(df,c)
            return {"kind":"distribution","variable":c,"top_categories":[
                {"category":str(i),"count":int(r["件数"]),"pct":float(r["割合(%)"])}
                for i,r in t.head(12).iterrows()
            ]}
        s=num_summary(df[c]); st.json(s); hist(df,c)
        return {"kind":"distribution","variable":c,**s}

    if kind=="cross":
        if len(vars_)<2: return {"kind":"error","message":"2変数必要"}
        r,c=vars_[:2]
        tmp=df[[r,c]].dropna(); table=pd.crosstab(tmp[r],tmp[c])
        if table.shape[0]<2 or table.shape[1]<2: return {"kind":"error","message":"2×2以上にならない"}
        res=chi2_contingency(table); low=float((res.expected_freq<5).mean())
        st.dataframe(table,use_container_width=True)
        st.write(f"χ²={res.statistic:.3f}, df={res.dof}, p={res.pvalue:.4g}, N={table.values.sum()}")
        if low>0: st.warning(f"期待度数5未満のセル: {low*100:.1f}%")
        return {"kind":"cross","variables":[r,c],"chi2":float(res.statistic),"df":int(res.dof),
                "p_value":float(res.pvalue),"n":int(table.values.sum()),"low_expected_ratio":low}

    if kind=="group_compare":
        if len(vars_)<2: return {"kind":"error","message":"2変数必要"}
        g,v=vars_[:2]
        tmp=df[[g,v]].copy(); tmp[v]=pd.to_numeric(tmp[v],errors="coerce"); tmp=tmp.dropna()
        groups=[(str(k),d[v].values) for k,d in tmp.groupby(g) if len(d)>=2]
        if len(groups)<2: return {"kind":"error","message":"十分な群がない"}
        desc=[{"group":n,"n":int(len(x)),"mean":float(np.mean(x)),"sd":float(np.std(x,ddof=1))} for n,x in groups]
        st.dataframe(pd.DataFrame(desc),use_container_width=True,hide_index=True)
        if len(groups)==2:
            stat,p=ttest_ind(groups[0][1],groups[1][1],equal_var=False,nan_policy="omit"); test="Welch t-test"
        else:
            stat,p=f_oneway(*[x for _,x in groups]); test="one-way ANOVA"
        st.write(f"{test}: statistic={stat:.3f}, p={p:.4g}")
        box(df,g,v)
        return {"kind":"group_compare","variables":[g,v],"test":test,"statistic":float(stat),"p_value":float(p),"groups":desc}

    if kind=="correlation":
        if len(vars_)<2: return {"kind":"error","message":"2変数必要"}
        x,y=vars_[:2]
        tmp=df[[x,y]].apply(pd.to_numeric,errors="coerce").dropna()
        if len(tmp)<3: return {"kind":"error","message":"データ不足"}
        r,p=pearsonr(tmp[x],tmp[y]); rho,sp=spearmanr(tmp[x],tmp[y])
        st.write(f"Pearson r={r:.3f}, p={p:.4g}")
        st.write(f"Spearman ρ={rho:.3f}, p={sp:.4g}")
        scatter(df,x,y)
        return {"kind":"correlation","variables":[x,y],"n":int(len(tmp)),"pearson_r":float(r),"pearson_p":float(p),
                "spearman_rho":float(rho),"spearman_p":float(sp)}

    if kind=="scale_reliability":
        cols=vars_ if len(vars_)>=2 else []
        if len(cols)<2: return {"kind":"error","message":"2項目以上必要"}
        a=cronbach_alpha(df[cols])
        if a is None: return {"kind":"error","message":"信頼性計算に十分な完全回答がない"}
        st.write(f"Cronbach α={a:.3f} (items={len(cols)})")
        return {"kind":"scale_reliability","variables":cols,"alpha":float(a),"complete_n":int(df[cols].apply(pd.to_numeric,errors="coerce").dropna().shape[0])}

    if kind=="rare_event_check":
        if not vars_: return {"kind":"error","message":"対象変数なし"}
        c=vars_[0]; x=pd.to_numeric(df[c],errors="coerce").dropna()
        vc=x.value_counts().sort_index(); st.dataframe(vc.rename("count"))
        rare=int(vc.min()) if len(vc)==2 else None
        return {"kind":"rare_event_check","variable":c,"counts":{str(k):int(v) for k,v in vc.items()},"rare_count":rare,"warning":bool(rare is not None and rare<20)}

    if kind=="repeated_structure":
        candidates=[]
        for c in (vars_ or list(df.columns)):
            s=df[c].dropna(); u=s.nunique()
            if 1<u<len(s) and s.duplicated().any(): candidates.append({"variable":str(c),"unique_n":int(u),"rows":int(len(s))})
        st.dataframe(pd.DataFrame(candidates),use_container_width=True,hide_index=True)
        return {"kind":"repeated_structure","candidates":candidates[:20],"row_n":int(len(df))}

    if kind=="missingness_pattern":
        if len(vars_)<2: return {"kind":"error","message":"欠損を見る変数とgroup/time変数が必要"}
        c,g=vars_[:2]
        rates=df.groupby(g,dropna=False)[c].apply(lambda x:x.isna().mean()).reset_index(name="missing_rate")
        st.dataframe(rates,use_container_width=True,hide_index=True)
        return {"kind":"missingness_pattern","variable":c,"by":g,"rates":rates.to_dict("records")}

    if kind=="exception_scan":
        if len(vars_)<2: return {"kind":"error","message":"2変数以上必要"}
        x,y=vars_[:2]
        tmp=df[[x,y]].copy(); tmp[x]=pd.to_numeric(tmp[x],errors="coerce")
        if tmp[x].notna().sum()>=5:
            q=tmp[x].quantile(.75); ex=tmp[(tmp[x]>=q)].copy()
            if ex[y].nunique()<=20:
                mode=tmp[y].mode().iloc[0] if not tmp[y].mode().empty else None
                ex=ex[ex[y]!=mode]
            st.dataframe(ex.head(20),use_container_width=True,hide_index=True)
            return {"kind":"exception_scan","variables":[x,y],"high_x_threshold":float(q),"examples":ex.head(20).astype(str).to_dict("records")}
        return {"kind":"error","message":"例外抽出に数値変数が必要"}

    if kind=="sequence_check":
        if len(vars_)<3: return {"kind":"error","message":"unit, stage/order, date の3変数が必要"}
        uid,stage,date=vars_[:3]
        tmp=df[[uid,stage,date]].copy(); tmp[stage]=pd.to_numeric(tmp[stage],errors="coerce"); tmp[date]=pd.to_numeric(tmp[date],errors="coerce")
        bad=[]
        for k,d in tmp.dropna().groupby(uid):
            if len(d)<2: continue
            a=d.sort_values(stage)[date].values
            if not (np.all(np.diff(a)>=0) or np.all(np.diff(a)<=0)): bad.append(str(k))
        st.write(f"順序矛盾候補 unit: {len(bad)}")
        return {"kind":"sequence_check","unit":uid,"stage":stage,"date":date,"contradictory_units":bad[:50],"n_contradictory":len(bad)}

    if kind=="text_overview":
        cols=profile.loc[profile["type"]=="text","variable"].tolist()
        st.write("自由記述列:",", ".join(cols) if cols else "なし")
        return {"kind":"text_overview","text_columns":cols}

    return {"kind":"error","message":"未対応"}

def interpret(client,model,question,goal,results,filters,lenses,detected,audit):
    instructions = """
Explain Python-computed survey analysis results in Japanese.

Rules:
- Do not invent numbers.
- Never claim causality unless supported.
- Separate statistical significance from substantive importance.
- Mention low expected counts or small subgroup instability.
- If p >= .05, say evidence was insufficient to confirm the association/difference.
- If ontology lenses were used, reflect them naturally without saying "the user likes X".
- Treat STOP/WARN audit flags as first-class findings.
- Respect evidence-gate verdicts: never repeat rejected structural hypotheses as findings; label unresolved ones explicitly.
- Distinguish raw N, provisional analytical-unit N, and candidate effective N when routing evidence exists.
- Explicitly distinguish row count from analytical-unit count when relevant.
- For rare outcomes, mention event counts before any model implication.
- Treat contradictions/exceptions as evidence that may reveal hidden mechanisms or protocol rules.
- End with the most decision-relevant takeaway and at most two next analyses.
"""
    payload={
        "question":question,
        "goal":goal,
        "filters":filters,
        "ontology_lenses_used":lenses,
        "detected_dataset_features":detected,
        "computed_results":results,
        "preflight_audit":audit,
        "interpretation_rules":ONTOLOGY["interpretation_rules"]
    }
    r=client.responses.create(model=model,instructions=instructions,input=json.dumps(payload,ensure_ascii=False))
    return r.output_text

st.title("📊 Survey CSV Analyzer v5")
st.caption("Preflight Audit + Toya Analysis Ontology v3。仮説を生データで検証し、反例を探し、データ生成過程を推理してから分析します。")

api_key=get_secret("OPENAI_API_KEY","")
model=DEFAULT_MODEL
use_ontology=True

with st.sidebar:
    st.markdown("### POC設定")
    if api_key:
        st.success("API接続済み")
    else:
        st.error("AI分析機能は現在利用できません。管理者がサーバー側のAPI設定を確認してください。")
    with st.expander("詳細設定"):
        st.caption(f"Model: {model}")
        use_ontology=st.toggle("Toya Analysis Ontologyを使う",value=True)
    st.caption("プライバシー: CSV/Excel全文はLLMへ送信せず、列名・型・欠損率・少数のカテゴリ例・Python監査結果を使って分析計画を作ります。")
    st.caption(f"Version: {APP_VERSION}")

uploaded=st.file_uploader("CSV / Excelをアップロード",type=["csv","xlsx","xls"])
if uploaded is None:
    st.info("① CSV / Excelをアップロード → ② Preflight Auditを確認 → ③ 下のチャットで知りたいことを日本語で入力してください。")
    st.markdown("**質問例**: `このデータを自分なら最初にどう見るべき？` / `購入者と非購入者の違いを見て` / `この尺度はそのまま使っていい？`")
    st.stop()

sheet_names=[]
if uploaded.name.lower().endswith((".xlsx",".xls")):
    xls_preview=pd.ExcelFile(io.BytesIO(uploaded.getvalue()))
    sheet_names=xls_preview.sheet_names
    selected_sheet=st.selectbox("分析するシート",sheet_names)
else:
    selected_sheet=None

df,encoding,raw_bytes,sheet_names=read_tabular_robust(uploaded,selected_sheet)
profile=profile_df(df)
detected=detect_ontology_features(df)
audit=structural_audit(df,profile)

key=f"{uploaded.name}:{len(df)}:{len(df.columns)}:{raw_bytes}"
if st.session_state.get("dataset_key")!=key:
    st.session_state.dataset_key=key
    st.session_state.messages=[{"role":"assistant","content":"CSVを読み込みました。**このデータで何を知りたいですか？**"}]

c1,c2,c3=st.columns(3)
c1.metric("回答数",f"{len(df):,}")
c2.metric("変数数",f"{len(df.columns):,}")
c3.metric("ファイルサイズ",f"{raw_bytes/(1024*1024):.2f} MB")

with st.expander("🛡️ Preflight Audit（分析前チェック）", expanded=True):
    if audit["status"]=="STOP": st.error("STOP: 分析前に確認すべき構造問題があります。")
    elif audit["status"]=="WARN": st.warning("WARN: 分析設計に反映すべき注意点があります。")
    else: st.success("GO: 自動監査で重大な問題は検出されませんでした。")
    for f in audit["flags"]:
        st.markdown(f"- **{f['severity']} / {f['rule']}**: {f.get('message','')}")
    if audit.get("hypotheses"):
        st.markdown("**Evidence Gate verdicts**")
        for h in audit["hypotheses"]:
            icon={"confirmed":"✅","rejected":"❌","unresolved":"❓"}.get(h.get("status"),"•")
            st.markdown(f"- {icon} `{h.get('id')}` → **{h.get('status')}**")
    if audit["insights"]: st.json(audit["insights"])
    audit_payload={"file":uploaded.name,"sheet":selected_sheet,"audit":audit,"profile":profile.to_dict("records"),"detected_features":detected}
    st.download_button("監査結果JSONを保存",data=json.dumps(audit_payload,ensure_ascii=False,indent=2),file_name="survey_analyzer_preflight.json",mime="application/json")

with st.expander("検出されたデータ特徴 / オントロジー候補",expanded=True):
    if detected:
        for d in detected:
            st.markdown(f"**{d['feature']}** — {', '.join(d['matched_variables'])}")
            st.caption("見る観点: " + " / ".join(d["questions"]))
    else:
        st.write("特徴ルールに一致する変数はまだありません。")

with st.expander("オントロジー本体"):
    st.json(ONTOLOGY)

with st.expander("変数プロファイル"):
    st.dataframe(profile,use_container_width=True,hide_index=True)

st.divider()
st.subheader("分析チャット")

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

q=st.chat_input("例：女性だけで、年代ごとに満足度が違うか見たい")

if q:
    st.session_state.messages.append({"role":"user","content":q})
    with st.chat_message("user"): st.markdown(q)

    with st.chat_message("assistant"):
        if not api_key:
            ans="APIキーを設定してください。"; st.warning(ans)
        else:
            client=OpenAI(api_key=api_key)
            try:
                with st.spinner("分析計画を作成中…"):
                    plan=make_plan(client,model,q,df,profile,st.session_state.messages[:-1],detected,audit,use_ontology)
                    plan=clean_plan(plan,df)

                if plan["needs_clarification"]:
                    ans=plan["clarifying_question"]; st.markdown(ans)
                else:
                    st.markdown(f"**分析目的:** {plan['goal']}")
                    if plan["ontology_lenses_used"]:
                        st.caption("使った分析レンズ: " + " / ".join(plan["ontology_lenses_used"]))

                    filtered,filter_notes=apply_filters(df,plan["filters"])
                    if filter_notes:
                        st.markdown("**フィルタ**")
                        for n in filter_notes: st.markdown(f"- {n}")

                    with st.expander("分析計画JSON"):
                        st.json(plan)

                    results=[]
                    for i,a in enumerate(plan["analyses"],1):
                        st.markdown(f"### {i}. {a['reason']}")
                        results.append(execute(a,filtered,profile))

                    with st.spinner("結果を解釈中…"):
                        ans=interpret(client,model,q,plan["goal"],results,plan["filters"],plan["ontology_lenses_used"],detected,audit)
                    st.markdown("### AIによる解釈")
                    st.markdown(ans)
            except Exception as e:
                ans=f"エラー: {e}"; st.error(ans)

    st.session_state.messages.append({"role":"assistant","content":ans})
