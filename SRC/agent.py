import json, os, re, time, inspect
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from groq import Groq, RateLimitError

from rag import retrieve
from tools import TOOLS, TOOL_DESCRIPTIONS, profile

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODELS = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"]   # main model, then backup
OUT = Path("outputs")
CORE_TOOLS = ["missing_values", "distributions", "outliers", "correlations"]


# ---------- talking to the LLM ----------

def llm(system, user, max_tokens=4000):
    """Call Groq; retry on rate limits, fall back to the backup model if needed."""
    for model in MODELS:
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                    temperature=0.2,
                    max_completion_tokens=max_tokens,
                )
                text = resp.choices[0].message.content or ""
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
                if text:
                    return text
            except RateLimitError:
                wait = 10 * (attempt + 1)
                print(f"    rate limited, waiting {wait}s...")
                time.sleep(wait)
        print(f"    {model} unavailable, trying backup model...")
    raise RuntimeError("All models failed. Check your Groq key and rate limits.")


def parse_json(text):
    """Pull the JSON object out of an LLM reply, even if it added extra text."""
    text = re.sub(r"```(?:json)?", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in the reply")
    return json.loads(text[start:end + 1])


def for_llm(obj):
    """Remove plot file paths before sending results to the LLM (saves tokens)."""
    if isinstance(obj, dict):
        return {k: for_llm(v) for k, v in obj.items() if k != "plot"}
    return obj


# ---------- retrieval ----------

def gather_guidance(queries, k=2):
    """Retrieve for several queries and remove duplicate sections."""
    seen, chunks = set(), []
    for q in queries:
        for chunk in retrieve(q, k=k):
            title = chunk.splitlines()[0]
            if title not in seen:
                seen.add(title)
                chunks.append(chunk)
    return chunks


# ---------- stage 1: plan ----------

def make_plan(prof):
    queries = ["order of steps in an EDA report",
               "checking missing values and data quality",
               "choosing an outlier detection method",
               "choosing a correlation method"]
    if prof["categorical_columns"]:
        queries.append("analysing categorical columns")
    if prof["duplicate_rows"]:
        queries.append("duplicate rows")
    if prof["id_like_columns"]:
        queries.append("ID columns in the dataset")
    guidance = gather_guidance(queries)

    tool_list = "\n".join(f"- {name}: {desc}" for name, desc in TOOL_DESCRIPTIONS.items())
    system = (
        "You are a data analyst planning an exploratory data analysis. "
        "Choose which tools to run, in what order and with what arguments, "
        "following the best practices provided.\n\n"
        f"Available tools:\n{tool_list}\n\n"
        "Reply with ONLY a JSON object and no other text, in this format:\n"
        '{"steps": [{"tool": "<tool name>", "args": {}, '
        '"reason": "<one sentence naming the best practice that justifies this step>"}]}'
    )
    user = (f"DATASET PROFILE:\n{json.dumps(prof, indent=2)}\n\n"
            "BEST PRACTICES:\n" + "\n\n---\n\n".join(guidance))

    try:
        raw_steps = parse_json(llm(system, user, max_tokens=2000)).get("steps", [])
    except Exception as e:
        print(f"    could not read the plan ({e}); using the default plan")
        raw_steps = []
    return validate_plan(raw_steps), guidance


def validate_plan(raw_steps):
    """Keep only real tools and real arguments; add any core step the LLM forgot."""
    steps, used = [], set()
    for s in raw_steps:
        name = s.get("tool")
        if name not in TOOLS or name in used:
            continue
        allowed = set(inspect.signature(TOOLS[name]).parameters) - {"df"}
        args = {k: v for k, v in (s.get("args") or {}).items() if k in allowed}
        steps.append({"tool": name, "args": args, "reason": s.get("reason", "")})
        used.add(name)
    for name in CORE_TOOLS:
        if name not in used:
            steps.append({"tool": name, "args": {},
                          "reason": "Added automatically: core EDA step missing from the plan."})
    return steps


# ---------- stage 2: execute ----------

def execute(df, steps):
    results = {}
    for s in steps:
        fn = TOOLS[s["tool"]]
        print(f"  running {s['tool']} {s['args'] or ''}")
        try:
            results[s["tool"]] = fn(df, **s["args"])
        except Exception as e:
            print(f"    failed ({e}); retrying with default settings")
            results[s["tool"]] = fn(df)
    return results


# ---------- stage 3: report ----------

def finding_queries(prof, results):
    """Turn what was actually found into retrieval questions."""
    q = ["structure of an EDA report and recommendations"]
    miss = results.get("missing_values", {}).get("columns_with_missing", {})
    if miss:
        q.append(f"column with {max(miss.values()):.0f}% missing values")
    if any(abs(v.get("skew") or 0) > 1 for v in results.get("distributions", {}).values()):
        q.append("highly skewed numeric distribution")
    if any(v["count"] > 0 for v in results.get("outliers", {}).values()):
        q.append("outliers: errors versus genuine extremes")
    if results.get("correlations", {}).get("strong_pairs"):
        q.append("strongly correlated features and multicollinearity")
    cats = results.get("categorical_summary", {})
    if any(v["high_cardinality"] for v in cats.values()):
        q.append("high-cardinality categorical column")
    if any(v["possible_inconsistent_spelling"] for v in cats.values()):
        q.append("inconsistent category spelling")
    if prof["duplicate_rows"]:
        q.append("duplicate rows")
    return q


def write_report(name, prof, steps, results, guidance):
    system = (
        "You are a senior data analyst writing an exploratory data analysis report in Markdown.\n"
        "Rules:\n"
        "- Use ONLY numbers that appear in the RESULTS JSON. Never calculate or invent statistics.\n"
        "- Interpret findings using the BEST PRACTICES and name the guideline you apply "
        "(for example: 'per the Skewed distributions guideline').\n"
        "- Flag problems; never claim that any data was cleaned, removed or changed.\n"
        "- Use these sections in order: ## Overview, ## Data quality, ## Distributions, "
        "## Outliers, ## Categorical variables, ## Relationships, ## Recommendations.\n"
        "- Recommendations: a numbered list, each naming the column and the reason.\n"
        "- Do not write a title; start directly with ## Overview."
    )
    user = (f"DATASET: {name}\n\n"
            f"PROFILE:\n{json.dumps(prof, indent=2)}\n\n"
            f"PLAN EXECUTED:\n{json.dumps(steps, indent=2)}\n\n"
            f"RESULTS:\n{json.dumps(for_llm(results), indent=2)}\n\n"
            "BEST PRACTICES:\n" + "\n\n---\n\n".join(guidance))
    return llm(system, user, max_tokens=5000)


# ---------- the full run ----------

def run(csv_path):
    df = pd.read_csv(csv_path)
    name = Path(csv_path).name
    print(f"Loaded {name}: {df.shape[0]} rows, {df.shape[1]} columns")

    plots = OUT / "plots"
    if plots.exists():                       # clear old plots from earlier runs
        for p in plots.glob("*.png"):
            p.unlink()

    prof = profile(df)

    print("\n[1/3] Planning...")
    steps, plan_guidance = make_plan(prof)
    for i, s in enumerate(steps, 1):
        print(f"  {i}. {s['tool']} {s['args'] or ''} - {s['reason']}")

    print("\n[2/3] Running analysis...")
    results = execute(df, steps)

    print("\n[3/3] Writing report...")
    guidance = gather_guidance(finding_queries(prof, results))
    report = f"# EDA report: {name}\n\n" + write_report(name, prof, steps, results, guidance)

    OUT.mkdir(exist_ok=True)
    (OUT / "report.md").write_text(report, encoding="utf-8")
    log = {"dataset": name, "profile": prof, "plan": steps, "results": results,
           "guidance_used": {"planning": [g.splitlines()[0] for g in plan_guidance],
                             "reporting": [g.splitlines()[0] for g in guidance]}}
    (OUT / "run_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")

    print("\nDone. Report: outputs/report.md | Log: outputs/run_log.json")
    return {**log, "report": report}


if __name__ == "__main__":
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else "data/synthetic_test.csv")