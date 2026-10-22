# Automated Exploratory Data Analysis Agent with Retrieval-Augmented Generation

Niraj Palve

## Overview

Exploratory data analysis (EDA) is the first step of almost every data
science project, yet it is largely manual and repetitive, and its quality
depends heavily on the analyst's experience. The same checks are rewritten
for every new dataset, and whether the right method is chosen (for example,
IQR versus z-score for outliers, or Spearman versus Pearson for correlation)
is left to individual judgement. Large language models (LLMs) can automate
the decision-making, but they introduce two problems of their own: they are
unreliable at arithmetic and can produce plausible but invented statistics,
and without guidance they follow no consistent methodology. This project
addresses all three issues by:

1. **Deterministic statistical tools.** Every number and plot is produced by
   ordinary Python code (pandas, SciPy, Matplotlib, Seaborn), never by the
   LLM. The LLM only ever sees compact JSON summaries of tool outputs, not
   the raw data, and is instructed to use only those values.
2. **LLM-driven planning and interpretation.** Following a plan-then-execute
   design (in the spirit of Plan-and-Solve prompting, Wang et al., 2023), an
   LLM inspects a profile of the dataset, decides which analyses to run, with
   which settings and why, and afterwards interprets the results in a
   structured written report with prioritised recommendations.
3. **Retrieval-augmented generation (RAG).** A curated knowledge base of EDA
   best practices (drawing on the exploratory tradition of Tukey, 1977) is
   embedded and indexed (Lewis et al., 2020). Relevant guidelines are
   retrieved **twice** per run: once to shape *what to analyse* during
   planning, and again, based on what was actually found, to shape *how to
   interpret* the findings. The report names the guideline behind each
   conclusion, so its reasoning can be traced.

The end goal is a single pipeline that goes from any raw CSV file to a
methodologically grounded, fully auditable EDA report, combining the
flexibility of an LLM agent with the reliability of deterministic statistics
rather than trusting either one alone.

## Project structure

```
eda-agent/
├── knowledge/
│   └── eda_best_practices.md   # RAG knowledge base (the agent's rulebook)
├── src/
│   ├── rag.py                  # builds the vector index, retrieves sections
│   ├── tools.py                # deterministic EDA functions + test data
│   ├── agent.py                # plan -> execute -> report pipeline
│   └── report.py               # Markdown -> self-contained HTML report
├── data/
│   ├── synthetic_test.csv      # test dataset with planted problems
│   └── uploads/                # files uploaded via the app (gitignored)
├── chroma_db/                  # vector index (gitignored, rebuilt locally)
├── outputs/                    # reports, run logs, plots (gitignored)
├── app.py                      # Streamlit web application
├── requirements.txt
└── .env                        # API key (gitignored, never committed)
```

## Setup

```bash
git clone <repo-url>
cd eda-agent
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

On Windows, if PowerShell refuses to run the activation script ("running
scripts is disabled on this system"), run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and activate
again. If `python` opens a Microsoft Store prompt, use `py -m venv .venv`
to create the environment; inside the activated environment, `python`
works normally.

All commands below must be run from the project root, not from inside
`src/`. Every script resolves its paths (`knowledge/`, `data/`,
`outputs/`, `chroma_db/`) relative to the directory it is launched from.

### LLM access

The agent uses the Groq API, whose free tier requires no credit card and
is limited by request and token rate limits rather than cost. Create a key
at [console.groq.com](https://console.groq.com) and add it to a `.env`
file in the project root:

```
GROQ_API_KEY=gsk_your_key_here
```

The main model is `openai/gpt-oss-120b`, with `qwen/qwen3.8-27b` as an
automatic fallback. Groq retires models periodically; if a model returns a
404 "model not found" error, list the models currently available to your
account and update the `MODELS` list in `src/agent.py`.

### Knowledge base and RAG index

```bash
python src/rag.py
```

Builds the retrieval index from `knowledge/eda_best_practices.md`. The
document contains 13 sections of concrete EDA rules with explicit
thresholds (report structure, dataset overview, missing values,
duplicates, numeric distributions, skewed distributions, outliers,
categorical variables, high-cardinality categoricals, correlation, target
variable, target leakage, and recommendations). It is chunked at each `##`
heading, so every retrieved chunk is a complete, self-contained rule set
rather than an arbitrary text window. Chunks are embedded locally with
`all-MiniLM-L6-v2` (a MiniLM sentence-transformer, Reimers & Gurevych,
2019; Wang et al., 2020), so retrieval needs no API and costs nothing, and
they are stored in a persistent ChromaDB collection in `chroma_db/`. The
script finishes with four test queries to confirm that retrieval returns
the right section (for example, "a column has 60% missing values" →
*Missing values*). The index is rebuilt from scratch on each run, so it
must be rerun whenever the knowledge base is edited.

### Analysis tools and synthetic test data

```bash
python src/tools.py
```

`src/tools.py` defines the deterministic analysis functions the agent can
call: `profile` (shape, column types, duplicate rows, ID-like columns),
`missing_values`, `distributions` (mean, median, standard deviation,
range, skewness), `outliers` (IQR or z-score), `categorical_summary`
(cardinality, top values, inconsistent spelling), and `correlations`
(Spearman or Pearson, with a strong-pair threshold). Each returns a
JSON-serialisable summary and saves its plots to `outputs/plots/`.
ID-like columns (integers unique on every row) are detected and excluded
from statistics automatically. The agent can only call functions listed
in the module's `TOOLS` registry, so it can never execute arbitrary code.

Running the module directly generates `data/synthetic_test.csv`, a
1,005-row customer dataset with seven deliberately planted problems: an
ID column, 30% missing values in `age`, ten impossible ages of 250, a
heavily skewed log-normal `income`, a `spend` column derived from
`income`, inconsistent category spelling ("Dublin" vs "dublin"), and five
duplicated rows. It then runs every tool on that dataset, so the output
can be checked directly against the known ground truth.

### Agent pipeline

```bash
python src/agent.py                       # analyses data/synthetic_test.csv
python src/agent.py data/your_file.csv    # analyses any other CSV
```

Runs the full three-stage pipeline. In the **planning** stage, the
dataset is profiled, relevant guidelines are retrieved, and the LLM
returns a JSON plan listing which tools to run, with what arguments, and
a one-sentence reason for each step citing the guideline that justifies
it. The plan is then validated before anything runs: unknown tools and
invalid arguments are dropped, and any core step the LLM omitted
(missing values, distributions, outliers, correlations) is added back, so
coverage stays consistent even though the plan itself varies between
runs. In the **execution** stage the tools run deterministically; a tool
that fails with the LLM's chosen arguments is retried with its defaults.
In the **reporting** stage, the agent turns its actual findings into new
retrieval queries (for example "highly skewed numeric distribution"
or "column with 30% missing values"), retrieves the matching guidelines,
and the LLM writes the report under explicit rules: use only numbers that
appear in the results, name the guideline applied, flag problems without
claiming any data was changed, and end with prioritised recommendations
that each name a column and a reason.

The LLM wrapper retries with increasing waits on rate-limit errors,
falls back to the backup model if the main one stays unavailable, and
extracts the JSON plan even when the model surrounds it with extra text,
using a default plan if parsing fails entirely. Outputs are saved to
`outputs/report.md`, `outputs/plots/`, and `outputs/run_log.json`, which
records the profile, the validated plan, every computed result, and the
knowledge-base sections retrieved at each stage, giving a full audit
trail for each report.

### Web application

```bash
streamlit run app.py
```

Launches a browser interface at `http://localhost:8501`. A user uploads
a CSV, previews it, and runs the analysis; results are shown in three
tabs (the written report, all generated plots, and the agent's plan with
the guidelines it retrieved), with downloads for the report as Markdown or
as a single self-contained HTML file with every plot embedded
(`src/report.py`). Results are held in Streamlit's session state so that
they persist across the reruns Streamlit triggers on every interaction.

### Verifying a report

Every number stated in `outputs/report.md` should appear in
`outputs/run_log.json`; a number that does not was introduced by the LLM
rather than computed by a tool. On the synthetic dataset, the report
should identify all seven planted problems. Recommended real-world test
datasets are Titanic (missing values, categorical and ID columns), House
Prices (wide, skewed, strongly correlated), Wine Quality
(semicolon-separated, all numeric, outliers), Telco Customer Churn
(numbers stored as text), and Iris, a small clean dataset on which a good
agent should *not* report problems that are not there.

## Known limitations

Dates and formatted numbers (for example `2024-03-15` or `"€1,200"`) are
read as text and therefore analysed as categorical columns. Datasets wider
than roughly 40 columns can exceed the Groq free tier's token-per-minute
limit, and the correlation heatmap becomes hard to read; distribution
plots are capped at 20 columns. There is not yet an option to designate a
target column, so the knowledge base's target-variable and target-leakage
guidance is never triggered. Very large files run slowly, mostly because
of plotting, and sampling is recommended for EDA purposes.

## References

- Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N.,
  Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D.
  (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks.
  *Advances in Neural Information Processing Systems (NeurIPS)*, 33,
  9459–9474.
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings
  using Siamese BERT-networks. *Proceedings of EMNLP-IJCNLP*, 3982–3992.
- Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley.
- Wang, L., Xu, W., Lan, Y., Hu, Z., Lan, Y., Lee, R. K.-W., & Lim, E.-P.
  (2023). Plan-and-Solve prompting: Improving zero-shot chain-of-thought
  reasoning by large language models. *Proceedings of ACL*, 2609–2634.
- Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., & Zhou, M. (2020).
  MiniLM: Deep self-attention distillation for task-agnostic compression
  of pre-trained transformers. *Advances in Neural Information Processing
  Systems (NeurIPS)*, 33, 5776–5788.

## Contributors

- Niraj Palve (25200843)
