# ILP-CoT: Inductive Logical Rule Induction with ILP‑constrained Chain‑of‑Thought (reference implementation)

> This repo demonstrates **ILP‑CoT** — integrating **Inductive Logic Programming (ILP)** into a custom Chain‑of‑Thought (CoT) pipeline to induce **verifiable, interpretable rules** from a few **positive/negative** examples. The key ideas are: build **meta‑rules** to constrain hypothesis space and run a **failure‑reflection loop** to correct perception/abduction errors; ILP then performs **formal verification** against positives/negatives to curb hallucinations. fileciteturn2file4L1-L6 fileciteturn2file12L1-L10

---

The entry point is `main.py`. At a high level the loop does:

1) Prepare device/paths and select target classes. fileciteturn2file6L15-L19  
2) Ask an MLLM to **propose capture tokens & facts**, then delete noisy facts → extract Prolog facts for **positives** and **negatives**. fileciteturn2file6L20-L35  
3) Ask the MLLM to **generalize concrete rules into meta‑rules**, then **build a Prolog program** from facts + meta‑rules and call the Prolog backend. fileciteturn2file6L37-L43  
4) If Prolog times out or fails, first **re‑verify each predicate** on positives and rebuild; if still failing, **re‑propose tokens/facts** for both pos/neg and retry. fileciteturn2file6L44-L65 fileciteturn2file1L8-L19  
5) Post‑process Prolog rules, convert to **natural language** (or pick with CLIP when multiple). fileciteturn2file0L21-L37  
6) Control flow via `--max-attempts`; Prolog timeout string is `"Query exceeded timeout of 100 seconds."`. fileciteturn2file0L17-L19 fileciteturn2file6L9-L13

CLI flags: `--num-classes/-n` and `--max-attempts/-m`. fileciteturn2file0L46-L63

---

## Features

- **Meta‑rule construction** from concrete rule drafts → reusable second‑order templates for **Metagol**, which strongly constrains search and yields interpretable candidates. fileciteturn2file5L28-L40 fileciteturn2file12L11-L18  
- **Formal ILP induction and verification** (with positives & **negatives as hard constraints**) to reduce hallucinations. fileciteturn2file12L16-L26 fileciteturn2file14L1-L8  
- **Failure‑reflection loop**: token splitting & re‑grounding; hypothesis‑space cropping/resampling; pipeline restart up to a max number of iterations. fileciteturn2file7L1-L12  
- **Rule explanation**: Prolog rules are translated to natural language while preserving logical fidelity. fileciteturn2file9L5-L8

---

## Installation

- Python 3.9+ (tested with standard libs).  
- An **ILP/Prolog backend** compatible with **Metagol** (the paper’s experiments use Metagol and compare it to Popper/ILASP). fileciteturn1file4L69-L75  
- Access to an **MLLM** used by the `query_GPT` calls (vision‑language API).

> Tip: ensure your Prolog executable is on `PATH` so the `prolog.capture_prolog_output(...)` call can run. fileciteturn2file0L14-L19

---

## Data layout

Provide a small set of labeled images split into positive/negative examples, e.g.:

```
data/
  pos/
    img_000.jpg
    ...
  neg/
    img_100.jpg
    ...
```

Negative examples serve as **binding constraints** during induction and markedly improve rule quality. fileciteturn2file10L15-L25

---

## Quickstart

```bash
# 2 target classes, up to 5 attempts (defaults from main.py)
python main.py -n 2 -m 5
```

The loop will print (i) proposed & filtered facts, (ii) Prolog outputs, (iii) post‑processed Prolog rules and the final **natural‑language rule**. fileciteturn2file0L21-L39

---

## How it works (paper ↔ code)

| Paper concept | This repo (main.py) |
| --- | --- |
| Generate **capture tokens** & logical facts for pos/neg | Propose → delete noisy → `tools.extract_prolog_content(...)` for each split. fileciteturn2file6L23-L35 |
| **Concrete → meta‑rule** generalization; constrain hypothesis space | `prompt.Build_meta_rules_prepration(...)` → `query_GPT.text_to_text(...)` → `tools.extract_metarules(...)` → `prompt.Build_PL_file(...)`. fileciteturn2file6L37-L43 fileciteturn2file5L28-L36 |
| **ILP induction** (Metagol) + verification against pos/neg | `prolog.capture_prolog_output(...)` then `prolog.post_rules_process(...)`. fileciteturn2file0L14-L23 fileciteturn2file12L11-L18 |
| **Failure reflection** when no valid rule | Per‑predicate re‑grounding → rebuild PL → full re‑proposal → retry until `--max-attempts`. fileciteturn2file1L8-L19 fileciteturn2file6L56-L65 fileciteturn2file7L1-L12 |
| Natural‑language rule(s) | `prompt.rules_to_nl(...)` → MLLM → optional CLIP selection. fileciteturn2file0L30-L37 fileciteturn2file9L5-L8 |

---

## FAQ

**Q: Prolog times out / no answer?**  
A: The code treats `"Query exceeded timeout of 100 seconds."` as a timeout and triggers reflection (re‑check predicates, then rebuild facts/meta‑rules). Increase `-m` if needed. fileciteturn2file6L9-L13 fileciteturn2file1L8-L19


