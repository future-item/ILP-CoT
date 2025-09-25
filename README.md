# ILP-CoT: Inductive Logical Rule Induction with ILP‑constrained Chain‑of‑Thought (reference implementation)

> This repo demonstrates **ILP‑CoT** — integrating **Inductive Logic Programming (ILP)** into a custom Chain‑of‑Thought (CoT) pipeline to induce **verifiable, interpretable rules** from a few **positive/negative** examples. The key ideas are: build **meta‑rules** to constrain hypothesis space and run a **failure‑reflection loop** to correct perception/abduction errors; ILP then performs **formal verification** against positives/negatives to curb hallucinations.
---

The entry point is `main.py`. At a high level the loop does:

1) Prepare device/paths and select target classes.   
2) Ask an MLLM to **propose capture tokens & facts**, then delete noisy facts → extract Prolog facts for **positives** and **negatives**. 
3) Ask the MLLM to **generalize concrete rules into meta‑rules**, then **build a Prolog program** from facts + meta‑rules and call the Prolog backend. 4) If Prolog times out or fails, first **re‑verify each predicate** on positives and rebuild; if still failing, **re‑propose tokens/facts** for both pos/neg and retry. 
5) Post‑process Prolog rules, convert to **natural language** (or pick with CLIP when multiple). 6) Control flow via `--max-attempts`; Prolog timeout string is `"Query exceeded timeout of 100 seconds."`. 
CLI flags: `--num-classes/-n` and `--max-attempts/-m`. ---

## Features

- **Meta‑rule construction** from concrete rule drafts → reusable second‑order templates for **Metagol**, which strongly constrains search and yields interpretable candidates.  
- **Formal ILP induction and verification** (with positives & **negatives as hard constraints**) to reduce hallucinations. 
- **Failure‑reflection loop**: token splitting & re‑grounding; hypothesis‑space cropping/resampling; pipeline restart up to a max number of iterations. - **Rule explanation**: Prolog rules are translated to natural language while preserving logical fidelity. 

---

## Installation

- Python 3.9+ (tested with standard libs).  
- An **ILP/Prolog backend** compatible with **Metagol** (the paper’s experiments use Metagol and compare it to Popper/ILASP). 
- Access to an **MLLM** used by the `query_GPT` calls (vision‑language API).

> Tip: ensure your Prolog executable is on `PATH` so the `prolog.capture_prolog_output(...)` call can run. 

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

Negative examples serve as **binding constraints** during induction and markedly improve rule quality. ---

## Quickstart

```bash
# 2 target classes, up to 5 attempts (defaults from main.py)
python main.py -n 2 -m 5
```

The loop will print (i) proposed & filtered facts, (ii) Prolog outputs, (iii) post‑processed Prolog rules and the final **natural‑language rule**.  
---

## How it works (paper ↔ code)

| Paper concept | This repo (main.py) |
| --- | --- |
| Generate **capture tokens** & logical facts for pos/neg | Propose → delete noisy → `tools.extract_prolog_content(...)` for each split. 
| **Concrete → meta‑rule** generalization; constrain hypothesis space | `prompt.Build_meta_rules_prepration(...)` → `query_GPT.text_to_text(...)` → `tools.extract_metarules(...)` → `prompt.Build_PL_file(...)`.
| **ILP induction** (Metagol) + verification against pos/neg | `prolog.capture_prolog_output(...)` then `prolog.post_rules_process(...)`. 
| **Failure reflection** when no valid rule | Per‑predicate re‑grounding → rebuild PL → full re‑proposal → retry until `--max-attempts`. | Natural‑language rule(s) | `prompt.rules_to_nl(...)` → MLLM → optional CLIP selection. 

---

## FAQ

**Q: Prolog times out / no answer?**  
A: The code treats `"Query exceeded timeout of 100 seconds."` as a timeout and triggers reflection (re‑check predicates, then rebuild facts/meta‑rules). Increase `-m` if needed.


