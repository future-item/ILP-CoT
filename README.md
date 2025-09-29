# Abductive Visual Rule Induction by Bridging Inductive Logic Programming and Multimodal Large Language Models (arXiv preprint https://arxiv.org/abs/2509.21874 Reference Implementation)

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

## Input data (image hosting) -> inference

> Because the local/automatic image-host loading was removed from the code, you must **upload images to an image host first** (publicly accessible HTTP(S) direct links), then put those image URLs into the parameters in `preparation.py` to run **ILP-CoT** inference.

### Step 1: Upload positive/negative samples to any host that serves direct links
- Object storage/CDN (S3, OSS, GCS), a static site, GitHub raw, or any other service that gives **publicly readable** direct links is fine.
- Requirements: URLs must be **HTTPS and directly downloadable**, no login required (or use long-lived pre-signed URLs).

A typical layout:
```
https://<your.cdn>/ilpcot/catdog/pos/...
https://<your.cdn>/ilpcot/catdog/neg/...
```

> Optional: if you only have **full URLs**, you can put full URLs directly into the filename lists below and set the matching `*_images_path` to an empty string `""`.

### Step 2: Fill the image host base URLs and filenames in `preparation.py`
Change the four variables to your own host paths and filenames (or full URLs). Example using **“base URL + filenames”**:

```python
# preparation.py (excerpt)
def preparation(index: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Option A: base URL + filenames
    input_images_path = [
        ["https://cdn.example.com/ilpcot/catdog/pos/"],
        # ... add more class groups here
    ]
    input_images_path_neg = [
        ["https://cdn.example.com/ilpcot/catdog/neg/"],
        # ...
    ]
    pos_image_filenames = [
        ["cat1.jpg", "cat2.jpg", "dog1.jpg"],
        # ...
    ]
    neg_image_filenames = [
        ["road1.jpg", "tree2.jpg"],
        # ...
    ]

    # Keep the rest unchanged
    target_classes = [['cat', 'dog'], ...]
    target_num = [[1, 1], ...]
    choose_class = index
    input_images_path = input_images_path[choose_class][0]
    input_images_path_neg = input_images_path_neg[choose_class][0]
    pos_image_filenames = pos_image_filenames[choose_class]
    neg_image_filenames = neg_image_filenames[choose_class]
    return device, input_images_path, input_images_path_neg, pos_image_filenames, neg_image_filenames, target_classes, target_num, choose_class
```

> Option B: if your host does **not** provide a clean common base, put **full URLs** directly into `pos_image_filenames` / `neg_image_filenames`, and set `*_images_path` to `""` (empty string).

### Step 3: Run inference
```bash
# pick the class group index (e.g., 0 for ['cat','dog']) and allow up to 5 retries
python main.py -n 0 -m 5
```

At runtime, the pipeline takes your **positive/negative sample URLs** from `preparation.py`, queries the multimodal model to extract candidate facts, filters noise, builds a **Prolog** program, runs **ILP** induction and verification, and finally emits **verifiable rules** plus a **natural-language explanation**.

### CLI notes
- `-n / --num-classes` is used as the **class group index** passed into `preparation.preparation(index)` (0-based). Make sure it is within range of your `target_classes` list.
- `-m / --max-attempts` controls the number of failure-reflect-retry loops.

### Troubleshooting
- **403/404 or not loading**: ensure the URL is public, HTTPS, and not expired (for pre-signed links).
- **Timeouts**: `"Query exceeded timeout of 100 seconds."` is treated as a timeout and triggers failure-reflect-retry. Increase `-m` if needed.
- **Unbalanced positives/negatives**: negatives are used as **hard constraints** during verification. Too few negatives weaken constraints; prepare several near-miss images that do **not** contain the target relation/attribute.

> Local debugging tip: serve a folder via `python -m http.server` and use `http://127.0.0.1:8000/...` as your “image host” URLs during quick tests.
