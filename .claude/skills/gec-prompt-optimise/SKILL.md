---
name: gec-prompt-optimise
description: Propose the next GEC system-prompt iteration by running an ERRANT per-category diagnostic on the current best run, picking ONE targeted rule, and registering a new prompt_NN_*.py + matching config + RESULTS.md row. Use when the user asks to iterate the prompt, propose a next variant, beat the current best, or "what should we try next" for BEA/CoNLL GEC.
---

# GEC prompt optimisation — rule-driven iteration

You iterate prompts by **diagnose → propose-one-change → register → eval → log**. Every iteration changes exactly one variable so deltas are attributable. No automated search (no GEPA, no APO) — this skill is purely rule-driven.

The canonical example to mimic is `src/agents/prompts/prompt_19_taxonomy_optimised_gemini_family_v5_orth_tight_punct_explicit_comma_splice.py` — read its header before you start.

## Phase 1: Diagnose the current best

1. **Find the parent.** Ask the user which prompt to iterate on, or scan `RESULTS.md` for the **bolded** best-F0.5 row under the target model section. Identify the matching `outputs/<run>/` directory by the `<split>_<model>_<prompt_name>_..._batch_<N>_temp...` naming convention.

2. **Get per-ERRANT-category P/R/F0.5.** The codebase only writes overall metrics; the category breakdown has to be regenerated. Run:

   ```bash
   PARENT=outputs/<parent-run>
   SPLIT_SRC=data/en_bea/dev/bea-dev.src         # or conll-2014 / test src, match the run
   SPLIT_M2=data/en_bea/dev/bea-dev.m2

   uv run errant_parallel -orig "$SPLIT_SRC" \
       -cor "$PARENT/predictions.eval.txt" \
       -out "$PARENT/predictions.m2"

   uv run errant_compare -hyp "$PARENT/predictions.m2" \
       -ref "$SPLIT_M2" -cat 2 \
       | tee "$PARENT/errant_per_category.txt"
   ```

   `-cat 2` is type-level (PUNCT, DET, ORTH, SPELL, MORPH, VERB:TENSE, NOUN:NUM, ...). Do not use `-cat 1` (operation only) or `-cat 3` (op+type) — type-level is what surfaces actionable rules.

3. **Rank categories.** Pull the table from `errant_per_category.txt`. Sort by FN descending for "missed corrections that we could add rules for"; sort by FP descending for "over-corrections that need a tightening rule". The biggest F0.5 movers are usually PUNCT, DET, ORTH, SPELL, MORPH, VERB:TENSE, NOUN:NUM.

4. **Inspect 10–20 real examples in the target category.** Don't propose a rule blind. Find sentences where the reference makes that edit and the prediction doesn't (or makes a wrong one). Quick recipe:

   ```bash
   # diff prediction vs reference line-by-line; the line numbers map to .src/.ref0/.m2
   paste -d'\t' "$SPLIT_SRC" data/en_bea/dev/bea-dev.ref0 "$PARENT/predictions.eval.txt" \
       | awk -F'\t' '$2 != $3' | head -40
   ```

   Then read `$SPLIT_M2` around those sentence indices to confirm the error-type label.

5. **Write the diagnostic in one sentence:** `"<CAT> had <FN> FN. Of those, ~<estimate> are <specific targetable sub-pattern>."` If you can't write a *specific sub-pattern*, the category isn't ready for a rule yet — pick the next-worst category.

## Phase 2: Propose exactly one change

**Rules for the proposal — non-negotiable:**

- **One variable per iteration.** Either *add one narrow targeted rule* OR *tighten one existing rule* — never both, never two new rules, never a rule plus a reorder. If you have two good ideas, split into two iterations.
  - **Why:** the whole point of this loop is attributable deltas. Two changes at once means the next iteration's delta is uninterpretable.
- **High-precision additions only.** F0.5 weights precision 2x recall. A rule that adds 50 TPs and 50 FPs is a wash; 30 TPs / 5 FPs is a win. If the rule can't beat ~80% precision on the diagnostic sample you inspected, don't propose it.
- **Be specific to the point of pedantry.** "Fix punctuation" is not a rule. The rule must be sharp enough that a junior annotator could apply it consistently. See `prompt_19` line 27–40 for the gold standard.
- **Predict the effect explicitly.** State expected TP gain in target category, FP risk in adjacent categories, and what observation would falsify the rule (e.g. "if PUNCT FP increases by >20 we over-fired").
- **No restructure + content change in the same iteration.** If you want to reorganise the prompt's bullet order or rewrite preamble, do that as its own no-content-change iteration so the reorder cost is attributable.

## Phase 3: Register the new prompt

Pick `NN` = next free integer after the highest `prompt_NN_*.py` in `src/agents/prompts/`. Three coordinated edits are required — none is optional:

### 3a. Create `src/agents/prompts/prompt_NN_<parent_slug>_<change_slug>.py`

Header comment must follow the lineage convention from `prompt_19`:

```python
# vN = vN-1 (<parent prompt_name key>, F0.5=<parent F0.5 to 4dp>) + <one-line change>.
# Diagnostic: <CAT> had <FN> FN. Of those, ~<count> are <specific pattern>.
# Expected: +<N> TPs in <CAT>, minimal FP risk in <other CATs>.
PROMPT_VARIABLE_NAME = (
    "You are a precision-focused grammatical error correction system.\n\n"
    ...
)
```

Use the same string-concatenation style as the existing prompts. Do not switch to triple-quoted strings, f-strings, or Jinja — consistency matters for diffs.

### 3b. Wire into `src/agents/prompts/base.py`

Two edits in this file:
- Add `from .prompt_NN_<slug> import (PROMPT_VARIABLE_NAME)` at the bottom of the import block (chronological order, not alphabetical — match the existing pattern).
- Add `"<prompt_name_key>": PROMPT_VARIABLE_NAME,` to the `GEC_PROMPTS` dict, in the same group as its parent (gepa group, gemini-tuned group, gpt41mini-tuned group, taxonomy-optimised-gemini-family group, etc. — see the blank-line groupings in the existing dict).

The `<prompt_name_key>` is what goes into the config's `prompt_name:` field. Keep it short — drop the `prompt_NN_` and the file's `_v<N>` redundancy isn't needed if the key already carries lineage (`taxonomy_optimised_gemini_family_v6_<change>`).

### 3c. Create `config.en.<split>.<short_change>.yaml`

Copy the parent's config (the one in `outputs/<parent-run>/config.en.*.yaml`) and change **only** `prompt_name:`. Do not adjust model, batch_size, temperature, few-shot config, seed, num_threads — anything else changing breaks attribution.

For dev iterations name it `config.en.dev.<short_change>.yaml`. For test runs, only create the test config *after* the dev iteration wins.

## Phase 4: Eval + log

1. **Run dev eval:**
   ```bash
   uv run python en_main.py --config config.en.dev.<short_change>.yaml
   ```

2. **Read the new `outputs/<run>/metrics.txt`.** Compare to parent.

3. **Re-run the per-category breakdown on the new run** (same commands as Phase 1 step 2) and check that the targeted category actually moved. If F0.5 went up but the targeted category didn't move, you got lucky elsewhere — flag this; the rule may not be the cause.

4. **Append a result row to `RESULTS.md`** under the appropriate model section. Match the existing column layout exactly (Prompt / Batch / Exact % / Precision / Recall / F0.5). Group with the parent under the same separator line. Bold the F0.5 if it took the lead for that model.

5. **Append a result note to the prompt file's header:**
   ```python
   # Result: F0.5=<x.xxxx> (Δ=<+/-y.yyyy> vs parent). Target CAT: <CAT> F0.5 <before>→<after>.
   ```
   This closes the loop — next iteration's parent inspection can read the result without re-running anything.

6. **Decide on the next parent:**
   - **F0.5 improved AND target category moved in the predicted direction:** new prompt is the parent.
   - **F0.5 improved but target category didn't move:** keep current parent; the win may be noise. Re-run with a different seed before adopting.
   - **F0.5 unchanged or dropped:** parent stays parent. Next iteration must target a *different* variable — do not tweak the same rule, that's chasing noise.

## Hard "do not" list

- Do **not** combine two rules in one iteration, even if both look promising.
- Do **not** propose a rule without inspecting actual error examples in Phase 1 step 4.
- Do **not** skip the per-category breakdown — overall F0.5 hides where the change is actually landing.
- Do **not** modify the agent code (`src/agents/gec_agent.py` etc.), the evaluation pipeline, or unrelated prompts as part of an iteration. Prompt-only changes.
- Do **not** change config knobs other than `prompt_name:` in the new config — you'd be confounding the experiment.
- Do **not** run GEPA, APO, or any automated search inside this skill. If the user wants those, that's a separate tool (`gepa_optimize.py`); offer it but don't invoke it here.
- Do **not** delete or rename the parent prompt file or its config, even if the new prompt wins. Keep lineage walkable.
