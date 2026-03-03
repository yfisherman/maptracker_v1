# MapTracker Temporal-Gating MVP Repair Walkthrough

## Executive repair strategy

A defensible MVP claim (per `docs/mvp_spec.md`) requires three things to be simultaneously true:

- **The training/eval pipeline actually runs** for the multi-frame memory setting (no hidden runtime traps in the corruption + gating wiring). citeturn3view0  
- **The mechanism matches the intended story at the exact call sites**: the gate reads **`q_cur` after BEV-to-vector cross-attention**, applies a **slotwise scalar alpha**, **shared across heads**, to **values only**, **pre-fusion**, in the decoder’s memory path. citeturn12view1turn6view2turn3view1  
- **The corruption protocol is faithful** to “read-path-only stale-memory substitution,” without accidentally injecting out-of-scope corruption modes (notably pose mismatch). citeturn12view0turn5view0  

From static inspection, the repo is close on the “shape of the mechanism,” but it is **not defensible yet** because:

- There is a **hard blocking runtime/correctness bug**: previous-frame metas are corruption-tagged using `clip_corruption_mode/clip_stale_offset` before those variables are assigned in `forward_train`. citeturn3view0  
- The decoder currently gates memory using a query tensor that is **not clearly `q_cur` (post-BEV cross-attention)** as required by spec; it instead gates using a reshaped slice of `query` while BEV cross-attention result is tracked separately as `query_bev`. citeturn12view1turn6view2  
- The stale substitution path uses **canonical unpropagated embeddings** as stale sources, while the “clean selected memory” embedded for attention is pose-propagated—this silently blends “stale semantic memory” with “pose misalignment” corruption. citeturn5view0  

Minimum “defensible MVP” bar (what must be true after repairs):

- Multi-frame training path **does not crash** and applies **one clip-consistent corruption setting** to all frames. citeturn3view0  
- In decoder memory cross-attn, the gate and memory attention use **the BEV-updated query (`q_cur`)** and perform **value-only scaling** at the **pre-fusion** insertion point. citeturn12view1turn6view2turn3view1  
- Corruption is strictly **read-path-only**, and “stale substitution” does not implicitly introduce out-of-scope pose corruption. citeturn12view0turn5view0  
- The “no-gate baseline” cannot accidentally run with gate-forward enabled (parity guard, not just loss guard). citeturn3view4turn8view0  

## Priority-ordered fix list

| Priority | Issue | Why it matters | Minimal fix | Files / functions likely affected | Risk of regression | Required validation |
|---|---|---|---|---|---|---|
| P0 | Corruption meta injection order bug (undefined variables) | Can crash multi-frame training or silently skip intended corruption protocol; invalidates any training conclusions | Define `clip_corruption_mode` + `clip_stale_offset` **before** applying to `all_prev_data` metas; enforce “one per clip” | `plugin/models/mapers/MapTracker.py` (`forward_train`) citeturn3view0 | Low (pure reorder) | Add targeted unit or static check for ordering; run a minimal multi-frame forward_train smoke under torch |
| P1 | Gate uses non-spec `q_cur` (post-BEV) | Breaks the causal story: “contradiction = mismatch between current perception and memory”; undermines spec compliance | In memory cross-attn branch, use `query_bev` (BEV-updated tensor) as the query for both gating + memory attention | `plugin/models/transformer_utils/MapTransformer.py` (`MapTransformerLayer.forward`) citeturn12view1turn6view2 | Medium (changes signal into gate + memory attention) | Add a targeted invariant test or static validation: gate call uses `query_bev`; run a short training/eval smoke |
| P1 | Stale substitution introduces pose mismatch corruption | Violates MVP scope (spec excluded translation/rotation corruption); can change what the gate learns | Replace stale source using **propagated clean selected** entries when available (instead of raw canonical); mark missing-source ineligible | `plugin/models/mapers/vector_memory.py` (`_build_local_corrupted_read_view`) citeturn5view0turn12view0 | Medium (changes corruption semantics) | Extend unit tests for `_build_local_corrupted_read_view` to assert propagated-source usage and eligibility behavior |
| P1 | No-gate baseline parity not enforced at forward-pass level | If gate-forward stays enabled while loss is disabled, baseline isn’t “no-gate”; comparisons become contaminated | Add an explicit runtime guard: when `corruption_trained_no_gate_baseline=True`, forcibly disable gate modules (or assert they are disabled) | `plugin/models/mapers/MapTracker.py` (train init or early `forward_train`) + transformer layer traversal | Low–Medium (depends how module graph is accessed) | Add assertion-based check; add config-based check; run B1/B2 smoke ensuring alpha=1 in baseline |
| P1 | Gate feature instability / spec mismatch: missing LayerNorm | Spec mandates LN; without it, gate may saturate or behave inconsistently across layers | Apply `layer_norm` to q and mem embeddings before cosine/L2 and MLP | `plugin/models/transformer_utils/MapTransformer.py` (`SlotwiseTemporalGate.forward`) citeturn12view2turn3view1 | Medium (changes gate behavior) | Update unit tests to ensure outputs are finite; run brief training step to confirm no NaNs and alpha distribution non-degenerate |
| P2 | “Per-query-slot scaling approximation” ambiguity (Gap 2 framing) | Status doc claims approximation; tests call gate with `Q>1`, which is misleading because real call path forces `Q=1` | Enforce and document invariant: gate is only exact when called with `q_len==1`; update unit tests to match intended call path | `SlotwiseTemporalGate.forward` + `tests/test_temporal_gate_mvp.py` + `docs/codex_mvp_status.md` citeturn3view4turn3view1turn6view2turn10view0 | Low | Unit tests pass after reshaping; static validator updated; optional extra assert in gate |
| P2 | Mask semantics are implicit (Gap 1 defensibility) | Spec calls masks “non-negotiable”; current proxy (`valid_track_idx` + padding) is brittle to future changes | Add minimal explicit guard: store track-query boundary mask/length in `memory_bank`, and enforce `valid_track_idx` cannot include “new” query slots; optionally thread an `is_propagated` boolean | `MapTracker._batchify_tracks` (track length) + `MapTransformerLayer.forward` | Low–Medium | Add assertions + a targeted unit test that fails if valid indices exceed tracked query length |
| P3 | Optional spec alignment: sinusoidal `pe_age` feature | Spec recommends PE; current normalized delta_t + rank may be sufficient | Defer unless gate collapses; if implemented, keep `d_pe=8` and minimal code | `SlotwiseTemporalGate` input dim + MLP | Medium (changes param shapes/checkpoint compat) | Only after core fixes; add unit tests for shape + finiteness |

## Detailed fix walkthroughs

### Fix A — P0 corruption meta injection order bug

#### Problem statement

In `MapTracker.forward_train`, inside the `all_prev_data` loop, the code calls:

- `self._inject_memory_corruption_meta(img_metas_prev, clip_corruption_mode, clip_stale_offset)`

but `clip_corruption_mode` and `clip_stale_offset` are assigned **later** in the same function. This is visible in the current control flow: the injection call appears before the assignments to `clip_corruption_mode` and `clip_stale_offset`. citeturn3view0  

#### Why this is truly a bug/gap

This is a **hard runtime correctness bug**: in Python, referencing a local variable before assignment raises an exception (or yields undefined behavior if refactored). This blocks multi-frame training—the exact scenario where memory and corruption matter. citeturn3view0  

It also invalidates the spec requirement that corruption settings be consistent across a clip (“one corruption state per clip”). Even if the code didn’t crash (e.g., after some partial refactor), the ordering would be structurally wrong. citeturn12view0turn3view0  

#### Minimal viable fix

Reorder logic so that:

1. Sample/resolve clip-level corruption choices **once** at the start of `forward_train`, before iterating `all_prev_data`.
2. Inject this metadata into **all** frame metas: all prev frames + current frame.

Concretely, move this block:

- `clip_corruption_mode = ...`
- `clip_stale_offset = ...`

to immediately before:

- the `if all_prev_data is not None:` section (or at least before the `for prev_data in all_prev_data:` loop), and keep:

- `self._inject_memory_corruption_meta(img_metas, ...)` after current frame meta acquisition (fine to keep where it is as long as vars exist).

Do **not** change corruption sampling policy beyond reorder; keep current logic that overrides for `corruption_trained_no_gate_baseline`. citeturn3view0  

#### Why this fix is preferable to alternatives

- It is the smallest possible change that restores correctness and the intended “clip-constant” semantics.
- It does not require altering corruption logic, distributions, or downstream consumers. citeturn3view0  

#### Regression / interaction risks

- **Low**: functionally, it just makes earlier frames receive the same meta they were intended to receive anyway.
- Potential subtle risk: if `_sample_corruption_mode()` relied on state that is only valid after processing prev frames (unlikely). Your current code already intends to sample once, independent of frame processing. citeturn3view0  

#### Verification plan for this fix

Targeted checks:

- Add a tiny static/unit check that parses `MapTracker.py` and confirms that `clip_corruption_mode` appears before `_inject_memory_corruption_meta(img_metas_prev, ...)`.
- Runtime smoke (minimal): run a small multi-frame forward_train with dummy metas and confirm no exception and that all metas in `all_prev_data` and `img_metas` contain identical `memory_corruption_mode` and `memory_stale_offset` fields afterward.

#### Acceptance criteria

- Multi-frame `forward_train` does not error due to undefined variables.
- All frames in a clip share the same injected corruption mode/offset.

---

### Fix B — P1 enforce true “no-gate baseline” at forward-pass level

#### Problem statement

The training code disables adding gate loss when `corruption_trained_no_gate_baseline=True`:

- `gate_loss_active = self.gate_supervision_enabled and _use_memory and (not self.corruption_trained_no_gate_baseline)` citeturn8view0  

But disabling loss **alone** is not sufficient to guarantee that the baseline run is actually “no gate,” unless configs also disable `SlotwiseTemporalGate.enabled`. The repo status doc claims parity control exists, but the explicit code evidence shown is loss-side. citeturn3view4turn8view0  

#### Why this is truly a bug/gap

This is an **experimental-defensibility gap**: if a “baseline” accidentally leaves gate-forward enabled, you will get a silently contaminated comparison:

- B1 (supposedly no gate) still gates memory values in forward
- but is not penalized by gate supervision loss

That is not a clean “no-gate baseline parity” story; it mixes mechanisms. citeturn3view4turn8view0  

#### Minimal viable fix

Add a **runtime guard that cannot be bypassed by misconfiguration**:

Option (minimal and robust): **assertion + optional hard-disable**

- In `MapTracker` initialization or at the start of `forward_train`:
  - Traverse the transformer decoder layers and confirm `temporal_gate.enabled == False` when `corruption_trained_no_gate_baseline=True`. If not, raise with a clear message.
- If you want “self-healing” instead of a hard fail:
  - Force-disable: set `layer.temporal_gate.enabled = False` for all layers when in baseline mode.

This stays in MVP scope: it does not change core model behavior beyond guaranteeing the baseline semantics required by spec. citeturn3view4turn12view0  

#### Why this fix is preferable to alternatives

- It is localized, does not require config archaeology, and prevents the single most common failure mode in ablation comparisons: “baseline accidentally includes the new mechanism.”
- It respects the MVP “parity control” goal without redesigning training loops. citeturn3view4  

#### Regression / interaction risks

- **Medium** only because it requires correctly locating gate modules in the model graph.
- Risk: if you reference the wrong path (e.g., the decoder layers are nested differently), your traversal could silently do nothing. Mitigate with an assertion that you found at least one `SlotwiseTemporalGate` when memory is enabled.

#### Verification plan for this fix

- Add a unit-ish test (can be static if runtime deps are heavy): instantiate the model via a config, enable baseline flag, and verify:
  - all `temporal_gate.enabled` are False
  - and that `MapTransformerLayer.forward` still runs (with `enabled=False` gate returns ones and identity values). citeturn3view1turn6view2  
- Add a log: in baseline mode print/record that gate is disabled.

#### Acceptance criteria

- In baseline mode, gate-forward is guaranteed disabled (or the run fails early).
- B1 vs B2 runs differ only by intended toggles (no accidental mixed mode).

---

### Fix C — P1 align gating with spec `q_cur` after BEV cross-attention

#### Problem statement

The spec is explicit:

- BEV-to-vector cross-attention produces `q_cur`
- the temporal gate must read `q_cur` and scale memory values **before** historical memory attention. citeturn12view1  

In the current decoder implementation, BEV cross-attn is computed as:

- `query_bev = self.attentions[attn_index](query, key, value, ...)`

but in the memory cross-attn branch and gating call, the code constructs `query_i` from `query`:

- `query_i = query[:, b_i].clone()`
- then calls the gate on `query_i[:, valid_track_idx]` citeturn6view2  

That means the gate is not clearly conditioned on the BEV-updated tensor (`query_bev`) the way the spec intends. citeturn12view1turn6view2  

#### Why this is truly a bug/gap

This is a **spec-contract mismatch** that changes the scientific interpretability:

- The gate is supposed to measure inconsistency between **current perception** (BEV-updated query) and memory.
- If it gates based on a pre-BEV query state, the gate is less “contradiction-aware” and more “query-internal,” which weakens the claim. citeturn12view1turn6view2  

#### Minimal viable fix

In `MapTransformerLayer.forward`, in the memory cross-attn branch (`attn_index == 2`):

1. Replace:
   - `query_i = query[:, b_i].clone()`
2. With:
   - `query_i = query_bev[:, b_i].clone()`

Then keep the existing reshape:
- `query_i = query_i[None, :]`

and use this `query_i` consistently for:
- the gate call
- the subsequent memory attention query tensor (`self.attentions[attn_index](query_i[:, valid_q_idx], ...)`) citeturn6view2  

This is narrowly local: it does not refactor the attention interface and preserves the pre-fusion structure: `query = query_memory + query_bev`. citeturn6view2turn3view4  

#### Why this fix is preferable to alternatives

- The smallest change that restores the spec-defined semantics without reorganizing the operation order or rewriting the transformer layer structure.
- The alternative (“make BEV cross-attn overwrite `query`”) is more invasive and risks altering baseline transformer behavior. citeturn6view2  

#### Regression / interaction risks

- **Medium**: you are changing what tensor the memory attention sees as “query.” That could affect training stability and performance (but it is the intended semantics).
- Watch for shape/grad issues: `query_bev` must exist on all code paths where memory cross-attn is executed.

#### Verification plan for this fix

- Static check: ensure the gate call references `query_bev` (or a variable explicitly named/defined as the BEV-updated query).
- Runtime check: assert that `query_bev.shape == query.shape` and that the memory attention call receives the same shape as before.
- Behavioral sanity: in a short run, verify gate alpha varies with BEV signals (nontrivial distribution) but clean performance does not collapse.

#### Acceptance criteria

- Gate and memory attention query are derived from BEV-updated tensor consistent with spec’s `q_cur`.
- No shape regressions; forward pass succeeds.

---

### Fix D — P1 remove implicit pose-corruption from stale substitution

#### Problem statement

In `VectorInstanceMemory.trans_memory_bank`, the code:

- constructs `clean_selected_mem_embeds` **after** pose propagation via `query_prop(...)`, and detaches it. citeturn5view0  
- then builds `corrupt_read_mem_embeds` by calling `_build_local_corrupted_read_view`, passing:
  - `clean_selected_mem_embeds` (propagated clean view)
  - and `self.mem_bank[:, b_i, active_mem_ids].detach()` as the canonical stale source pool citeturn5view0  

Inside `_build_local_corrupted_read_view`, the actual replacement assignment is:

- `mem_embeds_corrupt[pos, ins_idx] = canonical_mem_embeds[source_pos, ins_idx]` citeturn5view0  

This injects stale embeddings that are *not* pose-propagated the same way as the clean selected view. That effectively introduces out-of-scope pose mismatch corruption, even though MVP scope excluded explicit translation/rotation corruption. citeturn12view0turn3view2  

#### Why this is truly a bug/gap

This is a **correctness/spec-scope bug**:

- The spec explicitly narrows MVP to stale-bank substitution and excludes translation/rotation corruption modes. citeturn3view2turn12view0  
- By swapping in unpropagated canonical embeddings while the rest of the read view is propagated, corruption becomes a hybrid: “stale semantic content + pose inconsistency.” That can materially change what the gate learns, and it contaminates the claim “we trained against stale memory.” citeturn5view0turn12view0  

#### Minimal viable fix

Stay maximally local: modify `_build_local_corrupted_read_view` to use **the propagated clean selected view** as the stale source *when possible*.

Because the function already receives `mem_embeds_clean` and `all_select_indices`, you can do:

- Keep computing `source_pos = selected_indices[pos] - stale_offset` (canonical index).
- Instead of indexing `canonical_mem_embeds[source_pos, ins_idx]`, locate the **local slot** `src_local` where `selected_indices[src_local] == source_pos`.
  - If found: replace with `mem_embeds_clean[src_local, ins_idx]`
  - If not found: treat as “missing stale source → ineligible,” consistent with the current spec and existing missing-source logic. citeturn5view0turn12view0  

This preserves:
- read-path-only semantics (still modifying only the local clone)
- and the intended “stale offset” concept, without introducing pose corruption. citeturn5view0turn12view0  

#### Why this fix is preferable to alternatives

Alternative: propagate the canonical stale embedding by reconstructing the corresponding pose transform and calling `query_prop` again for each swapped slot. That is more invasive and error-prone.

Using the already-propagated clean pool is the smallest fix that stays entirely within MVP scope and uses existing computed tensors. citeturn5view0  

#### Regression / interaction risks

- **Medium**: in test-time selection mode, `source_pos` may not exist in the selected subset, reducing the fraction of corrupted slots (more ineligible slots).
  - This is an acceptable trade-off for MVP defensibility; you must not silently introduce pose corruption.
- Ensure tensor-type robustness: `selected_indices` can be a NumPy array in test-time selection; you’ll need a safe conversion to a torch tensor or a small helper to compare. citeturn5view0  

#### Verification plan for this fix

- Extend `tests/test_temporal_gate_mvp.py` with a targeted unit test:
  - Construct a `selected_indices` that is non-contiguous and ensure:
    - when source exists in selection: replacement equals `mem_embeds_clean[src_local]`
    - when source doesn’t exist: slot is ineligible (no corrupt mask set) citeturn10view0  
- Add an invariant check: corruption cannot change pose-propagated consistency, i.e., corrupted slots must come from the propagated pool, not canonical raw.

#### Acceptance criteria

- Corruption substitutions never pull unpropagated canonical embeddings into a propagated read view.
- Missing stale sources are consistently marked ineligible as before.

---

### Fix E — P1 add LayerNorm to gate feature inputs

#### Problem statement

The spec mandates LayerNorm on gate inputs (`u_cur`, `u_key`, `u_val`) to stabilize comparisons and prevent scale drift across layers. citeturn12view2turn12view1  

Current `SlotwiseTemporalGate.forward` computes cosine similarity and L2 and feeds raw concatenated embeddings (`q_expand`, `mem_expand`, plus scalar features) directly into the MLP, without any normalization. citeturn3view1  

#### Why this is truly a bug/gap

This is a **spec-contract mismatch** and a **stability risk**:

- Without LN, cosine/L2 and MLP inputs can vary wildly across layers and training steps, making the gate saturate or become non-informative.
- Spec explicitly calls LN part of MVP’s minimal stable design, not a “nice-to-have.” citeturn12view2turn12view1  

#### Minimal viable fix

In `SlotwiseTemporalGate.forward`:

- Apply `F.layer_norm` to `q_bt` and `mem_bt` **before** building `q_expand` and `mem_expand`.
- Use the normalized versions for:
  - cosine similarity
  - L2 distance
  - concatenation into `gate_inputs`

Example shape intent remains identical; no interface changes outside the gate module. citeturn3view1  

Do not add new features (like PE) yet as part of this fix; keep it strictly to LN stabilization.

#### Why this fix is preferable to alternatives

- It aligns with spec and is a single, localized change.
- It avoids altering the interface dimension (unlike adding PE), preserving current checkpoints’ parameter shapes—important for a minimization approach.

#### Regression / interaction risks

- **Medium**: gate behavior changes. Expect different alpha distributions; this is intended.
- Watch for dtype behavior: if mixed precision is used, LN must not produce NaNs; use torch’s standard LN.

#### Verification plan for this fix

- Update unit tests to assert:
  - `alpha` is finite (no NaN/Inf)
  - `values` is finite
- Add a runtime guard (optional): after computing `alpha_bqt`, assert `alpha_bqt.min() >= 0` and `alpha_bqt.max() <= 1`.

#### Acceptance criteria

- Gate forward produces finite outputs under typical random inputs.
- No change in tensor shapes and call-site contract.

---

### Fix F — P2 eliminate “Gap 2” ambiguity by enforcing the intended `q_len==1` contract

#### Problem statement

Your `SlotwiseTemporalGate` computes `alpha_qbt` as `[Q, B, T]` but then slices `alpha_bt = alpha_bqt[:, 0, :]` to produce a single per-slot scale for the `mem_embeds` values. citeturn3view1  

Repo status doc flags this as “approximated per-query-slot scaling.” citeturn3view4  

However, the actual decoder call path reshapes queries so that memory-branch attention runs with `q_len==1` (query length dimension is 1, and track queries are packed into the “batch” dimension). citeturn6view2  

The unit tests currently call the gate with `Q>1` and would misleadingly pass even though “per-query-slot scaling” is not meaningful for shared values in standard `MultiheadAttention`. citeturn10view0turn3view1  

#### Why this is truly a bug/gap

This is an **experimental-defensibility/documentation gap** more than a core mechanism bug:

- In the real call path (`q_len==1`), slicing `[:,0,:]` is not “query0 of many”; it is “the only query position,” so the scaling is exact for that packed formulation. citeturn6view2turn3view1  
- But because the gate is callable with arbitrary `Q`, the codebase is currently vulnerable to accidental misuse and misleading tests/docs.

#### Minimal viable fix

Choose the smallest, explicit contract that matches the implemented MVP:

1. In `SlotwiseTemporalGate.forward`, add:

- if `self.enabled` and `q_len != 1`: raise an assertion error explaining that the gate is implemented for the packed memory-branch path where `q_len==1`.

2. Update `tests/test_temporal_gate_mvp.py`:

- Replace `q = torch.randn(4, 2, 16)` with `q = torch.randn(1, 8, 16)` (or similar), aligning with real packed call path, and adjust alpha shape expectations accordingly. citeturn10view0  

3. Update `docs/codex_mvp_status.md` to remove or reframe Gap 2:
- Explicitly document that per-query-slot scaling is achieved by packing queries into the attention batch dimension, making `q_len==1` in the gate. citeturn3view4turn6view2  

#### Why this fix is preferable to alternatives

Alternative: implement full “fold Q×B into batch” generalization inside the gate and attention for arbitrary `Q`. That is a bigger refactor and is explicitly discouraged by your “no speculative expansion beyond MVP” constraint. citeturn3view2turn3view3  

The assertion-based contract is small, explicit, and prevents silent incorrect usage.

#### Regression / interaction risks

- **Low**: this only affects call sites that try to use the gate with `Q>1`. The actual decoder call path already uses `Q=1` for the memory branch. citeturn6view2  
- Unit tests will need adjustment, which is intended.

#### Verification plan for this fix

- Run unit tests to confirm:
  - gate tests now mirror real call path
  - gate raises for invalid shapes

#### Acceptance criteria

- Status doc no longer makes a misleading “approximation” claim without context.
- Gate cannot be silently misused in a way that would break the MVP premise.

---

### Fix G — P2 harden propagated/new/pad semantics with minimal explicit guards

#### Problem statement

Spec states mask semantics are “non-negotiable,” and requires explicit `propagated_mask/new_mask/pad_mask` semantics for eligibility and supervision. citeturn12view0turn12view1  

Current implementation uses:
- `valid_track_idx` (derived from memory entry lengths) and
- `query_key_padding_mask`
to build `eligible_mask` for gating. citeturn6view2turn3view4  

Repo status admits this is “only partial.” citeturn3view4  

#### Why this is truly a bug/gap

This is a **defensibility fragility**:

- Today’s behavior likely matches intended eligibility (track queries with real history and not padded).
- But eligibility is implicitly coupled to current memory-bank initialization semantics; subtle future changes can silently reclassify queries.

Even for MVP, the spec explicitly demands that these semantics be explicit to avoid silent metric invalidation. citeturn12view0turn3view4  

#### Minimal viable fix

Keep it minimal—do not rebuild query plumbing end-to-end. Implement the smallest explicit guard that prevents the worst failure mode:

1. **Expose the track-query boundary explicitly**:
   - In `_batchify_tracks`, you already compute `self.tracked_query_length[b_i] = lengths[b_i] - self.head.num_queries`. citeturn8view2turn16view0  
   - Store this into memory bank, e.g.:
     - `self.memory_bank.batch_tracked_query_len[b_i] = self.tracked_query_length[b_i]`

2. In `MapTransformerLayer.forward`, before using `valid_track_idx`:
   - Read `track_len = memory_bank.batch_tracked_query_len[b_i]`
   - Assert: `(valid_track_idx < track_len).all()`
   - This guarantees gating never touches “new queries appended this frame” (the non-track queries), even if future code changes accidentally expand memory mappings.

3. (Optional but still minimal) Use this boundary to define a minimal `propagated_mask` proxy:
   - `is_track_query = (index < track_len)`
   - Keep using `valid_track_idx` for “has history,” but enforce that “new” queries are never eligible.

This does not fully implement the spec’s propagated/new/pad tensor threading, but it materially strengthens defensibility with minimal surface area.

#### Why this fix is preferable to alternatives

- Threading `propagated_mask/new_mask` through the entire head→transformer call chain is more invasive and risks breaking the head’s query construction logic.
- The track-boundary guard directly addresses the most dangerous silent failure: new queries consuming memory. It also provides a clean invariant to cite in an audit. citeturn12view0turn6view2  

#### Regression / interaction risks

- **Low**: it’s an assertion + small metadata storage.
- Risk: if there are legitimate track query indices beyond `tracked_query_length` due to how queries are packed, you might trip the assertion. That would indicate your assumptions about query packing are wrong—and you should fix that before claiming MVP correctness.

#### Verification plan for this fix

- Add a targeted unit/invariant check:
  - Construct a fake `memory_bank.batch_tracked_query_len` and a fake `valid_track_idx` including an index ≥ track_len, and assert the transformer layer raises.
- In a short runtime log, print min/max of valid indices and track_len.

#### Acceptance criteria

- There is an explicit invariant preventing “new queries” from being gated.
- If query packing changes in the future, the run fails loudly instead of silently corrupting metrics.

---

### Fix H — P3 optional: add sinusoidal age encoding `pe_age` only if needed

#### Problem statement

Spec recommends sinusoidal encoding of `delta_t_int` (`pe_age`) as part of the gate feature inputs. citeturn12view4turn12view2  

Current implementation uses normalized `delta_t` scalar and `age_rank_norm`, but no sinusoidal encoding. citeturn3view1turn5view0  

#### Why this is a gap

This is **spec alignment / modeling-choice gap**, not a hard correctness bug. If LN is added and training remains stable and learns meaningful alpha patterns, you can defensibly defer `pe_age` as an enhancement.

#### Minimal viable fix (if you decide it is needed)

- Add a small helper (inside `SlotwiseTemporalGate`) to encode `delta_t_int` into `d_pe=8` sin/cos features.
- Update the first linear layer input dim accordingly.

#### Regression / interaction risks

- **Medium**: changes parameter shapes; breaks checkpoint compatibility.
- Makes debugging harder unless you lock it down with unit tests.

#### Verification plan

- Only after P0/P1 fixes: validate alpha distributions and learning curves; add a unit test for PE shape and finiteness.

#### Acceptance criteria

- Gate remains stable and trainable; no NaNs; alpha does not saturate immediately.

## Collective consistency check

After applying Fix A–G together, the combined repaired system should satisfy the MVP story without introducing new scope creep:

- **MVP scope preserved**: You are not adding multi-head gating, additive logits, BEV reliability, or new corruption families. All proposed fixes are either:
  - strict correctness (Fix A),
  - spec-aligned wiring (Fix C, E),
  - spec-aligned corruption semantics (Fix D),
  - parity guardrails (Fix B),
  - or explicit invariants to prevent silent drift (Fix F, G). citeturn3view2turn3view3turn12view0  

- **No fix conflicts**:
  - Fix C (use `query_bev` in memory branch) and Fix E (LN) interact positively: the gate becomes better-conditioned and uses intended signal.
  - Fix D (propagated stale sources) interacts positively with Fix C/E by ensuring the gate isn’t actually learning pose mismatch artifacts.
  - Fix B (baseline disables forward gate) remains correct regardless of Fix C/E/D.
  - Fix F (q_len==1 assertion) is satisfied by the existing packed call path (`query_i = query_i[None, :]`) and will prevent future misuse. citeturn6view2turn3view1  

- **B1 vs B2 parity assumptions improved**:
  - Fix B prevents a critical “half-on baseline” failure mode.
  - Fix A ensures both B1 and B2 have a consistent corruption context across frames, eliminating a clip-level confound. citeturn3view0turn8view0  

- **Pre-fusion, slotwise, value-scaling-only story preserved**:
  - You are not touching the fact that values are scaled (`mem_values`) while keys remain `mem_embeds`. citeturn6view2turn3view1  
  - The insertion point remains in memory cross-attn and still fuses via `query = query_memory + query_bev`. citeturn6view2turn3view4  

- **Read-path-only corruption semantics preserved**:
  - Fix D keeps corruption read-only; it only changes the source of substitution from “raw canonical” to “propagated clean selected,” which is still read-view and still does not overwrite the canonical bank. citeturn5view0turn10view0  

New likely failure modes introduced by these fixes (and how to avoid them):

- Fix C could reveal that `query_bev` is not always defined or not shape-aligned in some operation_order variants. Avoid by adding lightweight assertions for shape and existence in the memory branch. citeturn6view2  
- Fix D may reduce corruption coverage in test-time selection. Avoid by explicitly logging “eligible corrupted slots fraction” so you can see coverage; do not silently assume corruption fraction is unchanged.

## Recommended implementation order

1. **Fix A (P0)** — unblock runtime and restore correct clip-level corruption semantics. This is non-negotiable: nothing else is meaningful if training crashes or corruption metas are wrong. citeturn3view0  
2. **Fix B (P1)** — enforce baseline semantics early so you can trust any ablation runs you do after subsequent behavioral changes. citeturn3view4turn8view0  
3. **Fix C (P1)** — bring the gating mechanism onto the correct `q_cur` input per spec; this is central to the interpretability story. citeturn12view1turn6view2  
4. **Fix D (P1)** — remove implicit pose corruption so “stale substitution” means what you claim it means. citeturn5view0turn12view0  
5. **Fix E (P1)** — stabilize gate features and align with spec LN requirement; do this after the wiring fixes so you’re not diagnosing two failures at once. citeturn12view2turn3view1  
6. **Fix F (P2)** — lock down the q_len==1 contract and update tests/docs to eliminate confusion and future misuse. citeturn6view2turn10view0turn3view4  
7. **Fix G (P2)** — add explicit guardrails around query masks/track boundaries to close defensibility gaps without a full mask-refactor. citeturn12view0turn3view4turn6view2  
8. **Fix H (P3)** — only if needed after observing training behavior.

## Final narrowed fix set

### Must fix before claiming the MVP works

- Fix A (P0): corruption meta injection order bug in `forward_train`. citeturn3view0  
- Fix C (P1): gate and memory attention must use BEV-updated `q_cur` (`query_bev`) as spec requires. citeturn12view1turn6view2  
- Fix D (P1): stale substitution must not introduce implicit pose corruption; substitute from propagated selected pool or mark ineligible. citeturn5view0turn12view0  

### Should fix for defensibility

- Fix B (P1): enforce true no-gate forward semantics in baseline mode (not just loss disabling). citeturn3view4turn8view0  
- Fix E (P1): add LayerNorm to gate feature inputs, per spec and for stability. citeturn12view2turn3view1  
- Fix F (P2): assert intended `q_len==1` contract and update unit tests/docs to remove misleading ambiguity. citeturn6view2turn10view0turn3view4  
- Fix G (P2): add explicit guardrails around new/pad semantics via track-query boundary assertions/metadata. citeturn12view0turn6view2turn3view4  

### Can defer without invalidating the MVP

- Fix H (P3): sinusoidal `pe_age` encoding—implement only if gate behavior is unstable or fails to learn meaningful age dependence after LN and wiring are corrected. citeturn12view4turn12view2