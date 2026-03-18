# MapTracker v1 — Comprehensive Code Review

## Summary

Systematic review of the MapTracker v1 codebase covering correctness, numerical
stability, memory/state management, train/inference divergence, and data flow.
Seven new issues identified (excluding the two pre-known bugs in `seg_loss.py`
and `vector_memory.py`).

---

## Issue 1 — Hardcoded `history_steps=4` in BEVFormerEncoder TemporalNet

**File:** `plugin/models/backbones/bevformer/encoder.py`, line 46
**Severity:** High
**Category:** CORRECTNESS — shape mismatch / crash

### Description

`BEVFormerEncoder.__init__` instantiates each `TemporalNet` with a hardcoded
`history_steps=4`. The `TemporalNet.conv_in` layer expects
`(history_steps + 1) * hidden_dims` input channels. Meanwhile, the backbone
(`BEVFormerBackbone`) pads the warped history buffer to `self.history_steps`
entries (a *configurable* parameter). If the config sets `history_steps` to
anything other than 4, the channel count fed to `conv_in` will not match and
the forward pass will crash with a shape-mismatch error.

### Proposed Fix

```diff
--- a/plugin/models/backbones/bevformer/encoder.py
+++ b/plugin/models/backbones/bevformer/encoder.py
@@ -35,7 +35,7 @@

     def __init__(self, *args, pc_range=None, num_points_in_pillar=4, return_intermediate=False, dataset_type='nuscenes',
-                 **kwargs):
+                 history_steps=4, **kwargs):

         super(BEVFormerEncoder, self).__init__(*args, **kwargs)
         self.return_intermediate = return_intermediate
@@ -43,7 +43,7 @@
         temporal_mem_layers = []
         for _ in range(self.num_layers):
-            mem_conv = TemporalNet(history_steps=4, hidden_dims=self.embed_dims, num_blocks=1)
+            mem_conv = TemporalNet(history_steps=history_steps, hidden_dims=self.embed_dims, num_blocks=1)
             temporal_mem_layers.append(mem_conv)
         self.temporal_mem_layers = nn.ModuleList(temporal_mem_layers)
```

The corresponding transformer config must then pass `history_steps` through to
the encoder so the value stays consistent with the backbone.

---

## Issue 2 — Numerical Instability in Gate-Supervision BCE Loss

**File:** `plugin/models/mapers/MapTracker.py`, lines 474, 481, 489
**Severity:** Medium
**Category:** NUMERICAL STABILITY — NaN/inf risk

### Description

`_compute_gate_supervision` calls `F.binary_cross_entropy` directly on the
sigmoid-output gate values `alpha`:

```python
close_terms.append(F.binary_cross_entropy(a, torch.zeros_like(a)))  # L474
open_terms.append(F.binary_cross_entropy(p, torch.ones_like(p)))    # L481
clean_terms.append(F.binary_cross_entropy(c, torch.ones_like(c)))   # L489
```

`alpha` values come from `torch.sigmoid(...)` and are theoretically in (0, 1),
but in float32 arithmetic sigmoid can saturate to exactly 0.0 or 1.0 for
extreme inputs. `F.binary_cross_entropy` computes `log(input)` and
`log(1 - input)` without clamping, producing `-inf` → NaN gradients when the
input hits the boundary.

### Proposed Fix

Replace the raw `F.binary_cross_entropy` calls with a clamped version:

```diff
--- a/plugin/models/mapers/MapTracker.py
+++ b/plugin/models/mapers/MapTracker.py
@@ -470,17 +470,20 @@
+            _eps = 1e-6
+
             if affected.any():
                 affected_non_empty += 1
                 a = alpha_valid[affected]
                 affected_vals.append(a)
-                close_terms.append(F.binary_cross_entropy(a, torch.zeros_like(a)))
+                close_terms.append(F.binary_cross_entropy(a.clamp(_eps, 1 - _eps), torch.zeros_like(a)))

             recency_ref = age_rank.max(dim=1, keepdim=True).values
             preserved_recent = valid_mem & (~corrupt) & (age_rank >= recency_ref)
             if preserved_recent.any():
                 p = alpha_valid[preserved_recent]
                 preserved_vals.append(p)
-                open_terms.append(F.binary_cross_entropy(p, torch.ones_like(p)))
+                open_terms.append(F.binary_cross_entropy(p.clamp(_eps, 1 - _eps), torch.ones_like(p)))

             if str(mode).lower() == 'clean':
                 clean_recent = valid_mem & (age_rank >= recency_ref)
@@ -488,7 +491,7 @@
                     c = alpha_valid[clean_recent]
                     clean_vals.append(c)
                     if self.enable_clean_open_loss:
-                        clean_terms.append(F.binary_cross_entropy(c, torch.ones_like(c)))
+                        clean_terms.append(F.binary_cross_entropy(c.clamp(_eps, 1 - _eps), torch.ones_like(c)))
```

---

## Issue 3 — Variable Shadowing of Decoder Layer Index in `forward_train`

**File:** `plugin/models/heads/MapDetectorHead.py`, lines 244 and 253
**Severity:** Medium
**Category:** CORRECTNESS — latent logic bug

### Description

The outer loop enumerates decoder layers with variable `i`:

```python
for i, (queries) in enumerate(inter_queries):   # i = decoder layer index
    ...
    scores = self.cls_branches[i](queries)       # correct use of layer index
    ...
    for i in range(len(scores)):                 # SHADOWS outer i
        reg_points_list.append(reg_points[i])
        scores_list.append(scores[i])
```

After the inner loop, `i` holds the last batch index rather than the decoder
layer index. This is not currently a correctness bug (the layer index is
consumed before the shadow), but any future code added after the inner loop
that references `i` expecting the layer index will silently use the wrong
value.

### Proposed Fix

```diff
--- a/plugin/models/heads/MapDetectorHead.py
+++ b/plugin/models/heads/MapDetectorHead.py
@@ -250,7 +250,7 @@
             reg_points_list = []
             scores_list = []
-            for i in range(len(scores)):
+            for b_i in range(len(scores)):
                 # padding queries should not be output
-                reg_points_list.append(reg_points[i])
-                scores_list.append(scores[i])
+                reg_points_list.append(reg_points[b_i])
+                scores_list.append(scores[b_i])
```

---

## Issue 4 — `_denorm_lines` / `_norm_lines` Silently Mutate Input Tensors

**File:** `plugin/models/mapers/MapTracker.py`, lines 1182–1196
**Severity:** Medium
**Category:** MEMORY & STATE — in-place op on potentially shared tensor

### Description

Both helper functions modify the coordinate tensor **in-place**:

```python
def _denorm_lines(self, line_pts):
    line_pts[..., 0] = line_pts[..., 0] * self.roi_size[0] - self.roi_size[0] / 2
    line_pts[..., 1] = line_pts[..., 1] * self.roi_size[1] - self.roi_size[1] / 2
    return line_pts
```

Current call sites happen to pass freshly-cloned or freshly-created tensors,
so this doesn't corrupt data today. However, the pattern is fragile: any future
caller that passes a view or a tensor that is used elsewhere will get silent
data corruption. This is especially dangerous because `rearrange` sometimes
returns a view rather than a copy, and several call sites pass the result of
`rearrange` directly.

### Proposed Fix

Create new tensors instead of mutating:

```diff
--- a/plugin/models/mapers/MapTracker.py
+++ b/plugin/models/mapers/MapTracker.py
@@ -1182,14 +1182,16 @@
     def _denorm_lines(self, line_pts):
         """from (0,1) to the BEV space in meters"""
-        line_pts[..., 0] = line_pts[..., 0] * self.roi_size[0] \
-                        - self.roi_size[0] / 2
-        line_pts[..., 1] = line_pts[..., 1] * self.roi_size[1] \
-                        - self.roi_size[1] / 2
-        return line_pts
+        out = line_pts.clone()
+        out[..., 0] = line_pts[..., 0] * self.roi_size[0] - self.roi_size[0] / 2
+        out[..., 1] = line_pts[..., 1] * self.roi_size[1] - self.roi_size[1] / 2
+        return out

     def _norm_lines(self, line_pts):
         """from the BEV space in meters to (0,1) """
-        line_pts[..., 0] = (line_pts[..., 0] + self.roi_size[0] / 2) \
-                                        / self.roi_size[0]
-        line_pts[..., 1] = (line_pts[..., 1] + self.roi_size[1] / 2) \
-                                        / self.roi_size[1]
-        return line_pts
+        out = line_pts.clone()
+        out[..., 0] = (line_pts[..., 0] + self.roi_size[0] / 2) / self.roi_size[0]
+        out[..., 1] = (line_pts[..., 1] + self.roi_size[1] / 2) / self.roi_size[1]
+        return out
```

---

## Issue 5 — `BEVFormerEncoder.forward` Crashes When `warped_history_bev` Is None

**File:** `plugin/models/backbones/bevformer/encoder.py`, line 243
**Severity:** Medium
**Category:** CORRECTNESS — unconditional attribute access on nullable value

### Description

Inside the encoder's layer loop, the temporal memory fusion unconditionally
accesses `warped_history_bev.shape[3]`:

```python
curr_feat = rearrange(output, 'b (h w) c -> b c h w', h=warped_history_bev.shape[3])
fused_output = mem_layer(warped_history_bev, curr_feat)
```

The `forward` signature declares `warped_history_bev=None` as a valid default.
While the current backbone caller always pads this to a tensor, the encoder
itself is not defensive, so any standalone or future use with `None` causes an
`AttributeError` crash.

### Proposed Fix

```diff
--- a/plugin/models/backbones/bevformer/encoder.py
+++ b/plugin/models/backbones/bevformer/encoder.py
@@ -240,10 +240,11 @@

-            # BEV memory fusion layer
-            mem_layer = self.temporal_mem_layers[lid]
-            curr_feat = rearrange(output, 'b (h w) c -> b c h w', h=warped_history_bev.shape[3])
-            fused_output = mem_layer(warped_history_bev, curr_feat)
-            fused_output = rearrange(fused_output, 'b c h w -> b (h w) c')
-            output = output + fused_output
+            # BEV memory fusion layer (requires warped history features)
+            if warped_history_bev is not None:
+                mem_layer = self.temporal_mem_layers[lid]
+                curr_feat = rearrange(output, 'b (h w) c -> b c h w', h=warped_history_bev.shape[3])
+                fused_output = mem_layer(warped_history_bev, curr_feat)
+                fused_output = rearrange(fused_output, 'b c h w -> b (h w) c')
+                output = output + fused_output
```

---

## Issue 6 — `temporal_propagate` Returns `None` on Non-Loss Path

**File:** `plugin/models/mapers/MapTracker.py`, line 301
**Severity:** Low
**Category:** CORRECTNESS — implicit None return

### Description

When `get_trans_loss=False`, the function has no explicit `return` statement and
implicitly returns `None`. While callers currently only capture the return value
when `get_trans_loss=True`, this violates the principle of explicit returns and
will silently produce a `None` if a caller ever forgets to check the flag.

### Proposed Fix

```diff
--- a/plugin/models/mapers/MapTracker.py
+++ b/plugin/models/mapers/MapTracker.py
@@ -300,6 +300,7 @@
             }
             return trans_loss_dict
+        return None
```

---

## Issue 7 — `vector_eval.py` Hardcodes `mAP_normal` Divisor as 9

**File:** `plugin/datasets/evaluation/vector_eval.py`, line 294
**Severity:** Low
**Category:** CORRECTNESS — fragile hardcoded constant

### Description

```python
mAP_normal = mAP_normal / 9
```

This assumes exactly 3 categories × 3 thresholds = 9. If additional map
element categories are added (e.g., stop lines) or different threshold sets
are configured (the code already supports variable thresholds at lines 40–42),
this divisor becomes silently wrong and the reported `mAP_normal` metric will
be incorrect.

### Proposed Fix

```diff
--- a/plugin/datasets/evaluation/vector_eval.py
+++ b/plugin/datasets/evaluation/vector_eval.py
@@ -291,7 +291,7 @@
         for label in self.id2cat.keys():
             for thr in self.thresholds:
                 mAP_normal += result_dict[self.id2cat[label]][f'AP@{thr}']
-        mAP_normal = mAP_normal / 9
+        mAP_normal = mAP_normal / (len(self.id2cat) * len(self.thresholds))
```
