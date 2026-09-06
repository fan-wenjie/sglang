# VestigeKV port into standard sglang (production fork)

Training-free NoPE-MLA KV-cache compression, ported as an **attention-backend
wrapper**. No model code changes; the whole change is confined to the attention
layer, kill-switchable to byte-identical stock.

## Landing zone (mapped 2026-09-05, standard sglang)

- Kimi Linear's MLA is `DeepseekV2AttentionMLA`
  (`models/kimi_linear.py:48` -> `models/deepseek_v2.py`), imported with
  `skip_rope=True` = NoPE. VestigeKV's precondition (NoPE) holds natively.
- Kimi Linear is a hybrid: `KimiDecoderLayer` picks `self_attn` = KDA
  (`KimiDeltaAttention`) or MLA (`KimiMLAAttention`) per layer
  (`kimi_linear.py:565-585`).
- At runtime the MLA (full-attn) and KDA (linear) sub-backends are composed by
  `HybridLinearAttnBackend(full_attn_backend, linear_attn_backend,
  full_attn_layers)` (`layers/attention/hybrid_linear_attn_backend.py:1031`),
  which routes per layer via `_is_full_attn(layer_id)`.
- That hybrid is built in `attn_backend_wrapper(runner, full_attn_backend)`
  (`layers/attention/attention_registry.py:343`), the single place special/hybrid
  models wrap the full-attn backend.

## The seam (one injection point)

Inside `attn_backend_wrapper`, when VestigeKV is enabled (server flag) and the
model is Kimi Linear, wrap the MLA backend BEFORE it enters the hybrid:

    full_attn_backend = VestigeMLABackend(full_attn_backend, runner)   # <- added
    ... HybridLinearAttnBackend(full_attn_backend, linear_attn_backend, layers)

Then MLA-layer decodes flow: hybrid -> VestigeMLABackend -> base MLA backend.
KDA layers are untouched. Prefill is untouched (VestigeKV only compresses the
MLA latent cache for decode).

## VestigeMLABackend(AttentionBackend) contract

`attn_backend_list = [base]`; all of these DELEGATE to base:
  init_forward_metadata / _out_graph / _in_graph, forward_extend,
  get_cuda_graph_seq_len_fill_value, data_type, kv_cache_dtype,
  init_cuda_graph_state, and the cuda-graph capture/replay hooks.

Intercepted:
  - end of prefill (last MLA extend or a metadata hook): build the tier-2 index
    from the just-written latent rows (sidecar residual sigma over the 64-dim
    decoupled branch -> eviction set; exact 64-dim summand + rank-r sketch +
    per-row certificate -> recall index). Self-calibrated z + entropy gate.
  - `forward_decode(q, k, v, layer, forward_batch, ...)`: if disabled -> pure
    `base.forward_decode(...)` (kill-switch, byte-identical). If enabled and this
    is an MLA layer -> restrict the attended latent set to the kept rows + the
    per-step recalled rows (topj-capped bounded fetch, default 16, -1 = uncapped),
    then call base.forward_decode over that set.

## Reuse (fork-not-rewrite)

Port the VALIDATED core from the mini-sglang submodule verbatim where possible:
`minisgl/kimi/tier2.py` (RecallTier), `minisgl/kimi/policy.py` (VestigePolicy).
These are pure torch over latent rows; only the pool-read/index adaptation is
sglang-specific.

## Server args (to add)

`--enable-vestigekv` (default off), `--vestigekv-rho` (default 1/32),
`--vestigekv-topj` (default 16, -1 = uncapped bounded-fetch opt-out; documented),
`--vestigekv-index-rank` (r, default 64). All required knobs recorded per run
(no silent partial defaults).

## OPEN (pending interface map, then implement)

Exact `forward_decode` signature + how the base MLA backend reads latent rows
from the token-to-KV (MLA) pool, and how to restrict/augment the attended set
per request (modify kv_indices / req_to_token, or supply a compressed view).
This determines the forward_decode body. Validate parity (kill-switch ==
stock, bitwise) + a needle recovery run against the same Kimi checkpoint the
mini-sglang stack was validated on, once the GPU frees (capqual holds it now).

## Validation status (2026-09-05, RTX PRO 6000 Blackwell / SM120, single card)

### DONE — core validated by bitwise equivalence (GPU)
`test/manual/test_vestige_equiv.py` passes on this GPU: the vendored
sglang core (`recall_tier.py`, `eviction.py`) is BITWISE-IDENTICAL to the
mini-sglang reference (which is validated end-to-end to 512k, PREREG19/31/32):
  - eviction: sidecar sigma bitwise-equal, keep-mask equal;
  - recall: build stats identical (zp, n_hard, arch), 16/16 query steps identical.
This transfers the reference's end-to-end validation to the port's DECISION LOGIC.

### DONE — structural integration (SM120)
Whole sglang import chain loads on SM120; `VestigeMLABackend` is a valid
`AttentionBackend`; `vestige_mla` registers; the Kimi hybrid auto-adopts it.

### BLOCKED — end-to-end serving of full-precision Kimi-48B on ONE 96GB card
Stock sglang was brought to server-ready after this fix chain (all applied here):
  1. `flashinfer` MLA JIT fails on SM120 (`check_cuda_arch`, needs CUDA>=12.9) ->
     use `--attention-backend triton` (Triton JITs for the actual arch).
  2. `sgl_kernel` JIT: `cuda_runtime.h not found` -> set `CUDA_HOME=/usr/local/cuda`
     (nvcc 12.8 DOES support compute_120; it compiles sm120a fine once headers found).
  3. cpu_offload `functional_call` rejects Kimi's tied KDA `A_log` ->
     `offloader.py` patched: `tie_weights=False`.
  4. cpu_offload sends the tiny `gate.e_score_correction_bias` (captured by
     reference in TopK) to CPU -> a CPU tensor hits the `moe_fused_gate` Triton
     kernel -> `offloader.py` patched: skip params < 1MB from offload.
Then the wall: the model weights are ~92GB; on a 96GB card cpu_offload must
stream experts, and the MoE forward HOLDS ~51GB of streamed expert weight during
a single 26-layer forward (not reclaimable via empty_cache -> held references),
so it OOMs (~93GB used) even at cpu_offload_gb=50 / mem_fraction_static=0.55.
This is the same model-size-vs-single-card limit that made the native reference
stack use the dual-machine (local + remote 5090) pipeline. NOT a VestigeKV issue.

### Path to end-to-end serving validation (suitable hardware)
Any ONE of: (a) a card/allocation with >~110GB so no expert offload is needed;
(b) multi-GPU tensor/pipeline parallel (the native stack's dual-machine split);
(c) flashinfer built for SM120 with CUDA>=12.9 toolkit (removes the triton
detour) plus (a)/(b) for memory. Then: run with `--attention-backend vestige_mla`,
first `SGLANG_VESTIGE_ENABLED=0` for the kill-switch parity (must equal stock
flashinfer/triton bit-for-bit), then enabled for needle recovery. The two
`forward` bodies (`_build_tier2`, `_compressed_decode`) remain to be written
against a live ForwardBatch (they need prefill-query capture in `forward_extend`;
the pool-read/index seam is mapped above).

## Upstream-readiness status (2026-09-06)

- Guards added for upstream safety: NoPE-model allow-list (RoPE-MLA refused
  with the measured collapse cited), --page-size 1 required, speculative
  decoding refused (unwired).
- Tier-2 recall: vendored + equivalence-tested, NOT wired into serving decode
  (kept + tail only). An upstream feature PR either wires it or excludes it.
- Benchmark arm switch is env-configured: SGLANG_TEST_VESTIGE_FULL_ARM_FLAG
  names the flag file (unset by default = no switching, zero production
  cost); bench launch scripts must export it or the A/B silently compares
  VESTIGE against itself.
- Standalone bugfix split out on branch fix-set-mla-kv-buffer-noncontiguous
  (based on upstream main) for a separate PR.

## Change audit: attention-confined vs. stock-file changes

Everything vs. upstream base `f1f2380`, grouped per the porting rule ("confine
changes to attention; list anything else separately for audit").

### Attention-confined (the port itself; 7 files, all under `layers/attention/`)
| file | lines | what |
|---|---|---|
| `vestige_mla_backend.py` | +435 | the wrapper backend (all VestigeKV logic) |
| `vestige/{__init__,eviction,recall_tier}.py` | +203 | vendored core, bit-identical to mini-sglang (verified by test) |
| `test/manual/test_vestige_equiv.py` | +104 | GPU equivalence test vs. the mini-sglang reference (manual: needs the reference checkout) |
| `test/registered/unit/layers/attention/test_vestige_mla_backend.py` | +190 | CI unit tests: graph padding, slot reuse, row-invariant check semantics |
| `attention_registry.py` | +17 | register `vestige_mla` factory (additive) |

### Stock-file changes OUTSIDE attention (3 files, 13 lines -- the audit list)
| file | lines | why | risk |
|---|---|---|---|
| `kernels/ops/kvcache/set_mla_kv_buffer.py` | 8 | bugfix: `.view()` -> `.reshape()`; non-contiguous MLA latent slice at long chunked prefill crashes stock sglang too (upstream-worthy) | none: reshape is a no-op on the contiguous path |
| `srt/environ.py` | +4 | register `SGLANG_ENABLE_VESTIGE` EnvBool (required by env-var conventions; registry entry only) | none: pure registration |
| `srt/server_args.py` | +1 | add `"vestige_mla"` to ATTENTION_BACKEND_CHOICES | none: list entry |

No other stock file is touched. The gloo->NCCL metadata experiment was reverted
and is NOT in the tree (metadata stays on gloo by design; see nope_kv README).

## REQUIRED completion: wire the recall tier (rule: recall is a necessary part of the paper claims, the algorithm, and the engineering; NOT closeable; tier-1-only is an incomplete state, never a supported configuration)

The paper's deployment spec already declares tier-2 mandatory (the partition
deletes nothing); this port currently serves the tier-1 floor. Wiring plan:

1. Prefill: save the last-n expanded queries per (slot, MLA layer) for
   calibration; run RecallTier.build at prefill end (off the decode path).
2. Decode, IN-GRAPH under the cap: per MLA layer, scan = q @ [side|csk]
   (GEMV, fixed shape), add z*certificate, topk(topj) -> write topj*H fetch
   slots into a fixed-address buffer (pad row for unfired lanes); run the
   stock decode kernel over the fetch partition; LSE-merge with the main
   partition's output.
3. STRUCTURAL FACT this design rests on: the fetch cap makes recall
   fixed-shape and therefore graph-capturable end to end; uncapped recall is
   variable-shape and can never be captured. Under graph serving the cap is
   a hard requirement, not a recommendation.
4. Memory: GPU-resident index adds (64+r) fp32 per archived row (~45% of
   the bf16 row) -- re-size the KV pool accordingly.
5. Re-measure both speedup axes with recall on; quality claims then extend
   to the 128x recall-tier operating point in production.
