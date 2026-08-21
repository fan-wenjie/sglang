"""What using an early key in the query coefficient costs the model, in bits per byte.

The query coefficient `q~ = q - beta (k.q) k` is an identity: it reproduces the two-reading form
at 8.5e-08. Forming it from the SHIFTED residual instead -- an early key and an early write
strength -- is not. It buys the whole round trip hiding inside the previous feed-forward, and it
changes what the model computes, so it has to be paid for in quality rather than argued about.

Three arms, and the middle one is the point:

    exact       q~ and the state update both from x_l                     the model
    mixed       q~ from h_(l-1); the state update from x_l                the proposal
    all early   q~ and the state update both from h_(l-1)                 the control

Mixed against all-early is what says whether keeping the UPDATE exact is worth the second key
projection. The reasoning behind the split is that an approximation in the output is a per-step
error while one in the state compounds -- this measures whether that reasoning holds.

Bits per byte, not perplexity: it divides by UTF-8 bytes rather than tokens, so the three arms are
comparable without the tokenizer entering. Teacher-forced, one position at a time through the
recurrence, because the approximation is defined on the recurrent step and a chunked prefill kernel
would not contain it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time

import torch


def bits_per_byte(nll_nats: float, n_bytes: int) -> float:
    return nll_nats / math.log(2) / max(n_bytes, 1)


def load_docs(n: int, min_chars: int):
    """Documents from the fineweb-edu sample already on this box."""
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train",
                      streaming=True)
    out = []
    for row in ds:
        text = row["text"]
        if len(text) >= min_chars:
            out.append(text[:min_chars])
        if len(out) >= n:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--docs", type=int, required=True)
    ap.add_argument("--chars", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="exact,mixed,all_early")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    print("  loading", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    print("  loaded", flush=True)

    docs = load_docs(a.docs, a.chars)
    print(f"  {len(docs)} document(s), {sum(len(d.encode()) for d in docs)} bytes", flush=True)

    try:
        from sglang.srt.afd.early_k_arms import install_arm
    except ImportError:
        print(
            "  early_k_arms is not written yet, and writing it carelessly is how this measurement\n"
            "  goes wrong. The three arms must ALL run through one recurrent implementation, and\n"
            "  that implementation must first be shown to reproduce the stock forward's logits.\n"
            "  Using the stock forward as the `exact` arm and a hand-written recurrence for the\n"
            "  other two puts the approximation and the reimplementation's own error into the same\n"
            "  difference, with no way to tell them apart -- which is the shape of every wrong\n"
            "  number in this arrangement's history.\n"
            "\n"
            "  What it has to do:\n"
            "    1. replace Qwen3NextGatedDeltaNet.forward with a recurrent step, looped over\n"
            "       positions, because the chunked prefill kernel does not contain the step the\n"
            "       approximation is defined on\n"
            "    2. capture h_(l-1) -- the residual after the previous layer's attention and before\n"
            "       its feed-forward -- with a hook, which is the same value the read point uses\n"
            "    3. project a second key and write strength from it for the `mixed` and\n"
            "       `all_early` arms\n"
            "    4. assert the `exact` arm's logits match the unpatched model before any arm is\n"
            "       reported"
        )
        return 2

    results = {}
    for arm in a.arms.split(","):
        undo = install_arm(model, arm)
        nll, n_bytes, n_tok = 0.0, 0, 0
        began = time.perf_counter()
        for i, text in enumerate(docs):
            ids = tok(text, return_tensors="pt").input_ids.cuda()
            # ONE TOKEN AT A TIME. A whole-sequence forward takes the chunked prefill kernel,
            # which does not contain the recurrent step the arms patch -- every arm would then
            # report the same number and the measurement would look like a null result.
            past, logits = None, []
            for i in range(ids.shape[1]):
                with torch.no_grad():
                    out = model(ids[:, i : i + 1], past_key_values=past, use_cache=True)
                past = out.past_key_values
                logits.append(out.logits[:, -1].float())
            lp = torch.log_softmax(torch.cat(logits, 0)[:-1], dim=-1)
            nll += -lp.gather(1, ids[0, 1:, None]).sum().item()
            n_bytes += len(text.encode("utf-8"))
            n_tok += ids.shape[1] - 1
            print(f"    {arm} {i + 1}/{len(docs)}  bpb so far "
                  f"{bits_per_byte(nll, n_bytes):.4f}", flush=True)
        undo()
        results[arm] = {"bpb": bits_per_byte(nll, n_bytes), "nll": nll,
                        "bytes": n_bytes, "tokens": n_tok,
                        "seconds": time.perf_counter() - began}
        print(f"  {arm}: bpb {results[arm]['bpb']:.4f}", flush=True)

    if "exact" in results:
        base = results["exact"]["bpb"]
        for arm, r in results.items():
            r["delta_bpb"] = r["bpb"] - base
            r["percent"] = 100 * (r["bpb"] - base) / base
    json.dump(results, open(a.out, "w"), indent=2)
    print("\n    " + f"{'arm':12} {'bpb':>9} {'delta':>9} {'percent':>9}")
    for arm, r in results.items():
        print(f"    {arm:12} {r['bpb']:9.4f} {r.get('delta_bpb', 0):9.4f} "
              f"{r.get('percent', 0):8.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
