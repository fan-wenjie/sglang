"""The same comparison under deterministic decoding, with a null that still works.

At temperature 0 the previous null collapses: a model sampled twice gives the same text, so
"stock against a second sample of itself" is the identity and reads 1.0. But greedy decoding in
this server is not deterministic across BATCH COMPOSITION -- four prompts decoded together and the
same four decoded apart part company on three of them, because the kernels tile differently and
bfloat16 ties fall the other way. That is the noise floor for greedy, and it is measurable: run
the stock model twice over the same prompts in two different request orders.

    stock      shift 0, greedy, prompts in corpus order
    stock'     shift 0, greedy, prompts shuffled -- same model, same prompts, different batching
    shifted    shift 1, greedy, corpus order

Two instruments, because MAUVE is not the sharp one here. Greedy text is near-degenerate and both
arms will score low against human text; what separates them is measured directly:

    exact      what fraction of the 500 continuations are byte-identical
    divergence how many tokens the two agree on before parting
    MAUVE      the distributional statement, reported beside its own null as before

Measured, Qwen3.8-27B-FP8 shift 1 coverage all, 500 prompts, 128 greedy tokens:

    stock vs stock reordered   110/500 identical (22.0%), median 25 shared tokens
    stock vs shifted            10/500 identical ( 2.0%), median  3 shared tokens

    MAUVE stock vs human               0.2685
    MAUVE shifted vs human             0.1929
    MAUVE stock vs stock reordered     0.9844   (the null)
    MAUVE stock vs shifted             0.9469

The shift costs 0.0757 MAUVE and reordering the same model costs 0.0156, so the effect is 4.86x
the noise floor. Under SAMPLING the same comparison was 0.84x its noise floor and pointed the
other way.

That is not a contradiction and it is the point of running both. Greedy decoding is sensitive to
the argmax, so any reordering of the top two logits changes the token and compounds from there;
sampling at temperature 1.0 buries the same difference under its own randomness. The read point's
2.7% likelihood cost is visible where the decode rule reads the argmax and hidden where it does
not.

Read the agreement statistic before the MAUVE one. MAUVE was designed for sampled text and both
greedy arms score low against human text (0.27 and 0.19 against 0.72 and 0.76 sampled), so it is
being used outside its regime; the median-shared-tokens figure needs no such caveat and says the
same thing.
"""
import json
import os
import sys

import numpy as np

# `under_test` moved out of the runtime package and into this directory when the arrangement's
# code was split for merging; five benchmarks kept importing it from where it used to be and none
# of them had been run since. A benchmark that cannot import is at least loud -- the dangerous
# version of this is the one that imports something ELSE of the same name.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from under_test import model_path

# named once, in the environment, and checked for quantisation before the weights
# load. It was written down separately in eight tools, which is eight chances to
# leave one on the old checkpoint and report its numbers under the new one's name.
MODEL = model_path()
TOKENS = "/home/user/experiment/v6/run/tokens/val_Qwen_Qwen3.8-27B.u32"
N, PROMPT_LEN, GEN_LEN = 500, 32, 128
GREEDY = {"temperature": 0.0, "max_new_tokens": GEN_LEN}


def draw(tokenizer, seed=7):
    tokens = np.fromfile(TOKENS, dtype=np.uint32)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(tokens) - (PROMPT_LEN + GEN_LEN) - 1, size=N)
    prompts, humans = [], []
    for s in starts:
        window = tokens[s: s + PROMPT_LEN + GEN_LEN].astype(np.int64).tolist()
        prompts.append(tokenizer.decode(window[:PROMPT_LEN]))
        humans.append(tokenizer.decode(window))
    return prompts, humans


def run(shift, prompts, shuffled_too):
    import sglang as sgl

    engine = sgl.Engine(
        model_path=MODEL, tp_size=1, mem_fraction_static=0.80, disable_cuda_graph=True,
        attention_backend="triton", log_level="warning", afd_query_shift_layers=shift,
        afd_coverage="all", afd_split_attention=True)
    try:
        plain = [o["text"] for o in engine.generate(prompts, sampling_params=GREEDY)]
        print(f"    shift {shift}: {len(plain)} greedy continuation(s), corpus order", flush=True)
        if not shuffled_too:
            return plain, None
        order = np.random.default_rng(11).permutation(len(prompts))
        shuffled = [prompts[i] for i in order]
        got = [o["text"] for o in engine.generate(shuffled, sampling_params=GREEDY)]
        back = [None] * len(prompts)
        for position, original in enumerate(order):
            back[original] = got[position]
        print(f"    shift {shift}: {len(back)} greedy continuation(s), shuffled order", flush=True)
        return plain, back
    finally:
        engine.shutdown()


def agreement(tokenizer, a, b):
    """Exact matches and, where they differ, how many tokens they shared first."""
    exact, prefixes = 0, []
    for x, y in zip(a, b):
        if x == y:
            exact += 1
            prefixes.append(GEN_LEN)
            continue
        tx = tokenizer(x)["input_ids"]
        ty = tokenizer(y)["input_ids"]
        n = 0
        for u, v in zip(tx, ty):
            if u != v:
                break
            n += 1
        prefixes.append(n)
    return {"exact": exact, "exact_pct": 100.0 * exact / len(a),
            "median_shared_tokens": float(np.median(prefixes)),
            "mean_shared_tokens": float(np.mean(prefixes))}


def main():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prompts, humans = draw(tokenizer)
    print(f"  {N} prompt(s), greedy, {GEN_LEN} tokens", flush=True)

    stock, stock_reordered = run(0, prompts, True)
    shifted, _ = run(1, prompts, False)

    null = agreement(tokenizer, stock, stock_reordered)
    effect = agreement(tokenizer, stock, shifted)
    print(f"\n  stock vs stock, reordered : {null['exact']}/{N} identical "
          f"({null['exact_pct']:.1f}%), median {null['median_shared_tokens']:.0f} shared tokens",
          flush=True)
    print(f"  stock vs shifted          : {effect['exact']}/{N} identical "
          f"({effect['exact_pct']:.1f}%), median {effect['median_shared_tokens']:.0f} shared "
          f"tokens", flush=True)

    import mauve

    def score(p_text, q_text, name):
        r = mauve.compute_mauve(p_text=p_text, q_text=q_text, device_id=0,
                                max_text_length=PROMPT_LEN + GEN_LEN, verbose=False,
                                featurize_model_name="gpt2-large")
        print(f"  MAUVE {name:28s} {r.mauve:.4f}", flush=True)
        return float(r.mauve)

    full = [p + t for p, t in zip(prompts, stock)]
    full_re = [p + t for p, t in zip(prompts, stock_reordered)]
    full_sh = [p + t for p, t in zip(prompts, shifted)]
    scores = {
        "stock_vs_human": score(full, humans, "stock vs human"),
        "shifted_vs_human": score(full_sh, humans, "shifted vs human"),
        "stock_vs_stock_reordered": score(full, full_re, "stock vs stock reordered (null)"),
        "stock_vs_shifted": score(full, full_sh, "stock vs shifted"),
    }
    gap = scores["stock_vs_human"] - scores["shifted_vs_human"]
    null_gap = 1.0 - scores["stock_vs_stock_reordered"]
    print(f"\n  MAUVE: the shift costs {gap:+.4f}; reordering the SAME model costs "
          f"{null_gap:.4f}", flush=True)
    print(f"  -> {abs(gap) / null_gap:.2f}x the noise floor" if null_gap > 0 else
          "  -> the null is exactly 1.0; reordering changed nothing", flush=True)
    json.dump({"n": N, "gen_len": GEN_LEN, "greedy": True, "null_agreement": null,
               "effect_agreement": effect, "mauve": scores, "gap": gap, "null_gap": null_gap,
               "example": {"prompt": prompts[0], "human": humans[0][:600],
                           "stock": stock[0][:600], "shifted": shifted[0][:600]}},
              open("/home/user/experiment/sglang/afd_greedy_compare.json", "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
