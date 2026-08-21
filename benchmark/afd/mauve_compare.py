"""MAUVE between the shifted model's generations and the stock model's, against human text.

bpb says the shifted model is 2.7% worse at predicting held-out text. That is a statement about
one-step-ahead likelihood and it is not the same statement as "the text it generates is worse":
a model can lose likelihood and generate indistinguishably, or keep likelihood and degenerate.
MAUVE measures the gap between the DISTRIBUTION of generated text and the distribution of human
text, which is the question a serving audience is actually asking.

## Four sets, because two would not be readable

    human    the corpus's own continuations of the same prompts
    stock    shift 0, sampled
    stock'   shift 0, sampled again -- the SAME model, different randomness
    shifted  shift 1, coverage all, sampled

MAUVE(stock, human) and MAUVE(shifted, human) are the comparison. MAUVE(stock, stock') is the
null: it is what this metric reports at this sample size when there is no difference at all, and
without it a gap of a few hundredths is unreadable. MAUVE(stock, shifted) is the two models
against each other, to be read against that null.

Prompts and human continuations come from the same validation corpus the bits-per-byte number was
measured on, so the two measurements are about one model on one distribution.

Measured, Qwen3.8-27B-FP8, shift 1 coverage all, 500 prompts of 32 tokens, 128 sampled at
temperature 1.0 / top-p 0.95, gpt2-large featuriser:

    stock vs human              0.7230
    shifted vs human            0.7619
    stock vs stock (the null)   0.9539
    stock vs shifted            0.9406

The shifted model scores 0.0389 HIGHER against human text, and the null says this metric is
0.0461 short of 1.0 at this sample size when nothing differs at all. The difference is 0.84x the
noise floor and points the opposite way from bits-per-byte. That is what a null result looks like:
at 500 samples MAUVE cannot tell these two models apart, and the honest reading is not that the
read point improves generation but that its 2.7% likelihood cost does not show up here. Resolving
the sign needs roughly the 5000 samples the MAUVE paper uses.
"""
import json
import sys

import numpy as np

from sglang.srt.afd.under_test import model_path

# named once, in the environment, and checked for quantisation before the weights
# load. It was written down separately in eight tools, which is eight chances to
# leave one on the old checkpoint and report its numbers under the new one's name.
MODEL = model_path()
TOKENS = "/home/user/experiment/v6/run/tokens/val_Qwen_Qwen3.8-27B.u32"
N, PROMPT_LEN, GEN_LEN = 500, 32, 128
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "max_new_tokens": GEN_LEN}


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


def generate(shift, prompts, draws):
    import sglang as sgl

    engine = sgl.Engine(
        model_path=MODEL, tp_size=1, mem_fraction_static=0.80, disable_cuda_graph=True,
        attention_backend="triton", log_level="warning", afd_query_shift_layers=shift,
        afd_coverage="all", afd_split_attention=True)
    out = []
    try:
        for _ in range(draws):
            outs = engine.generate(prompts, sampling_params=SAMPLING)
            out.append([p + o["text"] for p, o in zip(prompts, outs)])
            print(f"    shift {shift}: drew {len(out[-1])} continuation(s)", flush=True)
    finally:
        engine.shutdown()
    return out


def main():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prompts, humans = draw(tokenizer)
    print(f"  {N} prompt(s) of {PROMPT_LEN} tokens, {GEN_LEN} generated", flush=True)

    stock, stock_again = generate(0, prompts, 2)
    shifted = generate(1, prompts, 1)[0]

    import mauve

    def score(p_text, q_text, name):
        result = mauve.compute_mauve(p_text=p_text, q_text=q_text, device_id=0,
                                     max_text_length=PROMPT_LEN + GEN_LEN, verbose=False,
                                     featurize_model_name="gpt2-large")
        print(f"  MAUVE {name:26s} {result.mauve:.4f}", flush=True)
        return float(result.mauve)

    scores = {
        "stock_vs_human": score(stock, humans, "stock vs human"),
        "shifted_vs_human": score(shifted, humans, "shifted vs human"),
        "stock_vs_stock_again": score(stock, stock_again, "stock vs stock (the null)"),
        "stock_vs_shifted": score(stock, shifted, "stock vs shifted"),
    }
    gap = scores["stock_vs_human"] - scores["shifted_vs_human"]
    null_gap = 1.0 - scores["stock_vs_stock_again"]
    print(f"\n  the shift costs {gap:+.4f} MAUVE against human text", flush=True)
    print(f"  the same model against itself falls {null_gap:.4f} short of 1.0 at this sample "
          f"size", flush=True)
    print(f"  -> the shift's effect is {abs(gap) / null_gap:.2f}x the metric's own noise floor",
          flush=True)
    json.dump({"n": N, "prompt_len": PROMPT_LEN, "gen_len": GEN_LEN, "sampling": SAMPLING,
               "scores": scores, "gap": gap, "null_gap": null_gap,
               "example": {"prompt": prompts[0], "human": humans[0][:600],
                           "stock": stock[0][:600], "shifted": shifted[0][:600]}},
              open("/home/user/experiment/sglang/afd_mauve.json", "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
