"""A REAL Qwen3.5 stack, small enough to build in a unit test, for checking the span against.

The span cut has no split-exactness check, and the reason it has none is this file: every earlier
attempt compared the span against a hand-written fake, and the fakes were wrong in ways the
comparison could not see. One of them had a different shape than the real decoder layer and 234
cases passed against it. A fake cannot be the reference for a check whose whole question is
"does this compute what the model computes".

So this builds the model's own classes -- `Qwen3_5LinearDecoderLayer` and
`Qwen3_5AttentionDecoderLayer`, from `sglang.srt.models.qwen3_5` -- at 1/8 the depth and 1/40 the
width of Qwen3.8-27B, with weights drawn by `_fill_with_values`. Drawn rather than loaded is
correct here -- the question is whether two computations of the SAME weights agree, and a trained
checkpoint would only make a wrong answer look plausible -- but they must actually be DRAWN. The
constructor leaves them zero, which is its own kind of wrong answer; see that function.

What is preserved from the deployed model, because the span's geometry is defined by it:

    full_attention_interval  4, so the pattern is [linear, linear, linear, full] repeated -- the
                             softmax layer is the LAST of each group, which is what makes a group
                             a span
    rope_parameters          mrope, interleaved, with a partial rotary factor. The deployed model
                             carries `rope_parameters` and NOT `rope_scaling`; a config built with
                             the latter raises "Unknown RoPE scaling type" at `get_rope`
    layer attributes         `qkv_proj` / `o_proj` / `attn` / `mlp` on the softmax layer and
                             `linear_attn` / `mlp` on the linear one. `layer.self_attention` is a
                             METHOD on these classes, not a submodule, and reaching for it as one
                             is a mistake this tree has already made

What is NOT preserved: the width, the depth, the vocabulary, and the values. Anything a check
here concludes about magnitudes rather than about agreement is a conclusion about this fixture.

Building it needs four things done first, in order, none of which a model constructor does for
itself -- a published runtime context, a distributed environment, model parallel, and dp
attention. `build_tiny_stack` does them once a process and hands back the stack.
"""

from __future__ import annotations

import json
import os
import tempfile

import torch

_BUILT = None

# Qwen3.8-27B: 64 layers, full_attention_interval 4, hidden 5120, 24 query heads of 256, 48 linear
# value heads of 128, 16 linear key heads of 128. Scaled down, the ratios kept where the span's
# arithmetic depends on them.
TINY = dict(
    hidden_size=128,
    intermediate_size=256,
    num_hidden_layers=8,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    linear_num_value_heads=4,
    linear_value_head_dim=32,
    linear_num_key_heads=2,
    linear_key_head_dim=32,
    full_attention_interval=4,
    vocab_size=256,
    linear_conv_kernel_dim=4,
    max_position_embeddings=512,
    rope_parameters={
        "mrope_interleaved": True,
        "mrope_section": [1, 1, 2],
        "partial_rotary_factor": 0.25,
        "rope_theta": 10000000,
        "rope_type": "default",
    },
)


def _fill_with_values(stack, seed: int = 20260821) -> None:
    """Give the stack values. It has none until a checkpoint is loaded, and they are not garbage.

    sglang allocates parameters with `torch.empty` and expects a loader to fill them. On a fresh
    CUDA allocation that is ZERO, not noise: 63 of this stack's 84 parameters came back all-zero,
    including every projection. A span run against them returns its input unchanged -- exactly the
    signature of the bug this fixture was built to hunt, produced entirely by the fixture.

    That reading was taken and very nearly reported as a reproduction. What caught it was checking
    `max|w|` rather than trusting the word "random" in the paragraph above this one.

    Norm weights sit near one because a norm scaled by ~0.02 would make every residual dwarf every
    contribution and hide the same failure a different way. Everything else is small and normal.
    `A_log` and `dt_bias` set the linear attention's decay, so they are given the ranges the
    recurrence is defined over instead of a normal draw.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def draw(shape, scale):
        return torch.randn(*shape, generator=generator) * scale

    with torch.no_grad():
        for name, parameter in stack.named_parameters():
            if name.endswith("A_log"):
                # log of a decay in (0, 1): alpha = exp(-exp(A_log) * dt) must not saturate
                filled = draw(parameter.shape, 0.5)
            elif name.endswith("dt_bias"):
                filled = draw(parameter.shape, 0.1)
            elif "norm" in name and name.endswith("weight"):
                filled = 1.0 + draw(parameter.shape, 0.02)
            else:
                filled = draw(parameter.shape, 0.05)
            parameter.copy_(filled.to(parameter.dtype).to(parameter.device))

    still_zero = [
        n for n, p in stack.named_parameters() if float(p.float().abs().max()) == 0.0
    ]
    if still_zero:
        raise RuntimeError(
            f"{len(still_zero)} parameter(s) are still all-zero after filling, first "
            f"{still_zero[:3]}. A stack with a zero projection computes the identity, and a check "
            f"against it reports agreement whatever the code under test does."
        )


def build_tiny_stack(*, device: str, port: int = 29591):
    """The stack, its config, and its layer kinds. Built once and reused -- it is 1.36M params but
    the four initialisations before it are process-global and cannot be undone.

    `device` is required and not guessed. sglang's RMSNorm dispatches to `sgl_kernel::gemma_rmsnorm`,
    which exists for CUDA and Meta and not for CPU -- so a stack built on the CPU raises
    "Could not run 'sgl_kernel::gemma_rmsnorm' with arguments from the 'CPU' backend" at the first
    norm, several frames inside a fused op. The stack is 1.36M parameters; putting it on the card
    beside a pool costs nothing measurable.
    """
    global _BUILT
    if _BUILT is not None:
        return _BUILT

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(port))

    import sglang.srt.runtime_context as rc
    from sglang.srt.configs import Qwen3_5TextConfig
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.layers.dp_attention import initialize_dp_attention
    from sglang.srt.server_args import ServerArgs

    config = Qwen3_5TextConfig(**TINY)
    # ServerArgs resolves its model_path, and a name that is not a directory is looked up on the
    # Hub -- "Repository Not Found for url: .../tiny/resolve/main/config.json". The config is
    # written out so the path is real; no weights are read from it, the stack is built with
    # `config` above and random values.
    home = tempfile.mkdtemp(prefix="afd-tiny-stack-")
    written = config.to_dict()
    # ServerArgs subscripts `architectures` while deciding a model implementation, and the config
    # class does not fill it in: without it the failure is `'NoneType' object is not subscriptable`
    # from inside server_args, naming neither this file nor the field.
    written["architectures"] = ["Qwen3_5ForCausalLM"]
    with open(os.path.join(home, "config.json"), "w") as f:
        json.dump(written, f)
    server_args = ServerArgs(
        model_path=home,
        tp_size=1,
        disable_cuda_graph=True,
        attention_backend="torch_native",
        device=device,
    )
    rc.publish(server_args, role="scheduler", hf_config=config)
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend="gloo" if device == "cpu" else "nccl",
    )
    initialize_model_parallel(tensor_model_parallel_size=1)

    # dp attention reads two fields off a ModelConfig and nothing else, and a real ModelConfig
    # wants a checkpoint on disk to build. Given the two rather than faked wholesale.
    model_config = ModelConfig.__new__(ModelConfig)
    model_config.hf_config = config
    model_config.hidden_size = config.hidden_size
    model_config.dtype = torch.bfloat16
    initialize_dp_attention(server_args=server_args, model_config=model_config)

    from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM

    # bfloat16, which is what the deployment runs. Not a preference: `gemma_rmsnorm` refuses
    # float32 with "failed to dispatch data type Float", so a stack in the precision an exactness
    # check would rather have cannot take a single norm. A check that wants float64 has to reach
    # past sgl_kernel, and that is a decision for the check rather than for this fixture.
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device(device):
            stack = Qwen3_5ForCausalLM(config=config, quant_config=None)
    finally:
        torch.set_default_dtype(previous)
    stack.eval()

    # `group_layers` matches "full_attention" exactly, and these are the strings `layer_types_of`
    # returns off a real stack. Shortened to "full"/"linear" here once, and the span refused the
    # whole model with "no full-attention layer in this stack".
    _fill_with_values(stack)
    kinds = [
        "full_attention" if hasattr(l, "attn") else "linear_attention"
        for l in stack.layers
    ]

    # SpanRunner and SpanRouting both reach `model.model.layers`, which is the deployed shape:
    # qwen3_vl holds a language_model that holds the layers. What the constructor above returns is
    # already the inner module, so it is given the same one-hop reach rather than special-cased in
    # the span -- a span that learned a second shape here would be carrying this fixture.
    if not hasattr(stack, "model"):
        stack.model = stack

    _BUILT = (stack, config, kinds)
    return _BUILT
