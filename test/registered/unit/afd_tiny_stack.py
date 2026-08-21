"""A REAL Qwen3.5 stack, small enough to build in a unit test, for checking the span against.

The span cut has no split-exactness check, and the reason it has none is this file: every earlier
attempt compared the span against a hand-written fake, and the fakes were wrong in ways the
comparison could not see. One of them had a different shape than the real decoder layer and 234
cases passed against it. A fake cannot be the reference for a check whose whole question is
"does this compute what the model computes".

So this builds the model's own classes -- `Qwen3_5LinearDecoderLayer` and
`Qwen3_5AttentionDecoderLayer`, from `sglang.srt.models.qwen3_5` -- at 1/8 the depth and 1/40 the
width of Qwen3.8-27B, with random weights. Random is correct here: the question is whether two
computations of the SAME weights agree, and a trained checkpoint would only make a wrong answer
look plausible.

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


def build_tiny_stack(port: int = 29591):
    """The stack, its config, and its layer kinds. Built once and reused -- it is 1.36M params but
    the four initialisations before it are process-global and cannot be undone."""
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
        model_path=home, tp_size=1, disable_cuda_graph=True,
        attention_backend="torch_native", device="cpu",
    )
    rc.publish(server_args, role="scheduler", hf_config=config)
    init_distributed_environment(
        world_size=1, rank=0, local_rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{port}", backend="gloo",
    )
    initialize_model_parallel(tensor_model_parallel_size=1)

    # dp attention reads two fields off a ModelConfig and nothing else, and a real ModelConfig
    # wants a checkpoint on disk to build. Given the two rather than faked wholesale.
    model_config = ModelConfig.__new__(ModelConfig)
    model_config.hf_config = config
    model_config.hidden_size = config.hidden_size
    model_config.dtype = torch.float32
    initialize_dp_attention(server_args=server_args, model_config=model_config)

    from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM

    with torch.device("cpu"):
        stack = Qwen3_5ForCausalLM(config=config, quant_config=None)
    stack.eval()

    kinds = ["full" if hasattr(l, "attn") else "linear" for l in stack.layers]
    _BUILT = (stack, config, kinds)
    return _BUILT
