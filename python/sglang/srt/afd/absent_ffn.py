"""Build a host that never allocates the feed-forward it is going to route away.

A host in this arrangement computes attention and sends its hidden state to a pool that owns the
feed-forward. Installing the router was enough to make it BEHAVE that way and not enough to make
it fit: sglang constructs the whole model first and installs afd afterwards, so a 52 GiB
checkpoint is materialised on the card before anything is routed anywhere. On a 32 GiB card that
is an out-of-memory during construction, and the arrangement's one unarguable benefit -- a card
serving a model larger than itself -- is exactly what it fails to deliver.

    weights a host actually needs   14.9 GiB   embeddings, q/kv/o, the linear layers' projections
    weights it was allocating       52.0 GiB   all of the above plus 31.9 GiB of feed-forward

So the feed-forward's parameters are built on the meta device: shapes and names, no storage. The
loader then has nothing to copy into for those names, and the router never calls them.

## Why meta rather than deleting afterwards

Deleting after construction frees the memory but does not avoid the peak, and the peak is what
fails. Meta parameters cost nothing at any point.

## What this refuses to do

Silently serve a model whose feed-forward is absent AND unrouted. A meta parameter that reaches a
matmul raises a device mismatch from somewhere deep in the layer, and that is not a message about
configuration. So `verify_routed` is called after the router installs and checks that every layer
whose weights were skipped has in fact been routed: skipped and routed is the arrangement, skipped
and local is a model that will fail on its first token with an error about devices.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


class SkippedFeedForward:
    """A record of what was not allocated, so it can be checked against what was routed."""

    def __init__(self) -> None:
        self.layers: set[int] = set()
        self.bytes_saved = 0
        self.modules: dict[int, object] = {}

    def note(self, layer: int, module, saved: int) -> None:
        self.layers.add(layer)
        self.modules[layer] = module
        self.bytes_saved += saved

    def report(self) -> dict:
        return {"layers": sorted(self.layers), "n_layers": len(self.layers),
                "gib_saved": self.bytes_saved / 1024 ** 3}


class BuildFeedForwardOnMeta:
    """While this is active, a routed layer's feed-forward is constructed with no storage.

    Wraps the MLP class's __init__ in a `torch.device("meta")` context. The allocation this
    prevents happens INSIDE construction -- `create_weights` calls `torch.empty` without naming a
    device, so it takes the default one -- which is why moving the module to meta afterwards does
    not help: the out-of-memory has already happened by then.

    Every layer is wrapped, not a chosen subset, because the class is shared and the host routes
    every layer it can. `verify_routed` afterwards is what turns "wrapped but not routed" into a
    message instead of a device error during the first token.
    """

    def __init__(self, mlp_classes) -> None:
        self.mlp_classes = list(mlp_classes)
        self._original: dict = {}
        self.built = 0

    def __enter__(self):
        for cls in self.mlp_classes:
            original = cls.__init__
            self._original[cls] = original

            def made(self_, *args, _original=original, _outer=self, **kwargs):
                with torch.device("meta"):
                    _original(self_, *args, **kwargs)
                _outer.built += 1

            cls.__init__ = made
        return self

    def __exit__(self, *exc):
        for cls, original in self._original.items():
            cls.__init__ = original
        self._original.clear()
        return False


def to_meta(module: torch.nn.Module) -> int:
    """Move every parameter of a module to the meta device. Returns the bytes not allocated."""
    saved = 0
    for name, param in list(module.named_parameters(recurse=True)):
        saved += param.numel() * param.element_size()
    module.to("meta")
    return saved


def skip_feed_forward(model, layers) -> SkippedFeedForward:
    """Strip the storage from the named layers' feed-forward, keeping shapes and names.

    Called after construction and before weights load. The parameters still exist as far as the
    loader's name lookup is concerned, so a checkpoint that carries them is not an error; what it
    copies into is a meta tensor, which discards it.
    """
    skipped = SkippedFeedForward()
    stack = model.model.layers
    for index in layers:
        mlp = getattr(stack[index], "mlp", None)
        if mlp is None:
            continue
        skipped.note(index, mlp, to_meta(mlp))
    logger.info(
        "afd host: %s feed-forward(s) built on the meta device, %.1f GiB not allocated. They are "
        "computed by the pool; nothing on this card holds them.",
        len(skipped.layers),
        skipped.bytes_saved / 1024 ** 3,
    )
    return skipped


def verify_routed(skipped: SkippedFeedForward, routed) -> None:
    """Every layer whose weights were skipped must be one the router speaks for.

    The failure this prevents is not subtle but its message would be: a meta parameter reaching a
    matmul raises "expected all tensors to be on the same device" from inside a linear layer,
    which says nothing about a pool, a route, or a configuration. Checked here, while both lists
    are in hand.
    """
    routed = set(routed)
    orphaned = sorted(skipped.layers - routed)
    if orphaned:
        raise RuntimeError(
            f"layer(s) {orphaned} had their feed-forward left unallocated and are not routed to "
            f"the pool. Whatever runs them will find parameters on the meta device and fail with "
            f"a message about devices rather than about this arrangement. Either route them or "
            f"allocate them."
        )
    unskipped = sorted(routed - skipped.layers)
    if unskipped:
        logger.info(
            "afd host: %s routed layer(s) still hold their feed-forward weights (%s). They are "
            "computed remotely, so the memory is reserved and unused.",
            len(unskipped), unskipped[:8],
        )


_LAST_CONTEXT: BuildFeedForwardOnMeta | None = None


def skipping() -> bool:
    """Whether this process should build the feed-forward without storage.

    Read from the resolved server args rather than from a flag some earlier code set. sglang runs
    the model loader in a scheduler process it spawns, and a module-level flag set in the launcher
    does not cross that boundary -- the first version of this set one, watched it stay False in
    the child, and reported the same out-of-memory it was written to prevent.
    """
    from sglang.srt.server_args import get_global_server_args

    args = get_global_server_args()
    return bool(getattr(args, "afd_mode", None) == "host" and getattr(args, "afd_pool_addr", None))


def feed_forward_build_context():
    """A context that builds every feed-forward on meta, or does nothing.

    Returns a no-op unless a host has asked for it, so the ordinary loading path is unchanged and
    carries no import of a model file it does not need.
    """
    import contextlib

    global _LAST_CONTEXT
    try:
        wanted = skipping()
        why = "server args say host mode with a pool" if wanted else "not a routed host"
    except Exception as e:
        # Reported, not swallowed. A silent False here is a host that allocates the whole
        # checkpoint and dies during construction, and the exception that caused it would be the
        # only thing that could have said why.
        wanted, why = False, f"could not read the server args: {type(e).__name__}: {e}"
    logger.info("afd loader: feed-forward built with storage=%s (%s)", not wanted, why)
    if not wanted:
        return contextlib.nullcontext()
    classes = _mlp_classes()
    if not classes:
        logger.warning(
            "afd host: --afd-mode=host asked to skip the feed-forward's weights and no MLP class "
            "was recognised, so all of them are being allocated. The host will work and will need "
            "room for the whole checkpoint."
        )
        return contextlib.nullcontext()
    _LAST_CONTEXT = BuildFeedForwardOnMeta(classes)
    return _LAST_CONTEXT


def _mlp_classes():
    """The MLP classes this arrangement knows how to route, named rather than guessed.

    A search for "anything called MLP" would eventually wrap a module whose output the router does
    not speak for, and the symptom would be a meta tensor deep inside a forward.
    """
    found = []
    try:
        from sglang.srt.models.qwen2_moe import Qwen2MoeMLP

        found.append(Qwen2MoeMLP)
    except Exception:                                  # a build without this family
        pass
    return found
