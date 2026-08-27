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
        return {
            "layers": sorted(self.layers),
            "n_layers": len(self.layers),
            "gib_saved": self.bytes_saved / 1024**3,
        }


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

    def __init__(self, mlp_classes, keep: dict | None = None) -> None:
        self.mlp_classes = list(mlp_classes)
        # {class: (dotted parameter name, ...)} -- parameters that must survive on the real device
        # even though every instance of the class is built on meta. See `_restore`.
        self.keep = dict(keep or {})
        self._original: dict = {}
        self.built = 0
        self.kept = 0

    def __enter__(self):
        for cls in self.mlp_classes:
            original = cls.__init__
            self._original[cls] = original

            def made(self_, *args, _original=original, _outer=self, _cls=cls, **kwargs):
                with torch.device("meta"):
                    _original(self_, *args, **kwargs)
                _outer.built += 1
                _outer._restore(self_, _outer.keep.get(_cls, ()))
                # Marked, so the loader's post-load staging knows these modules have nothing
                # to stage rather than discovering a meta tensor and refusing it.
                #
                # Every DESCENDANT, not just this module: the loader stages per submodule that
                # carries a `quant_method`, and those are the linears INSIDE the feed-forward.
                # Marking only the parent marks the one module the loader never looks at.
                #
                # Set last, after `_restore` has put back whatever an arm asked to keep -- a
                # module that kept one tensor is still a module whose others are absent.
                self_.afd_weights_absent = True
                for child in self_.modules():
                    child.afd_weights_absent = True

            cls.__init__ = made
        return self

    def _restore(self, module, names) -> None:
        """Re-allocate named parameters on the real device, after the meta build.

        The class is named whole because the loader wraps a class rather than its instances, but
        an arm may still need ONE tensor out of it. Naming the class is not negotiable and a few
        MiB out of a layer stack is not worth a second mechanism, so the exemption is by parameter
        NAME inside the class every instance of which is still built on meta.

        Allocated OUTSIDE the `torch.device("meta")` context, so `torch.empty` takes the loader's
        own default device -- the same one every unrouted module got. Empty rather than zeroed:
        sglang fills it from the checkpoint by name afterwards, and a parameter that is present
        with the right shape and dtype is exactly what that lookup needs. A zero here would be
        indistinguishable from a weight that failed to load. A Parameter rather than a bare
        tensor, because the lookup goes through `named_parameters()`.
        """
        for name in names:
            owner, _, attr = name.rpartition(".")
            parent = module.get_submodule(owner) if owner else module
            held = getattr(parent, attr, None)
            if held is None or not held.is_meta:
                continue
            setattr(
                parent,
                attr,
                torch.nn.Parameter(
                    torch.empty(held.shape, dtype=held.dtype),
                    requires_grad=held.requires_grad,
                ),
            )
            self.kept += 1

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
            len(unskipped),
            unskipped[:8],
        )


_LAST_CONTEXT: BuildFeedForwardOnMeta | None = None


def skipping() -> bool:
    """Whether this process should build the feed-forward without storage.

    Read from the published config bag rather than from a flag some earlier code set. sglang runs
    the model loader in a scheduler process it spawns, and a module-level flag set in the launcher
    does not cross that boundary -- the first version of this set one, watched it stay False in
    the child, and reported the same out-of-memory it was written to prevent. The bag crosses it
    because the child publishes its own from the args it was handed, which is the same reason a
    parent-side override would not reach here.
    """
    from sglang.srt.runtime_context import get_disagg

    disagg = get_disagg()
    return bool(disagg.afd_mode == "host" and disagg.afd_pool_addr)


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
    # the pool's word first: which classes to build without storage is itself configuration,
    # and this is the first consumer of it in this process
    from sglang.srt.afd.pushed_config import adopt_from_the_pool

    adopt_from_the_pool()
    classes = _mlp_classes() + list(_arm_classes())
    if not classes:
        logger.warning(
            "afd host: --afd-mode=host asked to skip the feed-forward's weights and no MLP class "
            "was recognised, so all of them are being allocated. The host will work and will need "
            "room for the whole checkpoint."
        )
        return contextlib.nullcontext()
    _LAST_CONTEXT = BuildFeedForwardOnMeta(classes, keep=_arm_kept())
    return _LAST_CONTEXT


def _arm_classes():
    """What the installed arms compute remotely. Empty when none is installed.

    Asked here rather than listed here: this file must not know that any arm exists, and the
    registry is what keeps that true. `load()` first, because the loader runs in a process sglang
    spawned and a registry populated in the parent is empty in the child.
    """
    from sglang.srt.afd.arms import absent_classes, load
    from sglang.srt.runtime_context import get_server_args

    try:
        load()
        return absent_classes(get_server_args())
    except Exception as e:  # noqa: BLE001 -- reported, never swallowed
        logger.warning("afd loader: could not ask the arms what to skip: %r", e)
        return ()


def _arm_kept():
    """Parameters the arms need kept inside classes they declared absent. See `arms.kept_parameters`."""
    from sglang.srt.afd.arms import kept_parameters, load
    from sglang.srt.runtime_context import get_server_args

    try:
        load()
        return kept_parameters(get_server_args())
    except Exception as e:  # noqa: BLE001 -- reported, never swallowed
        logger.warning("afd loader: could not ask the arms what to keep: %r", e)
        return {}


def _mlp_classes():
    """The MLP classes this arrangement knows how to route, named rather than guessed.

    A search for "anything called MLP" would eventually wrap a module whose output the router does
    not speak for, and the symptom would be a meta tensor deep inside a forward.
    """
    found = []
    try:
        from sglang.srt.models.qwen2_moe import Qwen2MoeMLP

        found.append(Qwen2MoeMLP)
    except Exception:  # a build without this family
        pass
    return found
