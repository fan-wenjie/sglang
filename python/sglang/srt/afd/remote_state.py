"""The pool's caches, named for where the state actually is: not here.

A routed pool answers frames and never receives a generate request. Its layers' attention runs
on the host, and where a recurrence is involved it calls back to the side that holds the history
(`afd/history_service.py`). So neither a KV cache nor a recurrent state belongs on this side --
and sglang nevertheless builds both, sized from whatever memory the weights did not take,
because nothing has told the memory pool that this process is not going to serve anyone.

Measured on a live pool, Qwen3.8-27B under the span cut: 12.88 GB of KV over 211,061 tokens and
11.33 GB of mamba state, against zero `/generate` requests, one logged batch -- the startup
warmup, 122 tokens -- and `full token usage: 0.00` for the whole run. About 24 GB reserved for a
warmup, on the card that also has to hold every weight in the model.

At that model it is waste. At a checkpoint whose weights are most of the card it is a refusal.

## Why a derived class rather than a bound

A cap on `--max-total-tokens` shrinks the KV cache and cannot touch the recurrent one: the mamba
size is divided by a per-request ratio to bound `--max-running-requests`, so lowering it lowers
the pool's concurrency rather than its footprint. And a bound leaves the arrangement's central
fact -- the state is the host's -- as a convention living in the callbacks, where a reader has to
already know it. A type says it.

sglang has the shape for this already: `NoOpMHATokenToKVPool` keeps the scheduler's view of
capacity and allocates placeholders, for embedding-mode prefill where no layer reads the pool.
Its precondition is checked against a list of families and refuses every hybrid and every MLA
one -- which is both models this arrangement runs. The precondition here is different and
stronger: not "the attention path happens to skip the cache" but "no attention runs in this
process at all", which is true of any family.

## What these hold

The smallest allocation the family will accept -- one page of slots, one state -- so that
pointer tables, stride arithmetic and any code holding a buffer reference keep working without
None-guards spreading through 135 files. The KV variant restores the LOGICAL size afterwards, so
admission accounting is unchanged; the recurrent variant does not, because its allocator indexes
the buffer it was given and a request admitted against a size that is not there is worse than a
request refused.

Every write path raises. If a routed pool ever does write a cache, that is a fact about the cut
worth an exception rather than a silent corruption, and it is how this file finds out it was
wrong.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_KV_VARIANTS: dict = {}
_STATE_VARIANTS: dict = {}
_REQ_VARIANTS: dict = {}


def serving_no_requests() -> bool:
    """Whether this process is a pool whose layers are routed, and so serves no request itself.

    Answers "no" when it cannot tell. A process with no published runtime -- a unit test, a tool
    -- is not an AFD pool, and a predicate that guesses "yes" would hand a real server a cache it
    needs.
    """
    try:
        from sglang.srt.runtime_context import get_disagg

        return get_disagg().afd_mode == "pool"
    except Exception:  # noqa: BLE001 -- no runtime, no routing: not a pool
        return False


def _refuse(what: str):
    def raiser(self, *args, **kwargs):
        raise RuntimeError(
            f"{type(self).__name__}.{what} was called on an AFD pool. This process serves "
            f"frames and its layers' attention runs on the host, so it holds no cache to write "
            f"-- the buffers here are the smallest the family would accept. Something ran an "
            f"attention on the pool: either a forward that was not routed (the startup warmup "
            f"is the usual one, and `handle_afd` disables it) or a cut that did not install."
        )

    return raiser


def remote_kv_cache(base: type) -> type:
    """`base`'s own family, standing in for a cache that lives on the host.

    Both `MHATokenToKVPool` and `MLATokenToKVPool` take `size` then `page_size` positionally, so
    one variant serves both and every subclass of either. Building with one page of slots rather
    than overriding `_create_buffers` is what keeps this family-independent: the family allocates
    its own shapes, in its own layout, and only the count changes.
    """
    if base in _KV_VARIANTS:
        return _KV_VARIANTS[base]

    class RemoteKVCache(base):
        def __init__(self, *args, **kwargs):
            # Two calling conventions in one tree: `MHATokenToKVPool(size, page_size, ...)`
            # takes them positionally and `HybridLinearKVPool(*, size=, page_size=, ...)` does
            # not. Handled here rather than by one variant per family, because what changes is
            # only which slot the count arrives in.
            if args:
                size, page_size, rest = args[0], args[1], args[2:]
                super().__init__(page_size, page_size, *rest, **kwargs)
            else:
                size = kwargs.pop("size")
                page_size = kwargs.get("page_size", 1)
                super().__init__(size=page_size, **kwargs)
            # The scheduler's view is unchanged: it is the physical bytes that are absent, not
            # the capacity the admission arithmetic was computed against.
            self.size = size
            logger.info(
                "afd pool: the KV cache is remote. %s built for %s logical tokens with one "
                "page of slots allocated here; this process runs no attention of its own.",
                base.__name__,
                size,
            )

        set_kv_buffer = _refuse("set_kv_buffer")

    RemoteKVCache.__name__ = f"Remote{base.__name__}"
    RemoteKVCache.__qualname__ = RemoteKVCache.__name__
    _KV_VARIANTS[base] = RemoteKVCache
    return RemoteKVCache


def remote_recurrent_state(base: type) -> type:
    """`base`'s own family of recurrent state, with one slot allocated instead of the limit.

    The logical size IS restored, exactly as on the KV side, and the first version of this did
    not -- which the scheduler caught on the first idle check: "[mamba] total=1, available=8".
    The allocator is built from the configurator's own number before this pool exists, so a pool
    that reports a smaller one puts the two out of agreement and every check that adds them up
    calls it a leak. Worse, the resolved `--max-running-requests` follows the mamba size down, and
    AFD sizes its own slot table from that limit (`afd/linear_runner.py::slots_wanted`), so a pool
    that shrank its state also shrank the concurrency it advertises to the host.

    What is left is a real hazard, stated rather than hidden: the buffer holds one slot and the
    accounting says several, so a process that ever admitted a request would index past it. It
    fails loudly -- an out-of-range index, not a silent write -- and it cannot happen in the
    process this class is for, which serves frames and no requests. A pool that admits one has a
    bigger problem than this line.
    """
    if base in _STATE_VARIANTS:
        return _STATE_VARIANTS[base]

    class RemoteRecurrentState(base):
        def __init__(self, *, size, **kwargs):
            super().__init__(size=1, **kwargs)
            # The scheduler's view is unchanged; it is the physical slots that are absent.
            self.size = size
            logger.info(
                "afd pool: the recurrent state is remote. %s asked for %s slots, one "
                "allocated here; this side reaches the history by callback.",
                base.__name__,
                size,
            )

    RemoteRecurrentState.__name__ = f"Remote{base.__name__}"
    RemoteRecurrentState.__qualname__ = RemoteRecurrentState.__name__
    _STATE_VARIANTS[base] = RemoteRecurrentState
    return RemoteRecurrentState


def remote_state_req_pool(base: type) -> type:
    """`base`, building the absent recurrent state instead of the family's own.

    The class attribute is how sglang already parameterises this (`mamba_pool_cls` on
    `HybridReqToTokenPool`), so a subclass that rebinds it is the whole change.
    """
    if base in _REQ_VARIANTS:
        return _REQ_VARIANTS[base]

    class RemoteStateReqPool(base):
        mamba_pool_cls = remote_recurrent_state(base.mamba_pool_cls)

    RemoteStateReqPool.__name__ = f"Remote{base.__name__}"
    RemoteStateReqPool.__qualname__ = RemoteStateReqPool.__name__
    _REQ_VARIANTS[base] = RemoteStateReqPool
    return RemoteStateReqPool


def _register() -> type:
    """Register the pool-side configurator. Called at import, which is what the loader does."""
    from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator

    @KVCacheConfigurator.register_substitute
    class PoolSideKVCacheConfigurator(KVCacheConfigurator):
        """The configurator for a process that serves frames and no requests.

        It replaces the default rather than editing it: everything about which pool family a
        model needs, how big it is, and how the allocator is wired stays the base's own decision.
        The one thing this changes is WHICH CLASS gets built, and it changes it in the one method
        the base asks.
        """

        @classmethod
        def claims(cls, **kwargs) -> bool:
            # Not a draft worker: a speculative draft is a second model with its own cache, and
            # this arrangement refuses speculative decoding anyway (`afd/compatibility.py`).
            # Answering for one here would be claiming a process this has never seen.
            return serving_no_requests() and not kwargs.get("is_draft_worker", False)

        def resolve_max_num_reqs(self, token_capacity: int) -> int:
            """How many requests this pool serves at once, which the operator states.

            The base caps the answer by how many state slots fit -- right for the side that
            keeps the state, and this side keeps none. Left to the base, `max_mamba_cache_size`
            of 1 over a ratio of 5 is zero requests, and a pool with weights loaded and its
            listener open declines to serve.

            `--max-running-requests` is the pool's own number and the only one that means
            anything here: it sizes the departure's seats, not a cache.
            """
            from sglang.srt.runtime_context import get_schedule

            wanted = get_schedule().max_running_requests
            if wanted is None:
                return super().resolve_max_num_reqs(token_capacity)
            return max(int(wanted) // self.ps.attn_dp_size, 1)

        def _profile_available_bytes(self, pre_model_load_memory):
            """The budget a routed pool has for its caches, which is a budget for nothing.

            The base measures what the weights left and REFUSES a non-positive answer, which is
            right for a server that will allocate a KV cache against it. This pool will not:
            `remote_kv_cache` reserves no storage and reports the logical size it was asked for.
            Running the base anyway turned a 48B checkpoint that fits -- 91.61 GiB loaded, 2.65
            GiB spare, `afd pool listening` already in the log -- into "raise
            --mem-fraction-static above 0.972", advice about a cache that does not exist.

            A nominal budget rather than zero, because the sizing downstream divides by a page
            and the remote pool restores its own logical size regardless. What it must not be is
            large: nothing here should look like room a caller could spend.
            """
            if self.mambaish_config is not None:
                self._handle_max_mamba_cache(0.0)
            return 1 << 20

        def _handle_max_mamba_cache(self, rest_memory):
            """A routed pool sizes for no recurrent state, because it holds none.

            The base computes how many request slots would fit and REFUSES a non-positive
            answer -- correct advice for a server that will actually allocate them. This pool
            will not: `remote_recurrent_state` hands back a pool that reserves nothing and
            reports the logical size it was asked for. Leaving the base to run anyway is not
            harmless arithmetic. It divides a budget that is what is left after 91.6 GiB of
            weights, gets `max_mamba_cache_size=-4`, and aborts a pool that had already logged
            `afd pool listening` -- a 48B checkpoint declared not to fit on a card it fits on,
            with 2.65 GB to spare.

            One slot, not zero: the release path checks that a pool hands back as many slots as
            it was built with, and the remote variants restore that logical size. Zero is the
            true reservation and one is the true count.
            """
            from sglang.srt.runtime_context import get_context

            get_context().override("afd.remote_state", max_mamba_cache_size=1)
            return rest_memory

        def pool_class(self, base: type) -> type:
            from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool

            if isinstance(base, type) and issubclass(base, HybridReqToTokenPool):
                return remote_state_req_pool(base)
            return remote_kv_cache(base)

    return PoolSideKVCacheConfigurator


PoolSideKVCacheConfigurator = _register()
