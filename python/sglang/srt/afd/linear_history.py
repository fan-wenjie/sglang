"""A linear-attention layer, split the way a softmax one already is: a read, then a mix.

The arrangement's rule is that a per-request read belongs with the request's data and a weight read
belongs with the weights. For a softmax layer that split is obvious and already built: the host
holds a KV cache, sweeps it with a query, and returns what it read; the pool holds the projections
and folds this step's key and value in.

A linear layer looked like it could not be split that way. `linear_state.py` said so, and the
chance to overlap an attention with a feed-forward was held to exist only at the sixteen layers in
sixty-four that sweep a cache. That was wrong, and `benchmark/afd/gdn_split.py` is the measurement
that says so: the recurrence is
AFFINE in the old state, so

    h_q = S q                                    a read, with the query alone
    h_k = S k                                    a read, with the key alone
    o   = alpha h_q + beta (v - alpha h_k)(k.q)  the two readings, mixed with scalars

reproduces the fused kernel elementwise -- 2.2e-3 relative on the output at bfloat16, 9.7e-8 on the
state update. The fusion is the kernel's, not the recurrence's.

## The interface the two kinds of layer now share

    stage        softmax layer                     linear layer
    read         sweep the cache with q            read the state with q and with k
    returns      o_hist, lse                       h_q, h_k
    mix          fold in this step's k, v          alpha h_q + beta (v - alpha h_k)(k.q)
    holds        the KV cache                      the recurrent state and the convolution's

Both sides are then doing one thing each in both kinds of layer. The host holds a history, reads
it, and returns the reading. The pool holds the weights, and mixes.

## Why the state stays on the host

Because it is the request's, and the pool is stateless -- a caller that stalls stops calling and
blocks nobody, which is the property the pool is a separate process for. Holding a request's
recurrent state on the pool would reserve it between that request's calls, and the arrangement's
own docstring says it must not.

It also puts the update where sglang's own prefill path already lives. A pool that held the state
would have to run a chunked delta rule for prefill as well as a recurrent one for decode; a host
that holds it just runs the model.

## What the split costs, and what it buys

It costs a reading that a fused kernel did in one pass: two contractions of the state instead of
one. The state is 1.5 MiB a layer a request and the read is 7.0 us at any context, so the second
contraction is the same 7.0 us again -- against a round trip of 628 us, which is what the split
exists to overlap.

It buys the window at every layer instead of at one in four. On this model that is 64 against 16.
"""

from __future__ import annotations

import torch


def gates(
    a: torch.Tensor, b: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The decay and the write strength, in the form the fused sigmoid gating defines them.

    Separated out because they are the only part of the recurrence that is not a state read, and
    they are computed WHERE THE WEIGHTS ARE: `A_log` and `dt_bias` are per-layer parameters, so
    the side that holds the layer holds them, and the host never has to be told about a model.
    """
    alpha = torch.exp(-torch.exp(A_log) * torch.nn.functional.softplus(a + dt_bias))
    return alpha, torch.sigmoid(b)


def write_strength(b: torch.Tensor) -> torch.Tensor:
    """`beta` alone, for a caller that has no use for the decay.

    The decay is five elementwise kernels -- two exponentials, a softplus, a multiply and a
    negation -- and the write strength is one sigmoid. A caller that takes the pair and discards
    the decay pays for all six; MEASURED at 0.154 ms of a 0.423 ms projection, on a path where
    0.261 ms is the whole of what the arrangement saves.

    Not a second definition of the same thing: `gates` returns this expression for beta and this
    function IS that expression, so a change to one is a change to both or a test failure.
    """
    return torch.sigmoid(b)


def normalise(
    q: torch.Tensor, k: torch.Tensor, *, scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """What the kernel does to the query and key before either touches the state.

    `use_qk_l2norm_in_kernel=True` is not decoration: the state read is a contraction against
    these vectors, and reading with the unnormalised ones is a different model that still produces
    text. Applied here so both ends of the split apply it once, in one place.
    """
    return (
        torch.nn.functional.normalize(q.float(), dim=-1) * scale,
        torch.nn.functional.normalize(k.float(), dim=-1),
    )


def query_coefficient(
    q: torch.Tensor, k: torch.Tensor, beta: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold the key's correction into the query, so the history is read ONCE.

    The reading the weight side needs is

        alpha h_q - alpha beta (k.q) h_k  =  alpha S q - alpha beta (k.q) S k
                                          =  alpha S [q - beta (k.q) k]

    because the state enters both terms linearly. So the whole of it is one contraction against
    one vector -- and that vector is computable HERE, from the query, the key and the write
    strength, all of which the weight side has. The history side never sees the key on the
    critical path.

    What it changes, which is not the byte count (2096 up against 2144) but the pass count:

        before   two contractions of the state, both on the critical path
        after    one contraction on the critical path; the second, and the update it feeds,
                 move to the deferred message that carries the key and the value

    Measured at 8.5e-08 relative against the two-reading form.

    Returns the coefficient and `beta (k.q)`, because the caller needs the second to add its own
    value term back: `core = alpha S q_tilde + beta (k.q) v`.
    """
    kq = (k * q).sum(-1)
    s = beta * kq
    return q - s.unsqueeze(-1) * k, s


def read(
    state: torch.Tensor, q: torch.Tensor, k: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The host's whole job in a linear layer: two contractions of the state it holds.

    `state` is (rows, value heads, value dim, key dim) and both queries are (rows, value heads,
    key dim) -- already expanded from the key heads, because the state is per value head and a
    read that broadcast wrongly here would mix one head's history into another's.

    Both readings are against the state as it stands BEFORE this step. That is what makes `h_q`
    sendable the moment the query arrives, one message ahead of the key and value.
    """
    if state.shape[:2] != q.shape[:2] or q.shape != k.shape:
        raise ValueError(
            f"state {tuple(state.shape)} against q {tuple(q.shape)} and k {tuple(k.shape)}: the "
            f"two readings and the state have to agree on rows and heads, and a broadcast that "
            f"papered over this would read one head's history for another and stay fluent."
        )
    return (
        torch.einsum("bhvk,bhk->bhv", state, q),
        torch.einsum("bhvk,bhk->bhv", state, k),
    )


def mix(
    h_q: torch.Tensor,
    h_k: torch.Tensor,
    *,
    v: torch.Tensor,
    k: torch.Tensor,
    q: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """The layer's output, from the two readings and this step's key and value.

    Runs where the weights are: everything here is either a scalar per head or a vector the
    weight-holding side projected itself, so the history never has to travel.
    """
    al, be = alpha.unsqueeze(-1), beta.unsqueeze(-1)
    kq = (k * q).sum(-1).unsqueeze(-1)
    return al * h_q + be * (v.float() - al * h_k) * kq


def update(
    state: torch.Tensor,
    h_k: torch.Tensor,
    *,
    v: torch.Tensor,
    k: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """The new state, from the same reading the mix used. Runs where the state lives.

    Takes `h_k` rather than recomputing it: it is the same contraction, and computing it twice
    would be the second read this split was careful to charge for once.
    """
    al, be = alpha.unsqueeze(-1), beta.unsqueeze(-1)
    written = be * (v.float() - al * h_k)
    return al.unsqueeze(-1) * state + written.unsqueeze(-1) * k.unsqueeze(-2)


def expand_to_value_heads(x: torch.Tensor, value_heads: int) -> torch.Tensor:
    """Repeat a key-head tensor across the value heads that share it.

    The state is per value head and the query and key are per key head, so one of them has to be
    expanded before they meet. Done explicitly rather than by broadcasting, because a broadcast
    with the factor wrong is a model that reads a neighbour's history and says nothing.
    """
    heads = x.shape[1]
    if value_heads % heads:
        raise ValueError(
            f"{value_heads} value head(s) do not divide among {heads} key head(s); the expansion "
            f"factor is not an integer and no rounding of it is the model."
        )
    return x.repeat_interleave(value_heads // heads, dim=1)


class HistoryCache:
    """What a request remembers in a linear layer, held where a KV cache is held.

    The softmax side of this arrangement has the host hold a KV cache and sweep it. This is the
    same thing for the other three layers in four: the host holds the recurrent state and the
    convolution's, reads them when the pool asks, and returns the reading.

    ## Why it is here and not on the pool

    The pool is stateless, which is the property it is a separate process for -- a caller that
    stalls stops calling and blocks nobody, and the pool need not be reserved for a request between
    that request's calls. A recurrent state on the pool would reserve it.

    It also puts the update where sglang's own prefill path already is. A pool holding the state
    would need a chunked delta rule for prefill and a recurrent one for decode; a host holding it
    runs the model it already has.

    ## What it costs, measured

    Every linear layer then needs a round trip that the span did not: 64 a decode step against 17.
    Under a max-of-both-ends accounting that is +19% of step time at batch 4 and at batch 16, and
    the link binds beyond about batch 29 on this 10 GbE overlay. Almost all of the 19% is the
    148 us of protocol a round trip costs, not the wire and not the latency -- so it is a transport
    number, and on a fabric where a round trip is tens of microseconds it goes to about 2%.

    ## The two states share one slot table

    Deliberately. Two tables that disagreed would fold one request's convolution into another's
    recurrence, and both states ARE the whole history compressed -- there is no length that would
    exclude a stale entry. The output stays fluent and is conditioned on somebody else's prompt.
    """

    def __init__(
        self,
        *,
        slots: int,
        layers: int,
        value_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        conv_width: int,
        conv_taps: int,
        device,
        dtype=torch.float32,
        conv_dtype=torch.bfloat16,
    ) -> None:
        if slots <= 0:
            raise ValueError(
                f"slots={slots}: a host with no room for a request holds nothing"
            )
        self.slots = slots
        self.device = device
        self.value_heads = value_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        # float32 because bfloat16 cannot hold this state, MEASURED
        # (`benchmark/afd/state_precision.py`, against a float64 reference):
        #
        #     decay   effective memory   fp32 error   bf16 error
        #     0.90            10 steps    2.03e-07     1.39e-02
        #     0.99           100 steps    5.71e-07     1.13e-01
        #     0.999         1000 steps    2.54e-06     2.47e-01
        #
        # Note what the measurement CORRECTS. The reason written here before was that a bfloat16
        # accumulator "drifts over thousands of updates", and that is the wrong axis: the recurrence
        # decays, so old error decays with old signal and the error SATURATES at about 1/(1-alpha)
        # steps -- 1.5e-2 at ten steps, 1.19e-1 at five hundred, and flat at 1.13e-1 out to four
        # thousand. It does not grow with the length of a generation.
        #
        # The conclusion survives the reason being wrong, and by a wide margin: the saturated error
        # is 11% to 25% of the state. bfloat16 has eight mantissa bits and this is a sum of roughly
        # 1/(1-alpha) rank-one terms, so the rounding compounds to exactly that order. It is not a
        # precision concession, it is a different model.
        self.state = torch.zeros(
            layers,
            slots,
            value_heads,
            head_v_dim,
            head_k_dim,
            device=device,
            dtype=dtype,
        )
        self.conv = torch.zeros(
            layers, slots, conv_width, conv_taps, device=device, dtype=conv_dtype
        )
        self._slot_of: dict[int, int] = {}
        self._free = list(range(slots))
        self.reads = 0

    def slot_of(self, request_id: int) -> int:
        slot = self._slot_of.get(request_id)
        if slot is not None:
            return slot
        if not self._free:
            raise RuntimeError(
                f"all {self.slots} history slot(s) are taken and request {request_id} wants one. "
                f"A recurrent state cannot be evicted and rebuilt from a prefix the way a KV cache "
                f"can -- it is the whole history compressed -- so this refuses rather than "
                f"dropping one."
            )
        slot = self._free.pop(0)
        self._slot_of[request_id] = slot
        return slot

    def release(self, request_id: int) -> bool:
        """Free a request's slot and ZERO both its states.

        Zeroed, unlike a KV cache where a length of zero already excludes stale positions. These
        states have no length: whatever is in the buffer IS the history, so a slot handed over
        without clearing gives the next request the previous one's memory, and nothing about the
        output says so.
        """
        slot = self._slot_of.pop(request_id, None)
        if slot is None:
            return False
        self.state[:, slot].zero_()
        self.conv[:, slot].zero_()
        self._free.append(slot)
        return True

    def read_and_update(self, request_ids, layer: int, *, q, k, v, alpha, beta):
        """A REFERENCE form of one step, kept for the checks that pin the recurrence. Not the path.

        It was the host's whole job under an earlier arrangement, where the host took BOTH
        readings -- one for the query and one for the state's own copy of the key. The production
        path does not: the pool forms the query coefficient itself, asks for one reading with
        OP_STATE_READ, and defers the advance with OP_STATE_UPDATE. That split is the arrangement
        rather than an omission -- the deferral is what lets the caller carry on without waiting
        for its own state to be written, and it is the one contraction the query coefficient
        bought.

        So nothing in the runtime calls this, and wiring it in would UNDO the split. It stays
        because the cases below use it to pin the recurrence against the reference step, and a
        reference implementation exercised by tests is not dead code -- but it is only that, and
        the docstring said "the host's whole job" long after it stopped being true.

        `q` and `k` arrive normalised and expanded to the value heads -- the pool does that,
        because the scale and the head expansion are the layer's own and the side holding the
        history should not have to know which model it is holding.

        Returns the two readings. The caller mixes them, because the mixing needs `v` and the
        gates and those belong with the weights.
        """
        slots = [self.slot_of(int(r)) for r in request_ids]
        if len(slots) != q.shape[0]:
            raise RuntimeError(
                f"{len(slots)} request id(s) for {q.shape[0]} row(s); every row has to say whose "
                f"history it reads, and a mismatch folds one request's token into another's."
            )
        index = torch.tensor(slots, device=self.state.device, dtype=torch.long)
        held = self.state[layer].index_select(0, index)
        h_q, h_k = read(held, q, k)
        self.state[layer].index_copy_(
            0, index, update(held, h_k, v=v, k=k, alpha=alpha, beta=beta)
        )
        self.reads += 1
        return h_q, h_k

    def report(self) -> dict:
        held = self.state.numel() * self.state.element_size()
        conv = self.conv.numel() * self.conv.element_size()
        return {
            "slots": self.slots,
            "slots_in_use": self.slots - len(self._free),
            "reads": self.reads,
            "bytes_recurrent": held,
            "bytes_conv": conv,
            "bytes_a_request": (held + conv) // max(self.slots, 1),
        }


def prefill_scan(
    state: torch.Tensor,
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
):
    """One request's chunk of tokens, in order, advancing one state.

    A prefill row is a TOKEN, not a request, and the tokens of one chunk are sequentially
    dependent: the state after token n is what token n+1 reads. Treating the chunk as a batch --
    which is what a decode step is, one token from each of several requests -- runs every token
    against the same starting state and never advances it. The output is fluent and the model has
    no memory of its own prompt.

    That mistake was made three times in this arrangement, in three places, because in DECODE a
    row is a request AND a token and the two are indistinguishable. This function exists so the
    difference has a name and a test.

    Shapes are (tokens, value heads, dim) with the state (value heads, value dim, key dim) for
    ONE request. Returns the readings, one a token, and the state after the last of them.
    """
    if q.shape != k.shape:
        raise ValueError(
            f"query {tuple(q.shape)} against key {tuple(k.shape)}: a chunk's tokens are read with "
            f"both, and a mismatch would scan a different number of steps for each."
        )
    readings = []
    for t in range(q.shape[0]):
        h_q, h_k = read(state.unsqueeze(0), q[t : t + 1], k[t : t + 1])
        readings.append(
            mix(
                h_q,
                h_k,
                v=v[t : t + 1],
                k=k[t : t + 1],
                q=q[t : t + 1],
                alpha=alpha[t : t + 1],
                beta=beta[t : t + 1],
            )[0]
        )
        state = update(
            state.unsqueeze(0),
            h_k,
            v=v[t : t + 1],
            k=k[t : t + 1],
            alpha=alpha[t : t + 1],
            beta=beta[t : t + 1],
        )[0]
    return torch.stack(readings, dim=0), state


def prefill_convolve(ring: torch.Tensor, x: torch.Tensor, weight: torch.Tensor):
    """A chunk's causal convolution, and the ring it leaves behind.

    `ring` is (channels, taps) -- the K entries before this chunk -- and `x` is (tokens, channels).
    The convolution is depthwise and causal, so a chunk is one `conv1d` over the ring followed by
    the chunk, and the new ring is the last K of that sequence.

    Scanning it token by token would give the same answer and cost a kernel launch each; treating
    it as a batch would give a DIFFERENT answer, because every token would be convolved against
    the pre-chunk ring instead of against its own predecessors.
    """
    taps = weight.shape[-1]
    if ring.shape[-1] != taps:
        raise ValueError(
            f"a ring of {ring.shape[-1]} against a {taps}-tap kernel. The window is "
            f"[ring[1:], x] and it has to be as wide as the weight; K-1 was the first version of "
            f"this and it is the shape that does not raise, it broadcasts."
        )
    seq = torch.cat([ring[:, 1:], x.transpose(0, 1)], dim=-1).unsqueeze(0)
    out = torch.nn.functional.conv1d(seq, weight.unsqueeze(1), groups=weight.shape[0])[
        0
    ].transpose(0, 1)
    return torch.nn.functional.silu(out), seq[0, :, -taps:]


def core_from_mixed(
    mixed: torch.Tensor,
    *,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    key_heads: int,
    value_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    read_state,
    query=None,
) -> torch.Tensor:
    """Everything between the convolution and the z-gate, for the side that HOLDS the history.

    This is the host's half once the convolution moves off the pool. `mixed` is the post-convolution
    `[q | k | v]` laid out flat; `read_state(q_tilde)` contracts the recurrent state this side keeps
    and answers `S q~`; the result is `core`, which is what the pool needs back to apply the z-gated
    norm and `out_proj`.

    A function of its arguments rather than a method, so the two arrangements can be run against
    each other in ONE process -- the pool doing all of it and calling back for `S q~`, and this
    doing it locally -- and be compared to the bit. A move that changes the arithmetic is not a
    move, and the only way to know is to run both.

    `z` is deliberately not here. It is projected on the pool and consumed there by the norm, so it
    never crosses. The traffic table in AFD_STATELESS_POOL.md was pessimistic by that much: 12 KiB
    a layer a row.
    """
    rows = mixed.shape[0]
    width = key_heads * head_k_dim
    q = mixed[:, :width].reshape(rows, key_heads, head_k_dim)
    k = mixed[:, width : 2 * width].reshape(rows, key_heads, head_k_dim)
    v = mixed[:, 2 * width :].reshape(rows, value_heads, head_v_dim)
    q, k = normalise(q, k, scale=head_k_dim**-0.5)
    q = expand_to_value_heads(q, value_heads)
    k = expand_to_value_heads(k, value_heads)
    if query is None:
        q_tilde, s = query_coefficient(q, k, beta)
        reading = read_state(q_tilde)
    else:
        # The query came from elsewhere and the state was contracted with it there. What is left
        # is `s`, and the operator says which key it takes: `beta (k.q) v` and `beta v k^T` are
        # the same term of the same product, so the key here is THIS step's -- the one that
        # advances the state -- and only the query is the caller's.
        #
        # `q` above is then a query nobody wants, computed from channels nobody needed to send.
        # It stays until the payload drops them, so that this change is one change.
        s = beta * (k * query).sum(-1)
        reading = read_state(None)
    core = alpha.unsqueeze(-1) * reading + s.unsqueeze(-1) * v.float()
    return core.reshape(rows, -1), (k, v)


# One index tensor per (slots, device, dtype), reused. Building it is a host-to-device copy of a
# Python list, and at decode the same slots recur every layer of every step -- so the copy was
# being paid 48 times a token for a tensor that had not changed.
#
# Keyed on slot NUMBERS, not on request ids, and that is what makes reuse safe: the tensor's value
# depends only on the numbers, so a slot handed from one request to the next gets the same tensor
# and it is the right one. Nothing per-request is cached here.
#
# CAPPED, because the key space is arrangements of slots rather than slots: a batch that changes
# shape every step would otherwise add an entry every step forever. The cap is generous next to
# how many arrangements a stable decode actually visits, and clearing wholesale is right for what
# this is -- a cache with nothing to lose, refilled in one step.
_SLOT_INDEX: dict = {}
_SLOT_INDEX_CAP = 512


def _slot_index(slots, device) -> torch.Tensor:
    key = (tuple(slots), device)
    found = _SLOT_INDEX.get(key)
    if found is None:
        if len(_SLOT_INDEX) >= _SLOT_INDEX_CAP:
            _SLOT_INDEX.clear()
        found = _SLOT_INDEX[key] = torch.tensor(slots, device=device, dtype=torch.long)
    return found


def _row_index(slots, device) -> torch.Tensor:
    """The same as `_slot_index`, in the dtype the state kernels take.

    Built per call from the same slots by whoever reads or writes the state, and at decode the
    slots do not change between layers or between steps.
    """
    key = (tuple(slots), device, "i32")
    found = _SLOT_INDEX.get(key)
    if found is None:
        if len(_SLOT_INDEX) >= _SLOT_INDEX_CAP:
            _SLOT_INDEX.clear()
        found = _SLOT_INDEX[key] = torch.tensor(slots, device=device, dtype=torch.int32)
    return found


def convolve_with_ring(
    ring,
    qkv,
    weight,
    *,
    slots,
    runs,
    write: bool = True,
    defer_write: bool = False,
    write_channels: int | None = None,
):
    """The short convolution against a ring, for whichever side of the wire holds it.

    Split out of the pool's runner unchanged so that the HOST can run it once the ring moves there.
    The ring is per-request state -- this side's own last K projections -- so it belongs wherever
    the rest of a request's history is, and this function does not care which side that is: it
    takes the buffer, the rows, and where each run of one request sits.

    `slots` and `runs` line up: `runs[i]` is `(request_id, start, count)` and `slots[i]` is where
    that request's ring lives. The one-row path and the chunk path are kept apart because a chunk's
    tokens read each other's advance and a single row does not.

    The kernel is depthwise and causal, K = 4, each channel filtered on its own:

        out[c] = silu( bias[c] + sum over t of w[c,t] * x[c, n-K+1+t] )

    so the ring holds K entries, not K-1: the window IS `[ring[1:], x_new]`, which is K wide
    against a K-wide weight. An earlier version allocated K-1 and built a window of three against
    four -- caught by writing the formula down rather than by a test.
    """
    if all(n == 1 for _, _, n in runs):
        index = _slot_index(slots, qkv.device)
        held = ring.index_select(0, index)
        window = torch.cat([held[..., 1:], qkv.unsqueeze(-1)], dim=-1)
        # `write=False` reads the ring and leaves it. The ring is a state reused across steps, so
        # a value written into it outlives the step that wrote it; a caller that wants a window
        # without advancing the state asks for one. Only this step's own projection advances it.
        #
        # `defer_write` hands the advance back instead of applying it, for a caller that has
        # something to let finish first. Two readers of a ring do not conflict; a reader and a
        # writer do, and this is what lets the write be the last thing rather than the first.
        out = torch.nn.functional.silu((window * weight).sum(-1))
        if not write:
            return out
        if write_channels == 0:
            return out
        if write_channels is not None:
            # A PREFIX of the channels advances and the rest do not. One caller convolves two
            # groups whose histories come from different projections -- the ring's own layout puts
            # them side by side -- and only one of them is this caller's to advance.
            #
            # Written through the same `index_copy_` on a slice of the ring, so the two groups
            # cannot end up in different rows: the index is the same tensor either way.
            ring[:, :write_channels].index_copy_(0, index, window[:, :write_channels])
            return out
        if defer_write:
            return out, lambda: ring.index_copy_(0, index, window)
        ring.index_copy_(0, index, window)
        return out

    if not write:
        raise ValueError(
            "a read-only convolution was asked for a run of more than one row. A chunk's tokens "
            "read each other's taps, so there is no window to take without advancing the ring."
        )
    if defer_write:
        raise ValueError(
            "a deferred ring write was asked for a run of more than one row. A chunk's tokens "
            "read each other's advance as they go, so there is no single write to hold back."
        )
    pieces = []
    for slot, (_, start, count) in zip(slots, runs):
        got, tail = prefill_convolve(ring[slot], qkv[start : start + count], weight)
        ring[slot] = tail
        pieces.append(got)
    return torch.cat(pieces, dim=0)
