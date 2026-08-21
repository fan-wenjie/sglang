"""Recurrent states held on the pool, so a linear-attention layer reads its history over the wire.

The cache pool holds what a request remembers. For a softmax layer that is a KV cache; for a
linear-attention layer it is a recurrent state, and the arrangement's own rule -- whatever is a
per-request read belongs with the data, not with the weights -- says both go to the same place.

## The measurement that says this will lose, kept here because it should be easy to find

A linear layer's state is `num_value_heads x head_k_dim x head_v_dim` and does not grow: 1.5 MiB in
bfloat16 on this model, at 1k context and at 256k alike. Reading it costs the host 7 us a layer at
batch 8. Moving that read to another machine costs a round trip, measured at 934 us on this link.
Forty-eight linear layers, so about 45 ms a step spent to save 0.34 ms -- and unlike a KV cache,
no context length improves it, because the numerator is constant.

    context      softmax sweep a layer    linear state read a layer
       1024                      19 us                       7.0 us
      32768                     599 us                       7.0 us
     262144                    4793 us                       7.0 us

What DOES improve by moving it is host memory: 72 MiB a request over 48 layers, and the server logs
"max_running_requests is capped to 30 by the mamba state cache". The cap is this state. So the
trade is throughput against concurrency, and this file exists so the trade can be measured rather
than argued about.

## Why the two learned constants travel in every frame

`A_log` and `dt_bias` are per-layer parameters the recurrent kernel needs. The pool holds no
checkpoint and is not going to start: they are 48 floats each on this model, so they ride in the
frame -- 384 bytes against a 165 KiB payload, which is 0.2% -- and the pool never has to be told
about a model, kept in sync with one, or restarted when one changes.

## What stays on the host, and under which cut

The causal convolution before the projection has its own per-request state. Under the PER-LAYER
cut it stays on the host: it is about 60 KiB a layer a request against the recurrent state's
1.5 MiB, so moving it would add a second round trip for a fortieth of the memory, and `mixed_qkv`
arrives here already convolved.

Under the GROUP cut (`span.py`) it moves here with everything else, because there is no host-side
call left to run it in. `conv_buffer` below serves that arrangement, on this class's slot table.
"""

from __future__ import annotations

import threading

import torch


class LinearStates:
    """Per-layer recurrent state buffers, one slot per live request.

    Shaped `(slots, num_value_heads, head_v_dim, head_k_dim)` to match what the decode kernel
    expects as `ssm_states`, so the kernel writes the update in place and nothing here has to
    understand the recurrence.
    """

    def __init__(self, *, slots: int, num_v_heads: int, head_k_dim: int, head_v_dim: int,
                 device, dtype=torch.float32) -> None:
        if slots <= 0:
            raise ValueError(f"slots={slots}: a pool with no room for a request holds nothing")
        self.slots = slots
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.device = device
        # float32 because the state is accumulated across every step of a generation and a
        # bfloat16 accumulator drifts over thousands of updates in a way a single step never shows
        self.dtype = dtype

        self._lock = threading.Lock()
        self._states: dict[int, torch.Tensor] = {}
        self._slot_of: dict[int, int] = {}
        self._free = list(range(slots))
        self._touched: set[tuple[int, int]] = set()

    def slot_of(self, request_id: int) -> int:
        with self._lock:
            slot = self._slot_of.get(request_id)
            if slot is not None:
                return slot
            if not self._free:
                raise RuntimeError(
                    f"all {self.slots} recurrent-state slot(s) are taken and request "
                    f"{request_id} wants one. A recurrent state cannot be evicted and rebuilt "
                    f"from a prefix the way a KV cache can -- it is the whole history compressed "
                    f"-- so this refuses rather than dropping one."
                )
            slot = self._free.pop(0)
            self._slot_of[request_id] = slot
            return slot

    def release(self, request_id: int) -> int:
        """Free a request's slot and zero its states, so the next occupant starts from nothing.

        Zeroed, unlike the KV cache, where a length of zero already excludes stale positions. A
        recurrent state has no length: whatever is in the buffer IS the history, so a slot handed
        over without clearing gives the next request the previous one's memory, and the output
        stays fluent.
        """
        with self._lock:
            slot = self._slot_of.pop(request_id, None)
            if slot is None:
                return 0
            cleared = 0
            for layer, buffer in self._states.items():
                if (slot, layer) in self._touched:
                    buffer[slot].zero_()
                    self._touched.discard((slot, layer))
                    cleared += 1
            self._free.append(slot)
            return cleared

    def buffer(self, layer: int) -> torch.Tensor:
        with self._lock:
            found = self._states.get(layer)
            if found is not None:
                return found
            made = torch.zeros(
                (self.slots, self.num_v_heads, self.head_v_dim, self.head_k_dim),
                device=self.device, dtype=self.dtype)
            self._states[layer] = made
            return made

    def conv_buffer(self, layer: int, *, width: int, taps: int, dtype) -> torch.Tensor:
        """The short convolution's own per-request state, on the SAME slot table.

        Under the group cut the pool runs the linear layer whole, so the convolution runs here too
        and its state has to be here with it. It is small -- `width x taps`, about 60 KiB a layer a
        request against the recurrence's 1.5 MiB -- which is why the note above says it stays on
        the host: that was true of the per-layer cut, where moving it would have bought a fortieth
        of the memory for a second round trip. It is not true of the span, where the host is not in
        the loop to run it.

        Sharing `_slot_of` with the recurrent state is the point. Two slot tables that disagree
        would fold one request's convolution into another request's recurrence, and there is no
        symptom for that: both states are the whole history compressed, so a swap produces fluent
        text conditioned on somebody else's prompt.
        """
        key = ("conv", layer)
        with self._lock:
            found = self._states.get(key)
            if found is not None:
                return found
            made = torch.zeros((self.slots, width, taps), device=self.device, dtype=dtype)
            self._states[key] = made
            return made

    def note_touched(self, slots, layer) -> None:
        with self._lock:
            for slot in slots:
                self._touched.add((int(slot), layer))

    def report(self) -> dict:
        with self._lock:
            # summed rather than one-times-count: the conv buffers are a different shape from the
            # recurrent ones, so multiplying any single buffer by the layer count reports a number
            # wrong by whatever the mix happens to be
            recurrent = {k: b for k, b in self._states.items() if not isinstance(k, tuple)}
            conv = {k: b for k, b in self._states.items() if isinstance(k, tuple)}
            total = lambda d: sum(b.numel() * b.element_size() for b in d.values())
            return {
                "slots": self.slots,
                "slots_in_use": self.slots - len(self._free),
                "layers_allocated": len(recurrent),
                "conv_layers_allocated": len(conv),
                "bytes": total(recurrent) + total(conv),
                "bytes_recurrent": total(recurrent),
                "bytes_conv": total(conv),
            }


class LinearStateService:
    """One linear-attention layer's decode step, run where the state lives."""

    def __init__(self, states: LinearStates, *, scale: float) -> None:
        self.states = states
        self.scale = scale
        self.served = 0

    def step(self, request_ids, layer: int, mixed_qkv: torch.Tensor, a: torch.Tensor,
             b: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor) -> torch.Tensor:
        """Read the state with this token's query and update it, returning the layer's output.

        The read and the update are one kernel here because the recurrence fuses them: a gated
        delta rule's update needs the OLD state contracted against this token's key, so the two
        cannot be separated without writing a different kernel. That fusion is also why this call
        cannot be issued in the Early-Q window the way a softmax sweep can -- it needs this step's
        key and value, which do not exist until the layer runs.
        """
        from sglang.srt.layers.attention.linear.kernels.gdn_triton import (
            fused_recurrent_gated_delta_rule_packed_decode,
        )

        rows = mixed_qkv.shape[0]
        if a.shape[0] != rows or b.shape[0] != rows:
            raise RuntimeError(
                f"{rows} projected row(s) against {a.shape[0]} and {b.shape[0]} gate row(s); the "
                f"gates and the projection describe different tokens"
            )
        slots = [self.states.slot_of(int(r)) for r in request_ids]
        if len(slots) != rows:
            raise RuntimeError(
                f"{len(slots)} request id(s) for {rows} row(s); every row has to say whose state "
                f"it advances, and a mismatch here folds one request's token into another's memory"
            )
        buffer = self.states.buffer(layer)
        indices = torch.tensor(slots, device=mixed_qkv.device, dtype=torch.int32)
        out = mixed_qkv.new_empty(rows, 1, self.states.num_v_heads, self.states.head_v_dim)
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv, a=a, b=b, A_log=A_log, dt_bias=dt_bias, scale=self.scale,
            initial_state=buffer, out=out, ssm_state_indices=indices,
            use_qk_l2norm_in_kernel=True,
        )
        self.states.note_touched(slots, layer)
        self.served += 1
        return out.reshape(rows, self.states.num_v_heads * self.states.head_v_dim)
