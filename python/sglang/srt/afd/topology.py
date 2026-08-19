"""Rank layout for a two-sided AFD world.

=============================================================================
DERIVED FROM FastAFD (Hao AI Lab, UCSD), file `python/minisgl/afd_protocol.py`,
class `AfdTopology`.

    upstream: https://github.com/sgl-project/FastAFD
    licence : MIT, Copyright (c) 2026 sgl-project
    obtained: cloned under qfirst_serve/upstream/FastAFD

The mapping rules below -- one union world holding every attention worker
first and every feed-forward worker after, the divisibility constraints
between the two sides' data- and tensor-parallel sizes, and the three
supported expert-parallel modes -- are FastAFD's design and are reproduced
here rather than reinvented, because two components that open-code slightly
different rank arithmetic is exactly the failure the original centralises
away.

WHAT WAS CHANGED, and why:

  * the expert-parallel vocabulary is kept ("expert group", ep_size) even
    though the model this port targets is dense, so that a reader comparing
    the two files sees the same words for the same quantities.
  * `mlp_*` is renamed `pool_*`. In this port the second side is a stateless
    pool that any request may call between its own sweeps, not a worker a
    request is routed to and stays with; the name carries that difference.
  * FastAFD's AFD is SYNCHRONOUS: an attention worker sends and waits. This
    file is the layout only, and says nothing about when a call is collected;
    the asynchrony lives in `pool_client.py` and is this port's own.
  * `@dataclass` is kept rather than converted to `msgspec.Struct` as the
    repository's `no-dataclasses` rule would otherwise require. Vendored code
    is kept diffable against its upstream; a rewrite here would make every
    future comparison with FastAFD a manual exercise.
=============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AfdTopology:
    """Static host/pool worker layout for AFD.

    One union world contains all host (attention) workers first, followed by all pool
    (feed-forward) workers:

        [host dp0/tp0..N, host dp1/tp0..N, ..., pool dp0/tp0..N, pool dp1/tp0..N, ...]

    Supported expert-parallel modes, unchanged from upstream:

        ep_size == 1                            no EP
        ep_size == pool_tp_size                 per-pool-DP EP; experts replicated across replicas
        ep_size == pool_dp_size * pool_tp_size  full-world EP across all pool workers
    """

    host_dp_size: int
    pool_dp_size: int
    host_tp_size: int
    pool_tp_size: int
    ep_size: int = 1

    def __post_init__(self) -> None:
        for name in ("host_dp_size", "pool_dp_size", "host_tp_size", "pool_tp_size", "ep_size"):
            value = int(getattr(self, name))
            if value < 1:
                raise ValueError(f"afd topology {name} must be >= 1, got {value}")
            object.__setattr__(self, name, value)
        self._validate_mutually_divisible(
            "host_dp_size", self.host_dp_size, "pool_dp_size", self.pool_dp_size
        )
        self._validate_mutually_divisible(
            "host_tp_size", self.host_tp_size, "pool_tp_size", self.pool_tp_size
        )
        pool_world = self.pool_world_size
        if self.ep_size > pool_world or pool_world % self.ep_size != 0:
            raise ValueError(
                "afd topology requires ep_size to divide the pool world: "
                f"ep_size={self.ep_size} pool_world_size={pool_world} "
                f"(pool_dp_size={self.pool_dp_size} pool_tp_size={self.pool_tp_size})"
            )
        if self.ep_size not in (1, self.pool_tp_size, pool_world):
            raise ValueError(
                f"afd topology supports ep_size in (1, pool_tp_size={self.pool_tp_size}, "
                f"pool_world_size={pool_world}); got {self.ep_size}"
            )

    @staticmethod
    def _validate_mutually_divisible(a_name: str, a: int, b_name: str, b: int) -> None:
        if a % b and b % a:
            raise ValueError(
                f"afd topology requires {a_name} and {b_name} to divide one another: "
                f"{a_name}={a} {b_name}={b}"
            )

    @property
    def host_world_size(self) -> int:
        return self.host_dp_size * self.host_tp_size

    @property
    def pool_world_size(self) -> int:
        return self.pool_dp_size * self.pool_tp_size

    @property
    def world_size(self) -> int:
        return self.host_world_size + self.pool_world_size

    def is_host_rank(self, rank: int) -> bool:
        return 0 <= rank < self.host_world_size

    def is_pool_rank(self, rank: int) -> bool:
        return self.host_world_size <= rank < self.world_size

    def host_rank(self, dp: int, tp: int) -> int:
        return dp * self.host_tp_size + tp

    def pool_rank(self, dp: int, tp: int) -> int:
        return self.host_world_size + dp * self.pool_tp_size + tp
