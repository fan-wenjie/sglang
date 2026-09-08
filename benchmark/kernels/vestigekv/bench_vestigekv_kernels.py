"""Per-kernel benchmarks for the VestigeKV scan pipeline.

Convention follows benchmark/kernels/decoding_attention_triton (torch.utils
.benchmark.Timer, mean over repeats). Each section compares the shipped
kernel against the implementation it replaced, at serving shapes
(P layer-request pairs, archives sized for 64k/128k/256k contexts), so a
regression in any one kernel is visible in isolation rather than only in
end-to-end serving numbers.
"""

import torch
import torch.utils.benchmark as benchmark

import sglang.srt.layers.attention.vestigekv.defaults as D
from sglang.srt.layers.attention.vestigekv.fused_prologue import (
    _NSPLIT,
    compact_fired,
    fused_prologue,
    fused_prologue_split,
)
from sglang.srt.layers.attention.vestigekv.scan_kernel import vestige_scan

P, H, R, KV = 7, 32, 64, 512


def t(fn, repeats=50):
    return benchmark.Timer(stmt="fn()", globals={"fn": fn}).timeit(repeats).mean * 1e3


def shapes(S, dev="cuda"):
    torch.manual_seed(S)
    nk, a = S // 32 + 2048, S - S // 32
    q = torch.randn(P, H, 576, device=dev)
    kr = torch.randn(P, nk, 576, device=dev, dtype=torch.bfloat16)
    v = torch.stack(
        [
            torch.linalg.qr(torch.randn(KV, R, device=dev))[0].T.contiguous()
            for _ in range(P)
        ]
    )
    nk_len = torch.full((P,), nk, device=dev, dtype=torch.int64)
    thr = torch.rand(P, device=dev)
    side = torch.randn(P, a, 64, device=dev, dtype=torch.bfloat16)
    csk = torch.randn(P, a, R, device=dev, dtype=torch.float16)
    rho = torch.rand(P, a, device=dev)
    return q, kr, v, nk_len, thr, side, csk, rho, nk, a


def bench_prologue(S):
    q, kr, v, nk_len, thr, *_ = shapes(S)
    dev = "cuda"
    out = (
        q.new_empty(P, H),
        q.new_empty(P, 64, H, dtype=torch.bfloat16),
        q.new_empty(P, R, H, dtype=torch.float16),
        q.new_empty(P, H),
    )
    parts = tuple(torch.zeros(P, _NSPLIT, H, device=dev) for _ in range(3))
    sc = D.ATTN_SCALE

    def eager():
        skept = (q.to(torch.bfloat16) @ kr.transpose(1, 2)).float() * sc
        m1 = skept.max(-1).values
        p1 = torch.softmax(skept, -1)
        ent = -(p1 * p1.clamp_min(1e-9).log()).sum(-1)
        gate = ent > thr[:, None]
        qsk = torch.bmm(q[:, :, :KV], v.transpose(1, 2))
        qres = (q[:, :, :KV] - torch.bmm(qsk, v)).norm(dim=-1)
        torch.where(gate, m1, torch.tensor(float("inf"), device="cuda"))
        q[:, :, KV:].transpose(1, 2).contiguous().to(torch.bfloat16)
        qsk.transpose(1, 2).contiguous().half()
        return qres

    single = lambda: fused_prologue(q, kr, v, nk_len, thr, sc, out=out)  # noqa: E731
    # fused signature reads queries in place from the stacked qbuf
    qbuf = q.unsqueeze(0).to(torch.bfloat16).contiguous()  # [1, P, H, 576]
    li = torch.zeros(P, dtype=torch.int64, device=dev)
    slot = torch.arange(P, dtype=torch.int64, device=dev)
    split = lambda: fused_prologue_split(  # noqa: E731
        qbuf, li, slot, kr, v, nk_len, thr, sc, out=out, partials=parts
    )
    print(
        f"  prologue S={S // 1024}k: eager {t(eager):.3f}  single-CTA {t(single):.3f}  "
        f"split-NK {t(split):.3f} ms"
    )


def bench_scan(S):
    q, kr, v, nk_len, thr, side, csk, rho, nk, a = shapes(S)
    qs = torch.randn(64, H, device="cuda", dtype=torch.bfloat16).contiguous()
    qk = torch.randn(R, H, device="cuda", dtype=torch.float16).contiguous()
    qres = torch.rand(H, device="cuda")
    m1g = torch.full((H,), 5.0, device="cuda")
    out = torch.empty(a, dtype=torch.int32, device="cuda")
    f = lambda: vestige_scan(  # noqa: E731
        qs, qk, qres, m1g, side[0], csk[0], rho[0], D.ATTN_SCALE, 0.01, out=out
    )
    ms = t(f)
    mb = a * 260 / 1e6
    print(
        f"  scan(single-pair) S={S // 1024}k: {ms:.3f} ms  ({mb:.0f}MB -> {mb / ms:.0f} GB/s)"
    )


def bench_compact(S):
    _, _, _, _, _, _, _, _, nk, a = shapes(S)
    dev = "cuda"
    hit = (torch.rand(P, a, device=dev) < 0.01).to(torch.int32)
    arch = torch.arange(a, device=dev, dtype=torch.int64).repeat(P, 1)
    a_len = torch.full((P,), a, device=dev, dtype=torch.int64)
    li = torch.arange(P, device=dev)
    slot = torch.zeros(P, dtype=torch.int64, device=dev)
    W = 4096
    fb = torch.zeros(P, 1, W, dtype=torch.int64, device=dev)
    fl = torch.zeros(P, 1, dtype=torch.int64, device=dev)
    NB = (a + 1023) // 1024
    scr = (
        torch.zeros(P, NB, dtype=torch.int32, device=dev),
        torch.zeros(P, NB, dtype=torch.int32, device=dev),
        torch.zeros(P, dtype=torch.int32, device=dev),
    )
    scratch = torch.zeros(P, W + 1, dtype=torch.int64, device=dev)

    def torch_chain():
        h = hit != 0
        h.sum(-1).clamp(max=W)
        pos = torch.cumsum(h.to(torch.int64), -1) - 1
        dst = torch.where(h & (pos < W), pos, W)
        scratch.zero_()
        scratch.scatter_(1, dst, arch)

    tri = lambda: compact_fired(hit, arch, a_len, li, slot, fb, fl, scr)  # noqa: E731
    print(
        f"  compact S={S // 1024}k: torch-chain {t(torch_chain):.3f}  triton {t(tri):.3f} ms"
    )


if __name__ == "__main__":
    for S in (65536, 131072, 262144):
        print(f"S = {S // 1024}k")
        bench_prologue(S)
        bench_scan(S)
        bench_compact(S)
