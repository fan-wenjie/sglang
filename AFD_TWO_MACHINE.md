# Running the two sides on two machines

The arrangement's point is that the feed-forward leaves the host. Everything up to the second
machine can be faked; the interconnect cannot. This is the procedure, with the numbers the link
actually gave, so a later run knows whether it is comparing like with like.

## The two roles

    host   owns the KV cache, runs attention, sends every dense layer's feed-forward to the pool
    pool   owns the weights, runs the feed-forward for whoever calls, serves no tokens

Both load the whole model. The pool uses one layer's MLP at a time and could hold only those
weights; loading the stack is wasteful and correct, and a pool with its own weight-mapping code
would have its own weight-mapping bugs, which produce plausible tokens from the wrong weights.

## The machines these numbers came from

    host   RTX PRO 6000 Blackwell Workstation, 97 GB
    pool   RTX 5090, 32 GB -- the weights are 29 GB, so the pool runs at --mem-fraction-static
           0.95 with the state cache floored at one request's worth. It never holds a KV cache or
           a recurrent state; sglang allocates both on the way to a loaded model anyway.

Link, measured rather than assumed:

    TCP round trip          0.3 ms      (connect+reply on the internal address, median 0.58 ms)
    bulk bandwidth          4.4 Gbit/s  (3.2 GB streamed)
    per-layer pool call     6.5 ms      steady state, decode, batch of 4

The gap between 0.3 ms and 6.5 ms is the point: the network is not what a pool call costs. It is
the device-to-host copy, the host-to-device copy on the other side, and Python on both.

## Setup

The pool needs the host's environment, not a similar one -- two sides on different wheels are two
models. Reproduce it, do not resolve it:

    python -m pip freeze | grep -v '^-e ' > reqs.txt        # keep sglang-kernel; see below
    # split out the 13.3 nvidia wheels, install the rest, then force those over the top:
    python -m pip install -r reqs_base.txt
    python -m pip install --no-deps -r reqs_forced.txt
    python -m pip install sglang-kernel==0.4.6.post1
    grep -m1 CUDART_VERSION $CU13/include/cuda_runtime_api.h  # must read 13030

Two traps, both of which cost a rebuild here:

  * `cuda-toolkit` is a dependency of torch and pins `nvidia-cuda-nvrtc` to the 13.0 line, while
    sm_120a needs 13.3. pip cannot resolve the host's freeze because the host's own state is that
    contradiction, forced. Reproduce the force.
  * a freeze filtered with `grep -v '^sglang'` silently drops `sglang-kernel`, which is a
    different package. The install reports success and the pool fails at its first quantised layer.

Copy the weights rather than downloading them twice -- 29 GB over the internal link takes about a
minute, and both sides then demonstrably hold the same files.

## Running

The pool must be under tmux: a foreground service blocks the container's terminal, and a job
backgrounded through ssh hangs the ssh client rather than the job. Through `afd_run.sh`, because
`source` does not cross tmux any more than it crosses nohup.

    # on the pool machine
    tmux new-session -d -s pool "bash ~/afd/start_pool.sh 0.95 8999 1 5 > ~/afd/pool.log 2>&1"

    # on the host machine, at the pool's INTERNAL address
    ./afd_run.sh python -m sglang.launch_server --model-path <path> \
        --afd-mode host --afd-pool-addr 172.20.19.32:8999 \
        --afd-q-shift-layers 1 --afd-coverage all --attention-backend triton --disable-cuda-graph

`--afd-min-batch 1` on the pool when one host calls it. A pool that waits for a partner that never
arrives spends `--afd-max-wait-ms` on every call, and the arm is then mostly the timeout.

## Reading the host's log

    afd host: N pool call(s); last 512: mean outstanding X ms, mean blocked Y ms, hidden Z%

Windowed, not cumulative: the pool JIT-compiles on its first frames, and those calls ran two
orders of magnitude slower than steady state -- 188 ms against 6.7 ms -- which a running mean
carries forever. `hidden` is the fraction of the round trip the host spent launching the sweep
rather than waiting. It is a floor on the overlap, not the overlap: the sweep's kernels run on the
GPU during the socket wait, and this number cannot see that.
