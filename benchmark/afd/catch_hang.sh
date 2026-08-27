#!/usr/bin/env bash
# Run the concurrency case and, the moment the host stops making progress, ask it for stacks.
#
# py-spy cannot attach here -- ptrace_scope is 1 and /proc/sys is read-only, so it fails even
# under sudo. But PYTHONFAULTHANDLER=1 is set, and faulthandler dumps EVERY thread's Python stack
# on SIGABRT. The process is doomed once it hangs (the watchdog kills it at 300 s), so aborting it
# a little earlier costs nothing and is the only way to see where it is stuck.
set -uo pipefail
LOG=${AFD_HOST_LOG:?set AFD_HOST_LOG to the log the host is tee-ing into}
OUT=${AFD_STACKS_OUT:-./afd-stacks.txt}
QUIET_FOR=90        # seconds without a new log line that counts as stuck
GIVE_UP=280         # stay inside the 300 s watchdog

cd /home/user/experiment/sglang
source ./afd_env.sh

python benchmark/afd/stress.py --host http://127.0.0.1:31002 --case concurrency \
  > ${AFD_STRESS_OUT:-./afd-stress-concurrency.txt} 2>&1 &
STRESS=$!
echo "stress pid $STRESS"

began=$SECONDS
last_size=0
last_change=$SECONDS
while kill -0 "$STRESS" 2>/dev/null; do
    size=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
    if [ "$size" != "$last_size" ]; then
        last_size=$size
        last_change=$SECONDS
    fi
    quiet=$(( SECONDS - last_change ))
    if [ "$quiet" -ge "$QUIET_FOR" ]; then
        PID=$(ps -eo pid,args --no-headers | grep "sglang::scheduler" | grep -v grep \
              | awk '{print $1}' | tail -1)
        echo "host quiet for ${quiet}s -- aborting scheduler $PID for its stacks"
        kill -ABRT "$PID"
        sleep 8
        # faulthandler writes to stderr, which the tmux pane tees into the log
        awk '/Thread 0x|Current thread/{f=1} f' "$LOG" > "$OUT"
        echo "stacks -> $OUT ($(wc -l < "$OUT") lines)"
        exit 0
    fi
    if [ $(( SECONDS - began )) -ge "$GIVE_UP" ]; then
        echo "gave up at ${GIVE_UP}s without a quiet window"
        exit 2
    fi
    sleep 5
done
echo "the stress case finished on its own:"
cat ${AFD_STRESS_OUT:-./afd-stress-concurrency.txt}
