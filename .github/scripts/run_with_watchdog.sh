#!/usr/bin/env bash
# Runs a command while logging cgroup memory and disk usage every few seconds,
# and kills it with a diagnostic dump when unreclaimable memory or disk is
# nearly exhausted. An evicted/OOM-killed runner pod leaves no step log at all
# ("runner lost communication"), so failing early from inside the job is the
# only way the cause gets recorded.
set -uo pipefail

interval=${WATCHDOG_INTERVAL_SEC:-10}
min_disk_mb=${WATCHDOG_MIN_DISK_MB:-5120}
mem_pct=${WATCHDOG_MEM_KILL_PCT:-90}
paths=("$PWD" "${JULIA_DEPOT_PATH:-$HOME/.julia}" "${TMPDIR:-/tmp}")

# echoes "unreclaimable anon file limit" in bytes. Clean page cache is
# reclaimable, so the OOM risk is anon + shmem + dirty/writeback + unevictable;
# git checkouts and large cp's fill the cgroup with dirty file pages faster
# than throttled pod storage can write them back, which is what OOMs the pod.
mem_status() {
    if [ -r /sys/fs/cgroup/memory.stat ]; then
        local limit; limit=$(cat /sys/fs/cgroup/memory.max 2>/dev/null)
        awk -v lim="${limit:-max}" '
            /^anon /{a=$2} /^file /{f=$2} /^shmem /{s=$2}
            /^file_dirty /{d=$2} /^file_writeback /{w=$2} /^unevictable /{u=$2}
            END{print a+s+d+w+u, a, f, lim}
        ' /sys/fs/cgroup/memory.stat
    elif [ -r /sys/fs/cgroup/memory/memory.stat ]; then
        local limit; limit=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null)
        awk -v lim="${limit:-max}" '
            /^rss /{a=$2} /^cache /{f=$2} /^shmem /{s=$2}
            /^dirty /{d=$2} /^writeback /{w=$2} /^unevictable /{u=$2}
            END{print a+s+d+w+u, a, f, lim}
        ' /sys/fs/cgroup/memory/memory.stat
    else
        free -b | awk '/Mem:/{print $3, $3, 0, $2}'
    fi
}

avail_mb() { df -m --output=avail "$1" | tail -1 | tr -d ' '; }

# prints one status line; returns 1 if a kill threshold is crossed
check() {
    local unrec anon file limit line reason=""
    read -r unrec anon file limit < <(mem_status)
    if [ -z "$limit" ] || [ "$limit" = max ] || [ "$limit" -gt 1000000000000000 ]; then
        line="unreclaimable=$((unrec / 1048576))MB anon=$((anon / 1048576))MB file=$((file / 1048576))MB limit=unlimited"
    else
        line="unreclaimable=$((unrec / 1048576))MB anon=$((anon / 1048576))MB file=$((file / 1048576))MB limit=$((limit / 1048576))MB"
        if [ $((unrec * 100)) -ge $((limit * mem_pct)) ]; then
            reason="unreclaimable memory at ${mem_pct}% of cgroup limit"
        fi
    fi
    for p in "${paths[@]}"; do
        [ -e "$p" ] || continue
        local avail; avail=$(avail_mb "$p")
        line="$line | $p avail=${avail}MB"
        [ "$avail" -lt "$min_disk_mb" ] && reason="disk under ${min_disk_mb}MB available at $p"
    done
    echo "[watchdog $(date -u +%H:%M:%S)] $line"
    [ -z "$reason" ] && return 0
    echo "::error::watchdog: $reason; killing the build so the log survives"
    return 1
}

echo "[watchdog] nproc=$(nproc) $(free -h | awk '/Mem:/{print "host_mem="$2}') cgroup_limit=$(mem_status | cut -d' ' -f4)"
df -h "${paths[@]}" 2>/dev/null

# keep dirty file pages draining instead of letting them accumulate to the
# cgroup ceiling in bursts; sync blocks until writeback finishes, so this loop
# effectively applies continuous writeback pressure
(while sleep 5; do sync; done) &
syncer=$!

"$@" &
cmd=$!
rc=0
while sleep "$interval"; do
    # kill -0 still succeeds on an unreaped zombie; the ps check catches that
    { ! kill -0 "$cmd" 2>/dev/null || [[ $(ps -o stat= -p "$cmd" 2>/dev/null) == Z* ]]; } && break
    if ! check; then
        free -m
        df -h
        for p in "${paths[@]}"; do [ -e "$p" ] && timeout 60 du -xm --max-depth=2 "$p" 2>/dev/null | sort -rn | head -25; done
        kill -TERM "$cmd" 2>/dev/null
        sleep 15
        kill -KILL "$cmd" 2>/dev/null
        wait "$cmd"
        rc=1
        break
    fi
done
kill "$syncer" 2>/dev/null
[ "$rc" = 1 ] && exit 1
wait "$cmd"
status=$?
check || true
exit "$status"
