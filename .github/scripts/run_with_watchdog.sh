#!/usr/bin/env bash
# Runs a command while logging cgroup memory and disk usage once a minute, and
# kills it with a diagnostic dump when memory or disk is nearly exhausted. An
# evicted/OOM-killed runner pod leaves no step log at all ("runner lost
# communication"), so failing early from inside the job is the only way the
# cause gets recorded.
set -uo pipefail

interval=${WATCHDOG_INTERVAL_SEC:-60}
min_disk_mb=${WATCHDOG_MIN_DISK_MB:-5120}
mem_pct=${WATCHDOG_MEM_KILL_PCT:-95}
paths=("$PWD" "${JULIA_DEPOT_PATH:-$HOME/.julia}" "${TMPDIR:-/tmp}")

# echoes "anon file limit" in bytes. `anon` is unreclaimable memory and is what
# actually trips a cgroup OOM; `file` is page cache, which the kernel reclaims.
mem_status() {
    if [ -r /sys/fs/cgroup/memory.stat ]; then
        local limit; limit=$(cat /sys/fs/cgroup/memory.max 2>/dev/null)
        echo "$(awk '/^anon /{print $2; exit}' /sys/fs/cgroup/memory.stat) $(awk '/^file /{print $2; exit}' /sys/fs/cgroup/memory.stat) ${limit:-max}"
    elif [ -r /sys/fs/cgroup/memory/memory.stat ]; then
        local limit; limit=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null)
        echo "$(awk '/^rss /{print $2; exit}' /sys/fs/cgroup/memory/memory.stat) $(awk '/^cache /{print $2; exit}' /sys/fs/cgroup/memory/memory.stat) ${limit:-max}"
    else
        free -b | awk '/Mem:/{print $3, 0, $2}'
    fi
}

avail_mb() { df -m --output=avail "$1" | tail -1 | tr -d ' '; }

# prints one status line; returns 1 if a kill threshold is crossed
check() {
    local anon file limit line reason=""
    read -r anon file limit < <(mem_status)
    if [ -z "$limit" ] || [ "$limit" = max ] || [ "$limit" -gt 1000000000000000 ]; then
        line="anon=$((anon / 1048576))MB file=$((file / 1048576))MB limit=unlimited"
    else
        line="anon=$((anon / 1048576))MB file=$((file / 1048576))MB limit=$((limit / 1048576))MB"
        [ $((anon * 100)) -ge $((limit * mem_pct)) ] && reason="anonymous memory at ${mem_pct}% of cgroup limit"
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

echo "[watchdog] nproc=$(nproc) $(free -h | awk '/Mem:/{print "host_mem="$2}') cgroup_limit=$(mem_status | cut -d' ' -f3)"
df -h "${paths[@]}" 2>/dev/null

"$@" &
cmd=$!
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
        exit 1
    fi
done
wait "$cmd"
status=$?
check || true
exit "$status"
