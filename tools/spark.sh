#!/usr/bin/env bash
# Job control for the two DGX Sparks. Run from the workstation (Git Bash on Windows works).
#
#   tools/spark.sh sync                           both nodes to origin/main; fails if they differ
#   tools/spark.sh status                         commit, GPU, memory and every job on each node
#   tools/spark.sh run [--after JOB] NODE NAME 'COMMAND'
#                                                 start COMMAND detached on NODE as job NAME
#   tools/spark.sh shards NAME FIRST_SEED COUNT 'GENERATOR COMMAND'
#                                                 COUNT shards, seeds dealt round-robin to the nodes
#   tools/spark.sh collect NAME NODE              bring every node's shards of NAME to NODE,
#                                                 over the direct link when it has addresses
#
# Inside COMMAND, $P is the node's Python. In a generator command {seed} and {out} are
# filled in per shard. Quote commands with single quotes so $P expands on the node:
#
#   tools/spark.sh shards m9 500 10 \
#     '$P benchmarks/generate_m2.py --N 1000000 --seed {seed} --no-networks --out {out}'
#   tools/spark.sh collect m9 spark
#   tools/spark.sh run spark train_m9 \
#     '$P benchmarks/npe_m2.py --data "data/m9_shards/train_m9_s*.npz" --out data/npe_m9'
#   tools/spark.sh run spark2 sbc_m9 '$P benchmarks/sbc_m2.py --posterior data/npe_m9.pt ...'
#
# Conventions, each one bought by a failure:
#   - a job is data/NAME.sh on the node, detached from the ssh session, with data/NAME.log
#     and data/NAME.pid beside it;
#   - Python runs unbuffered, so a day-long run logs as it goes and not when it ends;
#   - the log is started afresh on every launch (the last one is kept as NAME.log.prev), so
#     a watcher cannot match a marker left by an earlier run;
#   - one GPU job per node. Training and SBC together exhausted a node's unified memory on
#     2026-09-15. `run` refuses a second job on a busy node unless it is queued behind the
#     first with --after, or SPARK_FORCE=1 is set for a job known to be small.
#   - `sync` before a paired run: the nodes are git checkouts, and a comparison whose arms
#     ran different code is not a comparison. Shards and estimators record host and commit.
set -euo pipefail

read -r -a NODES <<< "${SPARK_NODES:-spark spark2}"
REPO="${SPARK_REPO:-saom-amortized}"   # relative to the remote home directory
PY="${SPARK_PY:-spark-env/bin/python}" # likewise
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10)

die() { echo "spark.sh: $*" >&2; exit 1; }
# -n: ssh must not read stdin, or it swallows the job body that `launch` is handed there
on() { local node=$1; shift; "${SSH[@]}" -n "$node" "cd ~/$REPO && $*"; }

# names of the jobs still running on a node, one per line
alive() {
  on "$1" 'for f in data/*.pid; do [ -f "$f" ] || continue
             kill -0 "$(cat "$f")" 2>/dev/null && basename "$f" .pid; done; true'
}

# a node's address on the cable that joins the two Sparks; empty if it has none
link_ip() {
  on "$1" "ip -4 -o addr show ${SPARK_LINK_IF:-enp1s0f1np1} 2>/dev/null | awk '{print \$4}' | cut -d/ -f1"
}

cmd_sync() {
  local first="" c
  for node in "${NODES[@]}"; do
    c=$(on "$node" 'git pull -q --ff-only && git rev-parse --short=12 HEAD &&
                    git status --porcelain --untracked-files=no | wc -l') ||
      die "$node: pull failed"
    set -- $c
    echo "$node  $1  ($2 modified tracked files)"
    [ "$2" -eq 0 ] || die "$node has local modifications; a run there would not be at $1"
    [ -z "$first" ] && first=$1
    [ "$1" = "$first" ] || die "nodes are at different commits"
  done
}

cmd_status() {
  for node in "${NODES[@]}"; do
    echo "== $node"
    on "$node" '
      echo "  commit $(git rev-parse --short=12 HEAD 2>/dev/null || echo none)," \
           "gpu $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader)," \
           "$(free -g | awk "NR==2{print \$7}") GB free"
      for f in data/*.pid; do [ -f "$f" ] || continue
        name=$(basename "$f" .pid)
        kill -0 "$(cat "$f")" 2>/dev/null || continue
        echo "  running  $name: $(tail -n 1 "data/$name.log" 2>/dev/null | cut -c1-110)"
      done' || echo "  unreachable"
  done
}

# launch NODE NAME AFTER  -- job body on stdin
launch() {
  local node=$1 name=$2 after=$3 busy body
  body=$(cat)
  busy=$(alive "$node")
  if grep -qx "$name" <<< "$busy"; then die "$name is already running on $node"; fi
  if [ -n "$after" ]; then
    grep -qx "$after" <<< "$busy" || die "--after $after: no such job running on $node"
  elif [ -n "$busy" ] && [ -z "${SPARK_FORCE:-}" ]; then
    die "$node is busy with: $(echo $busy). Queue with --after JOB, or SPARK_FORCE=1"
  fi
  {
    echo '#!/bin/bash'
    echo "cd ~/$REPO"
    echo "P=~/$PY"
    echo 'export PYTHONUNBUFFERED=1'
    if [ -n "$after" ]; then
      echo "while kill -0 \"\$(cat data/$after.pid)\" 2>/dev/null; do sleep 60; done"
    fi
    echo "echo \"=== $name START \$(date) on \$(hostname) at \$(git rev-parse --short=12 HEAD)\""
    printf '%s
' "$body"
    echo "echo \"=== $name EXIT \$? \$(date)\""
  } | "${SSH[@]}" "$node" "mkdir -p ~/$REPO/data && cat > ~/$REPO/data/$name.sh"
  on "$node" "[ -f data/$name.log ] && mv data/$name.log data/$name.log.prev
              setsid nohup bash data/$name.sh > data/$name.log 2>&1 < /dev/null &
              echo \$! > data/$name.pid; echo \"$node: started $name, pid \$(cat data/$name.pid)\""
}

cmd_run() {
  local after=""
  if [ "${1:-}" = "--after" ]; then after=$2; shift 2; fi
  [ $# -eq 3 ] || die "usage: run [--after JOB] NODE NAME 'COMMAND'"
  launch "$1" "$2" "$after" <<< "$3"
}

cmd_shards() {
  [ $# -eq 4 ] || die "usage: shards NAME FIRST_SEED COUNT 'GENERATOR COMMAND'"
  local name=$1 first=$2 count=$3 gen=$4 i k seeds
  [[ $gen == *"{seed}"* && $gen == *"{out}"* ]] || die "the command needs {seed} and {out}"
  cmd_sync
  for k in "${!NODES[@]}"; do
    seeds=""
    for ((i = 0; i < count; i++)); do
      if ((i % ${#NODES[@]} == k)); then seeds+=" $((first + i))"; fi
    done
    [ -n "$seeds" ] || continue
    gen_k=${gen//\{out\}/\$f}
    gen_k=${gen_k//\{seed\}/\$s}
    launch "${NODES[$k]}" "gen_$name" "" <<EOF
mkdir -p data/${name}_shards
for s in$seeds; do
  f=data/${name}_shards/train_${name}_s\$s.npz
  [ -f \$f ] && continue
  echo "=== shard \$s \$(date)"
  $gen_k || { echo "=== shard \$s FAILED"; exit 1; }
done
echo "=== $name SHARDS DONE on \$(hostname): \$(ls data/${name}_shards | wc -l) files"
EOF
  done
}

cmd_collect() {
  [ $# -eq 2 ] || die "usage: collect NAME NODE"
  local name=$1 to=$2 node dst_ip src_ip port
  dst_ip=$(link_ip "$to")
  for node in "${NODES[@]}"; do
    [ "$node" = "$to" ] && continue
    if grep -qx "gen_$name" <<< "$(alive "$node")"; then
      die "gen_$name is still running on $node"
    fi
    src_ip=$(link_ip "$node")
    if [ -n "$dst_ip" ] && [ -n "$src_ip" ]; then
      # straight down the 200 Gb/s cable: a one-shot listener on the link address, so the
      # nodes still need no credentials for each other
      port=$((47000 + RANDOM % 1000))
      "${SSH[@]}" -n "$to" "cd ~/$REPO && nc -l $dst_ip $port | tar -xf -" &
      sleep 2
      on "$node" "tar -cf - data/${name}_shards | nc -N $dst_ip $port"
      wait $!
    else
      # no link addresses: relay through this machine
      "${SSH[@]}" -n "$node" "cd ~/$REPO && tar -cf - data/${name}_shards" |
        "${SSH[@]}" "$to" "cd ~/$REPO && tar -xf -"
    fi
  done
  on "$to" "ls data/${name}_shards/*.npz | wc -l | sed 's/\$/ shards on $to/'"
}

case "${1:-}" in
  sync | status | run | shards | collect) c=$1; shift; "cmd_$c" "$@" ;;
  *) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
