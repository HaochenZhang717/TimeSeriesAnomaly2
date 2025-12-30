#!/usr/bin/env bash
set -e
shopt -s nullglob

#cd NN_evaluate

run_group () {
  local pattern=$1
  echo "==== Running group: $pattern ===="

  local scripts=($pattern)   # ⭐ 关键：这里展开 glob

  if [ ${#scripts[@]} -eq 0 ]; then
    echo "⚠️  No scripts matched $pattern"
    return
  fi

  for s in "${scripts[@]}"; do
    echo "▶ Running $s"
    bash "$s" > "${s%.sh}.log" 2>&1
  done
}

run_group nn_eval_*_mitdb.sh &
run_group nn_eval_*_qtdb.sh &
run_group nn_eval_*_svdb.sh &

wait
echo "✅ All dataset groups finished."