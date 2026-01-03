#!/usr/bin/env bash
set -e
shopt -s nullglob

run_group () {
  local scripts=("$@")   # ✅ 直接拿到所有展开后的文件名
  echo "==== Running group: ${scripts[0]%_*}.sh ===="

  if [ ${#scripts[@]} -eq 0 ]; then
    echo "⚠️  No scripts matched"
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