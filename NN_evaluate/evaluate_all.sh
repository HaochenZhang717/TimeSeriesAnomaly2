#!/usr/bin/env bash
set -e

cd NN_evaluate

run_group () {
  local pattern=$1
  echo "==== Running group: $pattern ===="
  for s in $pattern; do
    bash "$s" > "${s%.sh}.log" 2>&1
  done
}

run_group "nn_eval_*_mitdb.sh" &
run_group "nn_eval_*_qtdb.sh" &
run_group "nn_eval_*_svdb.sh" &

wait
echo "✅ All dataset groups finished."