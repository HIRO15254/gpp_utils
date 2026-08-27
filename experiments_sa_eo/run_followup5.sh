#!/usr/bin/env bash
# 追試5: EO の τ を 0.00 まで、SA の Θ を +1.50 まで 0.05 刻みで延長。
# 1 ラウンド = 実行シード 1 本（途中で止めても全条件が同数）。SA を先に流す。
set -euo pipefail
threads="${1:-14}"
bd="experiments_sa_eo/batches_followup"
echo "=== 追試5 開始 $(date) / threads=$threads ==="
for k in $(seq -w 0 31); do
    for tag in SA EO; do
        bf="$bd/f5_round${k}__${tag}.json"
        [ -e "$bf" ] || { echo "スキップ: $bf"; continue; }
        echo "--- round $k / $tag $(date) ---"
        ./target/release/cli.exe --batch "$bf" --out data/results_sa_eo \
            --graphs data/graphs --threads "$threads" 2>&1 | tail -1
    done
    echo "=== round $k 完了 $(date) （1 条件あたり $(( (10#$k + 1) * 4 )) 試行）==="
done
echo "=== 追試5 終了 $(date) ==="
