#!/usr/bin/env bash
# SA/EO x Flip/Swap 比較実験（iter7）のラウンドを順に流す。
#
# 1 ラウンド = 実行シード 1 本ぶん（1 条件あたり +4 試行）。ラウンド境界で経過時間を見て
# デッドラインを超えていたら停止するので、**どこで止めても全条件が同じ試行数**になる。
#
# 使い方:
#   experiments_sa_eo/run_rounds.sh [--threads N] [--deadline-hours H] [--from K] [--to K]
#
# 例（本番）:
#   nohup experiments_sa_eo/run_rounds.sh --threads 14 --deadline-hours 46 \
#         > experiments_sa_eo/logs/run.log 2>&1 &
set -euo pipefail

threads=14
deadline_hours=46
from_round=0
to_round=31

while [ "$#" -gt 0 ]; do
    case "$1" in
        --threads) threads="$2"; shift 2 ;;
        --deadline-hours) deadline_hours="$2"; shift 2 ;;
        --from) from_round="$2"; shift 2 ;;
        --to) to_round="$2"; shift 2 ;;
        *) echo "不明な引数: $1" >&2; exit 2 ;;
    esac
done

batch_dir="experiments_sa_eo/batches"
log_dir="experiments_sa_eo/logs"
mkdir -p "$log_dir"

if [ ! -d "$batch_dir" ]; then
    echo "バッチ定義がありません。先に make_batches.py を実行してください: $batch_dir" >&2
    exit 1
fi

start_ts=$(date +%s)
deadline_s=$(awk -v h="$deadline_hours" 'BEGIN{printf "%d", h*3600}')

echo "=== 開始 $(date) / threads=$threads / deadline=${deadline_hours}h / rounds ${from_round}..${to_round} ==="

for k in $(seq "$from_round" "$to_round"); do
    kk=$(printf "%02d" "$k")
    elapsed=$(( $(date +%s) - start_ts ))
    if [ "$elapsed" -ge "$deadline_s" ]; then
        echo "=== デッドライン到達（経過 $((elapsed/3600))h）。ラウンド $kk は開始しない ==="
        break
    fi

    # SA を先に（安いので応答曲線が早く埋まる）→ EO。
    for tag in SA EO; do
        bf="$batch_dir/round${kk}__${tag}.json"
        [ -e "$bf" ] || { echo "スキップ（無い）: $bf"; continue; }
        echo "--- round $kk / $tag  (経過 $((elapsed/60)) 分) ---"
        ./target/release/cli.exe --batch "$bf" \
            --out data/results_sa_eo --graphs data/graphs --threads "$threads" \
            2>&1 | tail -1
    done
    echo "=== round $kk 完了 $(date) （1 条件あたり $(( (k+1) * 4 )) 試行）==="
done

echo "=== 終了 $(date) / 総経過 $(( ($(date +%s) - start_ts) / 60 )) 分 ==="
