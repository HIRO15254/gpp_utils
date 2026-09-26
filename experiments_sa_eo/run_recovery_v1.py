"""復旧 run 計画 v1（RECOVERY_PLAN_v1.md）を A → C → B の順に実行する。

使い方（リポジトリのどこから実行してもよい）:
    python experiments_sa_eo/run_recovery_v1.py --check        # 実行前の確認だけ（計算しない）
    python experiments_sa_eo/run_recovery_v1.py                # A → C → B を実行
    python experiments_sa_eo/run_recovery_v1.py --stages C,B   # 一部の段階だけ（順番は A → C → B のまま）
    python experiments_sa_eo/run_recovery_v1.py --gpp PATH     # 検証済みのビルド済み gpp を使う

計算を始める前に次を確かめ、1 つでも違えば計算せずに止まる。
    1. 通常は gpp の release ビルドを現在のソースから作る
       （cargo build --release --locked --bin gpp）。--gpp 指定時はそのビルドを使い、自動ビルドしない
    2. 仕様ファイルが make_baseline_v1.py・make_recovery_v1.py の出力と一致する
    3. gpp validate の batch_id とジョブ数が、この計画の記録（STAGES）と一致する
各段階は --rounds で流す（シードごとに全条件を終えてから次のシードへ進む）。中断しても同じコマンドで続きから
再開できる。保存先にその段階の batch があれば gpp resume を使い、書きかけの結果ファイルも回復する。
ある段階で失敗・中断が起きたら、後の段階には進まない。進み具合と失敗した run はログ（既定 data/logs/recovery_v1/）に
残す。失敗理由は段階ごとに最初の 3 件まで gpp inspect で引いて記録する。
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path

import make_baseline_v1
import make_recovery_v1

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
GPP = REPO / "target" / "release" / ("gpp.exe" if os.name == "nt" else "gpp")

# 段階ごとの (仕様ファイル, batch_id, ジョブ数)。仕様を変えたら RECOVERY_PLAN_v1.md と一緒に更新する。
STAGES = {
    "A": (make_baseline_v1.SPEC_PATH, "651e26afb279e0bf302a05cbf30dc52aa57f4854193ea03a72f5549b0d19cc83", 163_584),
    "C": (make_recovery_v1.C_PATH, "c27cf7e92f29ece132de9f5ddc78fd3b95758b7661ba5b20920dd234d572af21", 843_264),
    "B": (make_recovery_v1.B_PATH, "dec0166f5feebfd0f0c47ec1579b4c3881ba9a2c9f93678fc9f887d6dff094b8", 373_248),
}
ORDER = ("A", "C", "B")
EVENT = re.compile(r"^([0-9a-f]{64}) seed=(\d+) (\S+)$")
FAILURE_DETAILS = 3  # 理由をログに書く失敗 run の数（段階ごと）


class Log:
    def __init__(self, path: Path | None):
        self.fh = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.fh = path.open("a", encoding="utf-8")

    def __call__(self, msg: str) -> None:
        line = f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
        print(line, flush=True)
        if self.fh is not None:
            self.fh.write(line + "\n")
            self.fh.flush()


def hms(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def build(log: Log) -> bool:
    log("gpp の release ビルドを確認: cargo build --release --locked --bin gpp")
    done = subprocess.run(["cargo", "build", "--release", "--locked", "--bin", "gpp"], cwd=REPO)
    if done.returncode or not GPP.exists():
        log(f"ビルドに失敗した（終了コード {done.returncode}）")
        return False
    return True


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def specs_match(log: Log) -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        make_baseline_v1.write_spec(tmp / "A.toml")
        make_recovery_v1.write_specs(tmp / "C.toml", tmp / "B.toml")
        bad = [s for s in ORDER if (tmp / f"{s}.toml").read_bytes() != STAGES[s][0].read_bytes()]
    for s in bad:
        log(f"段階 {s}: {STAGES[s][0].name} が生成スクリプトの出力と違う。生成し直すか、変更を確かめること")
    return not bad


def validate(stage: str, gpp: Path, log: Log) -> bool:
    spec, batch_id, jobs = STAGES[stage]
    done = subprocess.run([str(gpp), "validate", str(spec), "--json"], cwd=REPO, capture_output=True,
                          text=True, encoding="utf-8", errors="replace")
    if done.returncode:
        log(f"段階 {stage}: gpp validate が失敗した\n{done.stderr.strip()}")
        return False
    got = json.loads(done.stdout)
    if got.get("batch_id") != batch_id or got.get("jobs") != jobs:
        log(f"段階 {stage}: 記録と違う（batch_id {got.get('batch_id')}, jobs {got.get('jobs')}）。"
            f"記録は batch_id {batch_id}, jobs {jobs}。条件 ID が変わると A の結果を再利用できない")
        return False
    return True


def failure_reason(gpp: Path, root: Path, batch_id: str, condition_id: str, seed: str) -> str:
    done = subprocess.run(
        [str(gpp), "inspect", "--batch", batch_id, "--root", str(root), "--condition", condition_id,
         "--seed", seed, "--json"],
        cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    try:
        error = json.loads(done.stdout)["jobs"][0]["error"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return f"（gpp inspect で取得できなかった: {(done.stderr.strip() or done.stdout.strip())[:300]}）"
    return f"{error['code']}: {error['message']}" if error else "（理由の記録なし）"


def run_stage(stage: str, gpp: Path, root: Path, args: argparse.Namespace, log: Log) -> int:
    spec, batch_id, jobs = STAGES[stage]
    resume = (root / "batches" / batch_id).exists()
    cmd = [str(gpp)]
    cmd += ["resume", "--batch", batch_id] if resume else ["run", str(spec)]
    cmd += ["--root", str(root), "--threads", str(args.threads), "--rounds"]
    if args.deadline_seconds is not None:
        cmd += ["--deadline-seconds", str(args.deadline_seconds)]
    log(f"段階 {stage} を{'再開' if resume else '開始'}: {spec.name}（{jobs:,} ジョブ、batch {batch_id[:8]}）")
    log("  " + " ".join(cmd))
    start = last = time.monotonic()
    counts: Counter[str] = Counter()
    failed: list[tuple[str, str]] = []

    def handle(line: str) -> None:
        nonlocal last
        m = EVENT.match(line)
        if not m:
            log(f"  gpp: {line}")
            return
        condition_id, seed, status = m.groups()
        counts[status] += 1
        if status not in ("completed", "reused"):
            log(f"  {line}")
        if status == "failed":
            failed.append((condition_id, seed))
        now = time.monotonic()
        if now - last >= args.progress_seconds:
            last = now
            done = counts["completed"] + counts["reused"]
            rate = counts["completed"] / (now - start)
            eta = hms((jobs - done) / rate) if rate > 0 else "?"
            log(f"  段階 {stage}: {done:,}/{jobs:,}（{100 * done / jobs:.1f}%）"
                f" 新規 {counts['completed']:,} 再利用 {counts['reused']:,} 失敗 {counts['failed']:,}"
                f" 経過 {hms(now - start)} 残り約 {eta}")

    proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", errors="replace")
    # stdout には最後に要約 JSON が 1 回出るだけだが、パイプが詰まらないよう別スレッドで読み切る。
    out_parts: list[str] = []
    reader = threading.Thread(target=lambda: out_parts.append(proc.stdout.read()), daemon=True)
    reader.start()
    interrupted = False
    while True:
        # Ctrl+C は gpp にも届き、gpp が実行中の run を片付けて終わる。その間も出力を読み続ける。
        try:
            for line in proc.stderr:
                handle(line.rstrip("\n"))
            break
        except KeyboardInterrupt:
            if interrupted:
                proc.kill()
                break
            interrupted = True
            log("中断を受け付けた。gpp が実行中の run を片付けて終わるのを待つ（もう一度 Ctrl+C で即時終了）")
    code = proc.wait()
    reader.join()
    out = "".join(out_parts)
    try:
        summary = json.loads(out)
        log(f"  結果: {json.dumps(summary, ensure_ascii=False)}")
    except json.JSONDecodeError:
        summary = {}
        if out.strip():
            log(f"  gpp の出力: {out.strip()}")
    # gpp run の進行表示には失敗理由が出ないので、最初の数件だけ gpp inspect で引いて記録する。
    for condition_id, seed in failed[:FAILURE_DETAILS]:
        log(f"  失敗理由 {condition_id[:8]} seed={seed}: "
            f"{failure_reason(gpp, root, batch_id, condition_id, seed)}")
    if len(failed) > FAILURE_DETAILS:
        log(f"  失敗は計 {len(failed):,} 件。残りの理由は gpp inspect --batch {batch_id} --root {root} --json で確かめる")
    log(f"段階 {stage} 終了: 終了コード {code}、経過 {hms(time.monotonic() - start)}")
    if code == 0 and summary.get("deadline_reached"):
        return -1
    return code


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="確認だけして計算しない")
    ap.add_argument("--gpp", type=Path,
                    help="検証済みのビルド済み gpp。指定時は cargo build を省略し、全操作にこの実行ファイルを使う")
    ap.add_argument("--stages", default=",".join(ORDER), help="実行する段階（既定 A,C,B）")
    ap.add_argument("--root", type=Path, default=REPO / "data" / "v1", help="結果の保存先（既定 data/v1）")
    ap.add_argument("--threads", type=int, default=12, help="並列数（既定 12）")
    ap.add_argument("--deadline-seconds", type=int, help="段階ごとの時間制限（秒）。gpp run にそのまま渡す")
    ap.add_argument("--progress-seconds", type=float, default=600.0, help="進み具合を記録する間隔（秒）")
    ap.add_argument("--log-dir", type=Path, default=REPO / "data" / "logs" / "recovery_v1")
    args = ap.parse_args()
    wanted = {s.strip().upper() for s in args.stages.split(",") if s.strip()}
    if not wanted or not wanted <= set(ORDER):
        ap.error(f"--stages には {','.join(ORDER)} から 1 つ以上選ぶ")
    stages = [s for s in ORDER if s in wanted]
    root = args.root.resolve()
    log = Log(None if args.check else args.log_dir / f"run_{dt.datetime.now():%Y%m%d-%H%M%S}.log")
    if args.gpp is None:
        gpp = GPP
    else:
        gpp = args.gpp.resolve()
        if not gpp.exists() or not gpp.is_file():
            ap.error(f"--gpp が実行ファイルを指していない: {gpp}")
        log(f"指定された gpp を使用（自動ビルドを省略）: {gpp}")
        log(f"指定された gpp の SHA-256: {file_sha256(gpp)}")

    # 一部の段階だけ流すときも、計画全体（A・C・B）の仕様と batch_id を確かめる。
    # 段階間で条件 ID がずれると、C・B が A の結果を再利用できなくなるため。
    ok = (build(log) if args.gpp is None else True) and specs_match(log)
    ok = ok and all([validate(s, gpp, log) for s in ORDER])
    if not ok:
        log("確認に失敗したので実行しない")
        return 2
    for s in ORDER:
        spec, batch_id, jobs = STAGES[s]
        started = (root / "batches" / batch_id).exists()
        log(f"段階 {s}: {spec.name} {jobs:,} ジョブ batch {batch_id[:8]} {'（開始済み）' if started else ''}")
    if args.check:
        log(f"確認はすべて通った。実行する段階: {' → '.join(stages)}、保存先 {root}、{args.threads} 並列")
        return 0

    log(f"実行する段階: {' → '.join(stages)}、保存先 {root}、{args.threads} 並列")
    for s in stages:
        code = run_stage(s, gpp, root, args, log)
        if code == -1:
            log("時間制限で止めた。同じコマンドで続きから再開できる")
            return 0
        if code != 0:
            log(f"段階 {s} が完了しなかったので、後の段階には進まない。同じコマンドで続きから再開できる")
            return code
    log("すべての段階が完了した")
    return 0


if __name__ == "__main__":
    sys.exit(main())
