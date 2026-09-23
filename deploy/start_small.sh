#!/usr/bin/env bash
# 准备并启动 small profile（200/30/30）的 harness-only 实验。
#
#   bash deploy/start_small.sh
#
# 幂等：已建好的步骤会跳过，可重复执行。做七件事：
#   1. 前置检查（venv / improver.env / 旧 release / 实验目录未被占用）
#   2. 确定性重建 id200-val30-test30-v1（同一个 SPLIT_SEED）
#   3. 用硬链接填充 resources —— 不能用软链：
#      paths.py:29-33 的 local_tree 拒绝解析到 release 目录之外的符号链接，
#      而 ln -s ../id40-.../resources 正好指向兄弟目录，会被拒。
#      硬链接 is_symlink() 为假，且与原文件共享 inode，不占额外空间。
#   4. 重写 environment.lock.json 的 data_lake.files
#   5. 把 configs/local.json 指向新 release
#   6. 用上游自己的加载器做一次真实校验
#   7. nohup 启动，打印 pid 与监控命令
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

OLD_NAME=id40-val10-test10-v1
NEW_NAME=id200-val30-test30-v1
OLD="$ROOT/data/releases/$OLD_NAME"
NEW="$ROOT/data/releases/$NEW_NAME"
CONFIG=configs/h-only-small.toml
LOG="$ROOT/logs/h-only-small.log"
PY="$ROOT/.venv/bin/python"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
die() { printf '\n\033[31m%s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- 1. 检查
say "1/7 前置检查"
[ -x "$PY" ] || die "找不到 $PY，先跑 bash deploy/bootstrap.sh"
[ -f "$ROOT/$CONFIG" ] || die "找不到 $CONFIG，先 git pull"
[ -f "$HOME/.sciencebuddy/improver.env" ] \
    || die "找不到 ~/.sciencebuddy/improver.env，harness 第一次提案就会失败"
[ -d "$OLD/resources" ] || die "找不到 $OLD/resources，数据湖要从旧 release 复用"

EXP="$ROOT/runs/h-only-small-01"
[ -d "$EXP" ] && die "实验目录已存在：$EXP
loop.py:56 用 mkdir(exist_ok=False) 拒绝复用。改 $CONFIG 的 experiment，或删掉该目录。"

printf '  旧 release  %s（%s 个数据湖文件）\n' "$OLD_NAME" "$(ls -1 "$OLD/resources" | wc -l)"
printf '  可用磁盘    %s\n' "$(df -BG --output=avail "$ROOT" | tail -1 | tr -d ' ')"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader | sed 's/^/  GPU         /'

# ---------------------------------------------------------------- 2. 数据集
say "2/7 重建 release（profile=small）"
if [ -f "$NEW/manifest.json" ]; then
    echo "  已存在，跳过"
else
    "$PY" deploy/build_release.py --out "$NEW" --profile small \
        --cache "$ROOT/.cache/sources" 2>&1 | tail -14
fi
[ -f "$NEW/manifest.json" ] || die "release 构建失败"

# ---------------------------------------------------------------- 3. 数据湖
say "3/7 硬链接数据湖"
mkdir -p "$NEW/resources"
if [ "$(ls -1 "$NEW/resources" | wc -l)" -lt "$(ls -1 "$OLD/resources" | wc -l)" ]; then
    cp -al "$OLD/resources/." "$NEW/resources/" 2>/dev/null \
        || cp -a "$OLD/resources/." "$NEW/resources/"
fi
find "$NEW/resources" -type l -print -quit | grep -q . \
    && die "resources 里出现了符号链接，local_tree 会拒绝"
printf '  %s 个文件，占用 %s（与旧 release 共享 inode）\n' \
    "$(ls -1 "$NEW/resources" | wc -l)" "$(du -sh --apparent-size "$NEW/resources" | cut -f1)"

# ---------------------------------------------------------------- 4. lock
say "4/7 重写 environment.lock.json"
"$PY" deploy/update_lock.py --release "$NEW" --image "$ROOT/runtime.sif"

# ---------------------------------------------------------------- 5. 配置
say "5/7 指向新 release"
"$PY" - "$NEW_NAME" <<'PY'
import json, pathlib, sys
path = pathlib.Path("configs/local.json")
cfg = json.loads(path.read_text())
cfg["dataset"] = f"data/releases/{sys.argv[1]}"
path.write_text(json.dumps(cfg, indent=2) + "\n")
print("  dataset =", cfg["dataset"])
PY

# ---------------------------------------------------------------- 6. 校验
say "6/7 用上游加载器校验"
PYTHONPATH="$ROOT/src" SCIBUDDY_RUNTIME=apptainer "$PY" - "$CONFIG" <<'PY'
import sys, tomllib, pathlib
from simple_scibuddy.configuration import settings
from simple_scibuddy.data.dataset import TaskDataset   # loop.py:17 用的就是它

cfg = settings()
# 构造函数内部跑 local_tree + split_counts 校验，软链接或计数不符会在这里炸
data = TaskDataset(cfg["dataset"])
counts = {s: len(data.tasks(s)) for s in ("train", "val", "test")}
print("  split      ", counts)

want = tomllib.loads(pathlib.Path(sys.argv[1]).read_text())["harness_evolve"]["selection_tasks"]
if counts["val"] != want:
    sys.exit(f"  selection_tasks={want} 与 val={counts['val']} 不符，phase.py:126 会拒绝启动")

n = 1 + counts["test"] + 3 * (16 + 3 + 4 * counts["val"]) + counts["test"]
print(f"  预计 episode {n} 个（已完成那轮 174 个的 {n / 174:.1f} 倍）")
print("  数据湖     ", pathlib.Path(data.environment['data_lake']['path']).name)
PY

# ---------------------------------------------------------------- 7. 启动
say "7/7 启动"
mkdir -p "$ROOT/logs"
nohup bash deploy/run.sh "$CONFIG" > "$LOG" 2>&1 &
PID=$!
sleep 5
kill -0 "$PID" 2>/dev/null || { tail -30 "$LOG"; die "进程启动后立刻退出，日志见上"; }

cat <<EOF

  pid   $PID
  日志  $LOG

监控（任选）：
  tail -f $LOG
  grep -c 'Harness rollout' $LOG          # 已完成的 episode 数
  python3 deploy/analyze_episodes.py runs/h-only-small-01
  python3 deploy/estimate_runtime.py runs/h-only-small-01

停止：
  kill $PID

EOF
