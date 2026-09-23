"""从 Biomni 的文件描述 + 实际文件结构生成 DATA.md（容器内模型看到的数据字典）。"""
import argparse, csv, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--lake", required=True)
ap.add_argument("--desc", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

LAKE = Path(a.lake)
desc = json.loads(Path(a.desc).read_text())

rows = []
for path in sorted(LAKE.iterdir()):
    if not path.is_file() or path.name.endswith(".part"):
        continue
    name, size, cols = path.name, path.stat().st_size, ""
    try:
        if name.endswith(".parquet"):
            import pyarrow.parquet as pq
            cols = ", ".join(pq.read_schema(path).names[:14])
        elif name.endswith(".csv"):
            with path.open(newline="") as f:
                cols = ", ".join(next(csv.reader(f))[:14])
        elif name.endswith(".tsv"):
            with path.open(newline="") as f:
                cols = ", ".join(next(csv.reader(f, delimiter="\t"))[:14])
        elif name.endswith(".pkl"):
            import pandas as pd
            o = pd.read_pickle(path)
            cols = ", ".join(map(str, list(o.columns)[:14])) if hasattr(o, "columns") else type(o).__name__
        elif name.endswith(".json"):
            v = json.loads(path.read_text())
            cols = ", ".join(sorted(v)[:14]) if isinstance(v, dict) else "array"
    except Exception as exc:
        cols = "(读取失败: %s)" % type(exc).__name__
    rows.append((name, size, desc.get(name, ""), cols))

out = ["# Data lake", "", "Read-only at `/opt/data/biomni_data/data_lake`.",
       "%d files. Load with pandas: `pd.read_parquet(path)` / `pd.read_csv(path)`." % len(rows),
       "Inspect a schema without loading rows: `pyarrow.parquet.read_schema(path).names`.",
       "", "---", ""]
for name, size, d, cols in rows:
    out += ["## " + name,
            "`/opt/data/biomni_data/data_lake/%s` - %.1f MB" % (name, size / 1e6)]
    if d:
        out += ["", d.strip()]
    if cols:
        out += ["", "Columns: " + cols]
    out.append("")
Path(a.out).write_text("\n".join(out))
print("  DATA.md: %d 个文件条目" % len(rows))
