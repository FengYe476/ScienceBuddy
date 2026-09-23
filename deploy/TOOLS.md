# Runtime tools

A persistent Python 3.12 interpreter. State carries across `<execute>` blocks.
Print values to return them; bare expressions are not displayed.

## Installed libraries
| Library | Import | Use |
|---|---|---|
| numpy | `import numpy as np` | numerical arrays |
| pandas | `import pandas as pd` | tables; `read_parquet`, `read_csv` |
| pyarrow | `import pyarrow.parquet as pq` | parquet schemas without loading rows |
| BioPython | `from Bio import SeqIO, Seq` | sequences, translation, alignment |
| scipy | `import scipy` | statistics |
| requests / BeautifulSoup | `import requests`, `from bs4 import BeautifulSoup` | no network in this container |

## Filesystem
| Path | Contents |
|---|---|
| `/workspace/assets` | Task files, always includes `task_prompt.txt` |
| `/opt/data/biomni_data/data_lake` | Read-only scientific data lake, see DATA.md |
| `/opt/scitrace` | This documentation |

## Notes
- No network access. Everything must come from the mounted paths.
- Tool output is truncated to a token budget; print narrow selections or
  aggregates rather than whole files.
