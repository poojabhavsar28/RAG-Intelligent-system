"""Run the cache layer-ablation leak experiment; ``--check`` exits non-zero if any row deviates."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tenantrag.cache.ablation import run_leak_ablation, to_markdown

ap = argparse.ArgumentParser()
ap.add_argument("--check", action="store_true")
a = ap.parse_args()
rows = run_leak_ablation()
Path("results").mkdir(exist_ok=True)
Path("results/cache_leak_ablation.md").write_text(to_markdown(rows) + "\n")
print(to_markdown(rows))
bad = [r for r in rows if not r["as_expected"]]
if a.check and bad:
    print("FAIL:", [r["injected_bug"] for r in bad])
    sys.exit(1)
