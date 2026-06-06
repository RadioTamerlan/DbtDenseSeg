"""Plot DbtDenseSeg dense Dice on the external DBTex set, by view (CC / MLO /
combined), for the dense model and the ensemble. Patient-clustered bootstrap 95%
CI error bars. Reads dbtex_results.csv (this folder)."""
import csv, os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)
rows = list(csv.DictReader(open(os.path.join(HERE, "dbtex_results.csv"))))

def col(view, key):
    return np.array([float(r[key]) for r in rows if (view is None or r["view"] == view)])
def pats(view):
    return np.array([r["patient"] for r in rows if (view is None or r["view"] == view)])

def boot_ci(vals, patients, n=5000):
    g = {}
    for v, p in zip(vals, patients): g.setdefault(p, []).append(v)
    keys = list(g); arr = {k: np.array(v) for k, v in g.items()}
    b = np.array([np.concatenate([arr[k] for k in rng.choice(keys, len(keys), True)]).mean()
                  for _ in range(n)])
    return vals.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)

groups = [("CC", "CC"), ("MLO", "MLO"), ("Combined", None)]
methods = [("dense", "dice_dense", "#4C72B0"), ("ensemble", "dice_ensemble", "#55A868")]
x = np.arange(len(groups)); w = 0.36
fig, ax = plt.subplots(figsize=(8, 5))
for i, (label, key, color) in enumerate(methods):
    means, los, his = [], [], []
    for _, v in groups:
        m, lo, hi = boot_ci(col(v, key), pats(v))
        means.append(m); los.append(m - lo); his.append(hi - m)
    bars = ax.bar(x + (i - 0.5) * w, means, w, yerr=[los, his], capsize=5,
                  label=label, color=color)
    ax.bar_label(bars, fmt="%.3f", padding=6, fontsize=9)
ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups])
ax.set_ylabel("dense Dice"); ax.set_ylim(0, 1)
ax.set_title("DbtDenseSeg on external DBTex (100 patients)\nerror bars = 95% bootstrap CI (by patient)")
ax.legend(title="model"); plt.tight_layout()
out = os.path.join(HERE, "dbtex_performance.png")
plt.savefig(out, dpi=130); print("saved", out)
