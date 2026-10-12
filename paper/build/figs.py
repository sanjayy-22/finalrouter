import sys, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
out = sys.argv[1]

def fig(w, h):
    f = plt.figure(figsize=(w/100, h/100), dpi=100)
    ax = f.add_axes([0, 0, 1, 1]); ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.axis("off")
    return f, ax

def box(ax, x, y, w, h, text, bold=False, fs=17, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=10",
                 fc="white", ec="black", lw=1.4, ls="--" if dashed else "-"))
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.35)
    return (x, y, w, h)

def arrow(ax, p, q, rad=0.0, lw=1.6, style="->", ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=22, lw=lw,
                 color="black", connectionstyle=f"arc3,rad={rad}", ls=ls))

def label(ax, x, y, t, fs=14, it=True):
    ax.text(x, y, t, ha="center", va="center", fontsize=fs, style="italic" if it else "normal",
            bbox=dict(fc="white", ec="none", pad=1))

# ---------- Fig. 1: routing / feedback loop (one-column layout) ----------
f, ax = fig(1000, 1130)
L, R, W = 25, 535, 440
box(ax, L, 30, W, 110, "Incoming User Query", bold=True, fs=27)
box(ax, L, 215, W, 130, "Query Features\nlength, category, cues", fs=26)
box(ax, L, 420, W, 140, "LinUCB Router\npredicted reward\n+ exploration bonus", fs=24)
box(ax, L, 640, W, 230, "")
ax.text(L + W/2, 680, "Candidate LLM Pool (vLLM)", ha="center", va="center", fontsize=22, fontweight="bold")
for k, (nm, sz) in enumerate([("Small", "4B"), ("Medium", "8B"), ("Large", "14B")]):
    box(ax, L + 15 + k*140, 720, 130, 130, f"{nm}\nQwen3-\n{sz}", fs=21)
box(ax, R, 30, W, 150, "Update Chosen Model\nadd features and reward\nto its statistics", fs=23)
box(ax, R, 250, W, 130, "Reward Calculator\nquality − cost − latency", fs=23)
box(ax, R, 450, W, 130, "Evaluation & Feedback\nchecks, judge, user rating", fs=24)
box(ax, R, 670, W, 130, "Response + Measurements\nlatency, tokens, errors", fs=24)
box(ax, L, 960, 950, 130, "Response returned to user; every decision, outcome,\nreward and update written to the audit log", fs=23)
arrow(ax, (L + W/2, 140), (L + W/2, 215)); arrow(ax, (L + W/2, 345), (L + W/2, 420))
arrow(ax, (L + W/2, 560), (L + W/2, 640)); label(ax, L + W/2 + 85, 600, "select model", fs=20)
arrow(ax, (L + W, 735), (R, 735))
arrow(ax, (R + W/2, 670), (R + W/2, 580)); arrow(ax, (R + W/2, 450), (R + W/2, 380))
arrow(ax, (R + W/2, 250), (R + W/2, 180))
arrow(ax, (R, 150), (L + W, 470), ls="--")   # dashed: statistics feed future routing decisions
arrow(ax, (R + W/2, 800), (R + W/2, 960))
f.savefig(f"{out}/image1.png"); plt.close(f)

# ---------- Fig. 2: evaluation design ----------
f, ax = fig(1500, 700)
box(ax, 40, 60, 300, 120, "GSM8K Pilot\n300 test problems", bold=True)
box(ax, 40, 280, 300, 120, "Development split\n90 problems (warm-up)")
box(ax, 40, 500, 300, 120, "Held-out test split\n210 problems")
box(ax, 430, 60, 330, 120, "Run every problem on\n4B, 8B and 14B (FP8)")
box(ax, 430, 280, 330, 120, "Full-information log\nQ, C, L, failures per\n(query, model)")
box(ax, 430, 500, 330, 120, "Numeric answer check\n(1,319 / 1,319 GSM8K\nreferences reproduced)")
box(ax, 850, 50, 300, 270, "Routing Policies\nSmall-only · Large-only\nRandom · Rule-based\nLinUCB (proposed)", fs=16)
box(ax, 850, 400, 300, 220, "Replay / Simulation\nchosen model's logged\noutcome revealed as\nbandit feedback")
box(ax, 1230, 60, 240, 560, "Metrics\n\nQuality\nLatency (mean, p95)\nCost / request\nFailure rate\nCumulative reward\nRegret vs. oracle\nSelection share\n\n5 seeds,\nmean ± SD", fs=15)
arrow(ax, (190, 180), (190, 280)); arrow(ax, (140, 400), (140, 500))
arrow(ax, (340, 120), (430, 120)); arrow(ax, (595, 180), (595, 280)); arrow(ax, (595, 400), (595, 500))
arrow(ax, (760, 340), (850, 480)); arrow(ax, (1000, 320), (1000, 400))
arrow(ax, (1150, 510), (1230, 510)); label(ax, 640, 668, "Learning policies warm up on the development split; all reported numbers come from the held-out split", fs=13)
f.savefig(f"{out}/image2.png"); plt.close(f)

# ---------- Fig. 3: pilot results (from results/gsm8k_pilot_models_4090/results.json) ----------
import json, os
import numpy as np
res_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                              "../../results/gsm8k_pilot_models_4090/results.json")
R = json.load(open(res_path))
P = R["policies"]
plt.rcParams.update({"font.family": "serif", "font.size": 22})
f = plt.figure(figsize=(10, 12.4), dpi=100)
axa = f.add_axes([0.13, 0.585, 0.83, 0.375]); axb = f.add_axes([0.13, 0.07, 0.83, 0.375])
style = {"small_only": ("Small-only (= rule-based)", "-", "0.0", 2.6),
         "large_only": ("Large-only", "--", "0.0", 2.2),
         "random": ("Random", ":", "0.0", 2.6),
         "linucb": ("LinUCB (proposed)", "-", "0.55", 5.0)}
for k in ("linucb", "small_only", "large_only", "random"):
    lab, ls, c, lw = style[k]
    cum = np.cumsum(R["reward_traces"][k])
    axa.plot(np.arange(1, len(cum) + 1), cum, ls=ls, color=c, lw=lw, label=lab)
    if k != "small_only":   # small-only ends next to LinUCB; both values go on one label
        txt = f"{cum[-1]:.1f}" if k != "linucb" else \
            f"{np.sum(R['reward_traces']['small_only']):.1f} / {cum[-1]:.1f}"
        axa.text(len(cum) + 3, cum[-1], txt, va="center", fontsize=18)
axa.set_xlim(0, 285); axa.set_xlabel("Held-out query index (mean over 5 seeds)"); axa.set_ylabel("Cumulative reward")
axa.set_title("(a) Cumulative reward on the test stream", fontsize=19)
axa.legend(loc="lower right", frameon=False, fontsize=22)
for sp in ("top", "right"): axa.spines[sp].set_visible(False); axb.spines[sp].set_visible(False)
axa.grid(axis="y", color="0.85", lw=0.8)
pts = [("small_only", "Small-only /\nrule-based", "o", (12, -38)), ("large_only", "Large-only", "s", (-60, 18)),
       ("random", "Random", "^", (14, -8)), ("linucb", "LinUCB", "D", (14, 14))]
for k, lab, mk, off in pts:
    x = P[k]["mean_cost"]["mean"]; y = 100 * P[k]["mean_quality"]["mean"]
    xe = P[k]["mean_cost"]["std"]; ye = 100 * P[k]["mean_quality"]["std"]
    axb.errorbar(x, y, xerr=xe, yerr=ye, fmt=mk, ms=13, color="0.0" if k != "linucb" else "0.45",
                 mec="black", capsize=4, lw=1.4)
    axb.annotate(lab, (x, y), xytext=off, textcoords="offset points", fontsize=19)
axb.set_xlim(1.0, 5.2); axb.set_ylim(93.5, 96.5)
axb.set_xlabel("Cost per request (token-based proxy)"); axb.set_ylabel("Accuracy on held-out split (%)")
axb.set_title("(b) Accuracy versus cost per request", fontsize=22)
axb.grid(color="0.85", lw=0.8)
f.savefig(f"{out}/image3.png"); plt.close(f)
