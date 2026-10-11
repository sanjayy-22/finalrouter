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

# ---------- Fig. 1: routing / feedback loop ----------
f, ax = fig(1500, 850)
box(ax, 60, 60, 330, 100, "Incoming User Query", bold=True, fs=19)
box(ax, 60, 240, 330, 110, "Query / Context\nFeature Extraction  x_t")
box(ax, 60, 430, 330, 120, "LinUCB Router\nscore = θᵀx + α·√(xᵀA⁻¹x)")
box(ax, 505, 400, 450, 180, "")
ax.text(730, 425, "Candidate LLM Pool (vLLM)", ha="center", va="center", fontsize=16, fontweight="bold")
box(ax, 520, 455, 135, 95, "Small\nQwen3-4B", fs=14)
box(ax, 665, 455, 135, 95, "Medium\nQwen3-8B", fs=14)
box(ax, 810, 455, 135, 95, "Large\nQwen3-14B", fs=14)
box(ax, 1070, 430, 370, 120, "Response + Measurements\nlatency, tokens, errors")
box(ax, 1070, 240, 370, 110, "Evaluation & Feedback\nchecks, judge, user rating")
box(ax, 1070, 60, 370, 100, "Reward Calculator\nR = α_q·Q − β_c·C − γ_l·L")
box(ax, 520, 60, 420, 100, "Update Statistics of Chosen Model\nA_a ← A_a + x xᵀ,   b_a ← b_a + R x")
box(ax, 60, 680, 1380, 100, "Response returned to user  ·  every decision, outcome, reward and update written to audit log", fs=16)
arrow(ax, (225, 160), (225, 240)); arrow(ax, (225, 350), (225, 430))
arrow(ax, (390, 490), (505, 490)); label(ax, 447, 465, "select a_t", fs=13)
arrow(ax, (955, 490), (1070, 490))
arrow(ax, (1255, 430), (1255, 350)); arrow(ax, (1255, 240), (1255, 160))
arrow(ax, (1070, 110), (940, 110))
arrow(ax, (600, 160), (360, 430), ls="--"); label(ax, 545, 300, "improved future\nrouting", fs=13)
arrow(ax, (1255, 550), (1255, 680))
f.savefig(f"{out}/image1.png"); plt.close(f)

# ---------- Fig. 2: evaluation design ----------
f, ax = fig(1500, 700)
box(ax, 40, 60, 300, 120, "Evaluation Dataset\n7 task categories", bold=True)
box(ax, 40, 280, 300, 120, "Development / Tuning\nsplit")
box(ax, 40, 500, 300, 120, "Held-out Test\nsplit")
box(ax, 430, 60, 330, 120, "Run every query on\nevery candidate model")
box(ax, 430, 280, 330, 120, "Full-information log\nQ, C, L, failures per\n(query, model)")
box(ax, 430, 500, 330, 120, "Manual audit of\njudge / rubric scores")
box(ax, 850, 50, 300, 270, "Routing Policies\nSmall-only · Large-only\nRandom · Rule-based\nLinUCB (proposed)", fs=16)
box(ax, 850, 400, 300, 220, "Replay / Simulation\nchosen model's logged\noutcome revealed as\nbandit feedback")
box(ax, 1230, 60, 240, 560, "Metrics\n\nQuality\nLatency (mean, p95)\nCost / request\nThroughput\nCumulative reward\nRegret vs. oracle\nSelection share\nFailure rate\nShift adaptation", fs=15)
arrow(ax, (190, 180), (190, 280)); arrow(ax, (140, 400), (140, 500))
arrow(ax, (340, 120), (430, 120)); arrow(ax, (595, 180), (595, 280)); arrow(ax, (595, 400), (595, 500))
arrow(ax, (760, 340), (850, 480)); arrow(ax, (1000, 320), (1000, 400))
arrow(ax, (1150, 510), (1230, 510)); label(ax, 640, 668, "α, λ and reward weights tuned on the development split; final comparison reported on the held-out split only", fs=13)
f.savefig(f"{out}/image2.png"); plt.close(f)

# ---------- Fig. 3: implementation ----------
f, ax = fig(1500, 720)
box(ax, 40, 280, 240, 140, "Client /\nApplication", bold=True)
box(ax, 360, 60, 560, 560, "")
ax.text(640, 95, "Router Service", ha="center", va="center", fontsize=19, fontweight="bold")
box(ax, 395, 140, 490, 90, "Feature Extractor")
box(ax, 395, 260, 490, 90, "LinUCB Policy  (per-model A_a, b_a)")
box(ax, 395, 380, 490, 90, "Model Interface + Safeguards\ntimeouts · retries · fallback (logged)")
box(ax, 395, 500, 490, 90, "Reward Calculator + Feedback Intake")
box(ax, 1010, 60, 450, 330, "")
ax.text(1235, 95, "Serving Layer (vLLM)", ha="center", va="center", fontsize=19, fontweight="bold")
box(ax, 1045, 140, 380, 65, "Qwen3-4B endpoint", fs=15)
box(ax, 1045, 220, 380, 65, "Qwen3-8B endpoint", fs=15)
box(ax, 1045, 300, 380, 65, "Qwen3-14B endpoint", fs=15)
box(ax, 1010, 460, 450, 160, "Audit Log / Evaluation Store\ncontext · model · outcome ·\nreward · update  (privacy-aware)")
arrow(ax, (280, 330), (395, 185), style="->"); label(ax, 320, 230, "query", fs=13)
arrow(ax, (395, 425), (280, 385)); label(ax, 330, 372, "response", fs=13)
arrow(ax, (640, 230), (640, 260)); arrow(ax, (640, 350), (640, 380)); arrow(ax, (640, 470), (640, 500))
arrow(ax, (885, 425), (1045, 252), style="<->"); arrow(ax, (885, 545), (1010, 545))
arrow(ax, (160, 420), (395, 560), rad=0.3, ls="--"); label(ax, 200, 560, "user\nfeedback", fs=13)
f.savefig(f"{out}/image3.png"); plt.close(f)
