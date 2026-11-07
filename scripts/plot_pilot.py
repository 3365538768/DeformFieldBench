"""Standalone scientific figure from recorded exploratory validation curves."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]
fig,axes=plt.subplots(1,2,figsize=(11,4.2),sharey=True)
for ax,file,title in zip(axes,("pilot_v1_results.json","pilot_v2_balanced_results.json"),("Plain contact BCE","Balanced contact BCE")):
    d=json.loads((root/"docs"/file).read_text())
    for name,label,color in (("C_aux_no_feedback","C: auxiliary fields, no feedback","#2369BD"),("D_field_bridge","D: predicted-field feedback","#D36A19")):
        history=d["variants"][name]["validation_curve"]
        ax.plot([x["epoch"]+1 for x in history],[x["mae"] for x in history],marker="o",ms=3,label=label,color=color)
    ax.axhline(d["constant_baselines"]["train_median_standardized_mae"],color="#777777",ls="--",lw=1,label="Train-median constant")
    ax.axvspan(.5,2.5,color="#E5E5E5",alpha=.6)
    ax.set(title=title,xlabel="Epoch",xlim=(.5,12.5))
    ax.grid(alpha=.2)
axes[0].set_ylabel("Validation standardized MAE (lower is better)")
axes[1].legend(fontsize=8,loc="upper right")
fig.suptitle("Engineering pilot: 226 training / 14 validation clips; frozen DINOv2-S; one seed",fontsize=11)
fig.text(.5,.01,"Nonrandom partial data; best epochs selected on this validation set. No test evaluation or superiority claim.",ha="center",fontsize=8)
fig.tight_layout(rect=(0,.04,1,.94))
fig.savefig(root/"docs/pilot_validation_curves.png",dpi=180)
fig.savefig(root/"docs/pilot_validation_curves.svg")
