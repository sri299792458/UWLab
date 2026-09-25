"""Verify paired replay inputs and compare 120 Hz against each gain's 500 Hz trace."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, required=True)
args = parser.parse_args()
root = args.root
launches = json.loads((root / "chirp_launches.json").read_text())
folders = {r["hz"]:Path(r["command"][r["command"].index("--output")+1]) for r in launches}
reports = {hz:json.loads((p / "report.json").read_text()) for hz, p in folders.items()}
traces = {hz:np.load(p / "trace.npz") for hz, p in folders.items()}
initial = {hz:np.load(p / "initial.npz") for hz, p in folders.items()}
assert all(r["status"] == "complete" for r in reports.values())
for key in ("record_sha256", "fit_sha256", "asset_sha256", "duration_s"):
    assert reports[120][key] == reports[500][key], key
checks = {}
for key in initial[120].files:
    a, b = initial[120][key], initial[500][key]
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert np.array_equal(a[0], a[1]) and np.array_equal(b[0], b[1]), key
    assert np.allclose(a, b, rtol=1e-7, atol=1e-8), key
    checks[key] = float(np.max(np.abs(a-b)))
for hz, trace in traces.items():
    assert all(np.isfinite(trace[k]).all() for k in trace.files)
    assert reports[hz]["implemented_delay_steps"] == (1 if hz == 120 else 3)
    assert reports[hz]["rows"][0]["kp"] == [200., 200., 200., 3., 3., 3.]
    assert reports[hz]["rows"][1]["kp"] == [1000., 1000., 1000., 50., 50., 50.]
rows = []
for i, name in enumerate(("uwlab", "thunder")):
    coarse, fine = traces[120], traces[500]
    same_time = {key:np.column_stack([np.interp(coarse["t"], fine["t"], fine[key][:, i, j]) for j in range(6)]) for key in ("q", "v")}
    qe, ve = coarse["q"][:, i]-same_time["q"], coarse["v"][:, i]-same_time["v"]
    rows.append({"gains":name,
                 "rate_position_difference_rms_deg":float(np.rad2deg(np.sqrt(np.mean(qe**2)))),
                 "rate_velocity_difference_rms_deg_s":float(np.rad2deg(np.sqrt(np.mean(ve**2)))),
                 "rate_position_difference_per_joint_rms_deg":np.rad2deg(np.sqrt(np.mean(qe**2,axis=0))).tolist(),
                 "120hz":reports[120]["rows"][i], "500hz":reports[500]["rows"][i]})
result = {"status":"passed", "initial_max_differences":checks, "rows":rows,
          "scope":"Fixed dynamics and starting states, two gains, two rates. Coarse rate includes nearest-step delay quantization. Hardware trace used for the fit was UMI; stock comparison is not new hardware validation. The chirp supplies a target every servo step, not a 10 Hz policy action."}
(root / "chirp_analysis.json").write_text(json.dumps(result, indent=2)+"\n")
labels = ["Shoulder pan", "Shoulder lift", "Elbow", "Wrist 1", "Wrist 2", "Wrist 3"]
for key, ylabel, filename in (("q", "Joint angle (degrees)", "chirp_positions.png"), ("v", "Joint speed (degrees/s)", "chirp_velocities.png")):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for j, ax in enumerate(axes.flat):
        ax.plot(traces[500]["t"], np.rad2deg(traces[500]["real_q" if key == "q" else "real_v"][:, j]), color="black", alpha=.55, lw=1, label="Recorded UMI hardware, Thunder gains")
        for i, gain in enumerate(("UWLab", "Thunder")):
            for hz, ls in ((500, "-"), (120, "--")):
                ax.plot(traces[hz]["t"], np.rad2deg(traces[hz][key][:, i, j]), color=("#c77715", "#1874bd")[i], ls=ls, lw=1.3, label=f"Stock {gain}, {hz} Hz")
        ax.set_title(labels[j]); ax.grid(alpha=.2); ax.set_xlabel("Time (s)"); ax.set_ylabel(ylabel)
    handles, legends = axes.flat[0].get_legend_handles_labels()
    fig.suptitle("Corrected stock D405: existing chirp, fixed fitted dynamics", y=.995)
    fig.legend(handles, legends, loc="upper center", bbox_to_anchor=(.5,.965), ncol=3, fontsize=9)
    fig.tight_layout(rect=(0,0,1,.88)); fig.savefig(root / filename, dpi=150, bbox_inches="tight"); plt.close(fig)
print(json.dumps(result, indent=2))
