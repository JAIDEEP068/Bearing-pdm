"""Show WHY envelope analysis works: peaks in the envelope spectrum land on the
predicted BPFO/BPFI defect frequencies for OR/IR faults, but not for Normal."""
import numpy as np, glob, os, re
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.io import loadmat
from features import _bandpass, FS, WIN, MULT
from scipy.signal import hilbert

def load(path):
    m = loadmat(path); num = re.search(r'_(\d+)\.mat$', os.path.basename(path)).group(1)
    sig = m[f'X{num}_DE_time'].ravel().astype(float)
    rpm = float(m[f'X{num}RPM'].ravel()[0]) if f'X{num}RPM' in m else 1772.0
    return sig, rpm/60.0

picks = [("Time_Normal_1_098.mat","Normal", None),
         ("IR007_1_110.mat","Inner Race (BPFI)","BPFI"),
         ("OR007_6_1_136.mat","Outer Race (BPFO)","BPFO")]

fig, axes = plt.subplots(3,2, figsize=(13,9))
for row,(fn,title,defect) in enumerate(picks):
    sig, fr = load(f"/mnt/user-data/uploads/{fn}")
    x = sig[20000:20000+WIN]
    t = np.arange(WIN)/FS*1000
    axes[row,0].plot(t, x, lw=0.7, color="#333"); axes[row,0].set_title(f"{title} - raw signal")
    axes[row,0].set_xlabel("time (ms)"); axes[row,0].set_ylabel("accel (g)")
    # envelope spectrum
    xb = _bandpass(x, 2000, 6000); env = np.abs(hilbert(xb)); env -= env.mean()
    E = np.abs(np.fft.rfft(env*np.hanning(len(env)))); f = np.fft.rfftfreq(len(env),1/FS)
    axes[row,1].plot(f, E, color="#4C72B0"); axes[row,1].set_xlim(0,600)
    axes[row,1].set_title(f"{title} - envelope spectrum"); axes[row,1].set_xlabel("Hz")
    for name,c in [("BPFO","#C44E52"),("BPFI","#55A868")]:
        for h in (1,2,3):
            axes[row,1].axvline(MULT[name]*fr*h, color=c, ls="--", lw=1, alpha=.7)
    if defect:
        axes[row,1].text(.98,.9,f"peaks at {defect}\n({MULT[defect]*fr:.0f} Hz x harmonics)",
                         transform=axes[row,1].transAxes, ha="right", fontsize=9,
                         bbox=dict(boxstyle="round", fc="white", ec="gray"))
plt.suptitle("Envelope analysis: defect frequency = the bearing element that is damaged", y=1.0)
plt.tight_layout(); plt.savefig("outputs/09_envelope_physics.png", dpi=120); plt.close()
print("Saved physics plot. BPFO line=red, BPFI line=green.")
