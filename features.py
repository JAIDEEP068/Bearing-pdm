"""
Feature extraction for CWRU drive-end bearing signals (48 kHz).

For each non-overlapping 2048-sample window we compute three families:
  (A) Time-domain statistics  -> replicates the 9 columns in the original CSV
  (B) FFT / spectral features -> shape & energy distribution of the spectrum
  (C) Envelope-spectrum features -> amplitude at bearing DEFECT frequencies
      (BPFO/BPFI/BSF). This is the physics-aware part: a localized defect
      produces periodic impacts that AMPLITUDE-MODULATE a high-freq resonance.
      Demodulating (Hilbert envelope) and reading the modulation frequency
      tells you WHICH element is damaged. This is standard bearing diagnostics.
"""
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import hilbert, butter, filtfilt

FS = 48000          # sampling rate (Hz)
WIN = 2048          # window length (samples) -- matches the original CSV
# SKF 6205-2RS JEM drive-end bearing, defect-frequency multipliers (x shaft freq)
MULT = {"BPFO": 3.5848, "BPFI": 5.4152, "BSF": 4.7135, "FTF": 0.39828}

TIME_COLS = ["max","min","mean","sd","rms","skewness","kurtosis","crest","form"]

def time_features(x):
    """The 9 time-domain stats present in the provided CSV."""
    mx, mn = x.max(), x.min()
    mean = x.mean()
    sd   = x.std(ddof=1)                       # sample std (matches CSV scale)
    rms  = np.sqrt(np.mean(x**2))
    sk   = skew(x)
    ku   = kurtosis(x)                          # Fisher (excess) kurtosis
    crest= mx / rms if rms else 0.0             # peak / rms  -> impulsiveness
    form = rms / (np.mean(np.abs(x)) + 1e-12)   # rms / mean-abs
    return [mx, mn, mean, sd, rms, sk, ku, crest, form]

def spectral_features(x, fs=FS, n_bands=8):
    """FFT-based descriptors of the magnitude spectrum."""
    X = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1/fs)
    p = X**2
    psum = p.sum() + 1e-12
    pn = p / psum
    centroid = (f * pn).sum()                              # spectral centroid
    spread   = np.sqrt(((f - centroid)**2 * pn).sum())     # spectral spread
    entropy  = -(pn * np.log(pn + 1e-12)).sum()            # spectral entropy
    peak_f   = f[np.argmax(X)]                             # dominant frequency
    peak_mag = X.max()
    # energy in equal-width sub-bands across 0..Nyquist
    edges = np.linspace(0, fs/2, n_bands+1)
    band_e = [p[(f>=edges[i])&(f<edges[i+1])].sum()/psum for i in range(n_bands)]
    return [centroid, spread, entropy, peak_f, peak_mag] + band_e

def _bandpass(x, lo, hi, fs=FS, order=4):
    b, a = butter(order, [lo/(fs/2), hi/(fs/2)], btype="band")
    return filtfilt(b, a, x)

def envelope_features(x, fr, fs=FS):
    """
    Envelope (Hilbert) spectrum amplitudes at bearing defect frequencies.
    fr = shaft rotational frequency (Hz) = rpm/60.
    Steps: band-pass to a high-freq resonance band -> Hilbert envelope ->
           FFT of envelope -> read amplitude at BPFO/BPFI/BSF and harmonics.
    """
    # demodulate around a mid/high resonance band (2-6 kHz works well for CWRU DE)
    xb = _bandpass(x, 2000, 6000, fs)
    env = np.abs(hilbert(xb))
    env = env - env.mean()                       # remove DC so 0 Hz doesn't dominate
    E = np.abs(np.fft.rfft(env * np.hanning(len(env))))
    f = np.fft.rfftfreq(len(env), 1/fs)
    res = f[1] - f[0]                             # bin width (~23.4 Hz)
    Esum = E.sum() + 1e-12

    def amp_at(freq, harmonics=(1,2,3), tol_bins=2):
        """Sum envelope amplitude near freq and its harmonics (+/- tol bins)."""
        tot = 0.0
        for h in harmonics:
            target = freq*h
            if target >= f[-1]:
                continue
            idx = int(round(target/res))
            lo, hi = max(0, idx-tol_bins), min(len(E), idx+tol_bins+1)
            tot += E[lo:hi].max()
        return tot

    feats = []
    for name in ["BPFO","BPFI","BSF"]:
        a = amp_at(MULT[name]*fr)
        feats.append(a)            # raw amplitude
        feats.append(a/Esum)       # normalized share of envelope energy
    return feats

ENV_COLS, SPEC_COLS = None, None
def column_names(n_bands=8):
    spec = ["spec_centroid","spec_spread","spec_entropy","peak_freq","peak_mag"] \
           + [f"band{i}_energy" for i in range(n_bands)]
    env  = []
    for name in ["BPFO","BPFI","BSF"]:
        env += [f"{name}_amp", f"{name}_share"]
    return TIME_COLS, spec, env

if __name__ == "__main__":
    # smoke test on a synthetic signal
    t = np.arange(WIN)/FS
    x = np.sin(2*np.pi*2000*t) + 0.1*np.random.randn(WIN)
    print("time:", len(time_features(x)))
    print("spec:", len(spectral_features(x)))
    print("env :", len(envelope_features(x, 29.5)))
    print(column_names())
