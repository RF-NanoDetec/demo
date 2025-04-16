
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, freqz

st.set_page_config(page_title="Frequency‑Domain Filtering Demo", layout="wide")

st.title("🪄 Frequency‑Domain Filtering Demo")
st.markdown(
    """
    Adjust the parameters on the left and **watch what happens** in both the time domain
    and the frequency domain.  
    This illustrates how a band‑pass **or** low‑pass / high‑pass filter cleans up 
    short 1 ms pulses buried in noise.
    """
)

# -------------------- Sidebar controls --------------------
st.sidebar.header("Signal parameters")
fs = st.sidebar.number_input("Sampling frequency (Hz)", value=10_000, step=1000)
duration_ms = st.sidebar.slider("Signal duration (ms)", 50, 1000, 200)
duration = duration_ms / 1_000
pulse_width_ms = st.sidebar.slider("Pulse width (ms)", 0.1, 5.0, 1.0)
pulse_width = pulse_width_ms / 1_000
num_pulses = st.sidebar.slider("Number of pulses", 1, 200, 100)
noise_amp = st.sidebar.slider("White noise amplitude", 0.0, 2.0, 0.8)
tone_amp = st.sidebar.slider("High‑freq tone amplitude", 0.0, 1.0, 0.5)
tone_freq = st.sidebar.slider("Tone frequency (Hz)", 500, int(fs/2) - 100, 2_000)

st.sidebar.header("Filter parameters")
lowcut = st.sidebar.slider("Low‑cut (Hz, 0 = off)", 0, int(fs/2) - 200, 0, step=10)
highcut = st.sidebar.slider("High‑cut (Hz)", 10, int(fs/2) - 10, 500, step=10)
order = st.sidebar.slider("Butterworth order", 1, 8, 4)
show_db = st.sidebar.checkbox("Show magnitude in dB", value=False)

# -------------------- Validation --------------------
nyq = 0.5 * fs
if lowcut >= highcut and lowcut != 0:
    st.error("Low‑cut must be smaller than High‑cut (unless Low‑cut is 0).")
    st.stop()
if highcut >= nyq:
    st.error("High‑cut must be below the Nyquist frequency.")
    st.stop()

# -------------------- Signal generation --------------------
t = np.linspace(0, duration, int(duration * fs), endpoint=False)

def gaussian_pulse(t, center, width):
    return np.exp(-0.5 * ((t - center) / width) ** 2)

if num_pulses > 1:
    pulse_centers = np.linspace(pulse_width * 3, duration - pulse_width * 3, num_pulses)
else:
    pulse_centers = np.array([duration / 2])

signal = np.zeros_like(t)
for c in pulse_centers:
    signal += gaussian_pulse(t, c, pulse_width)

white_noise = noise_amp * np.random.randn(len(t))
tone_noise = tone_amp * np.sin(2 * np.pi * tone_freq * t)
noisy_signal = signal + white_noise + tone_noise

# -------------------- Filtering --------------------
def design_filter(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    if lowcut <= 0:  # Low‑pass
        b, a = butter(order, highcut / nyq, btype="low")
        ftype = "low"
    elif highcut >= nyq:  # High‑pass
        b, a = butter(order, lowcut / nyq, btype="high")
        ftype = "high"
    else:  # Band‑pass
        b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
        ftype = "band"
    return b, a, ftype

b, a, ftype = design_filter(lowcut, highcut, fs, order)
filtered_signal = filtfilt(b, a, noisy_signal)

# FFT helper
def fft_mag(x, fs):
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    mag = np.abs(X)
    if show_db:
        mag = 20 * np.log10(mag + 1e-12)
    return freqs, mag

freqs_noisy, mag_noisy = fft_mag(noisy_signal, fs)
_, mag_filtered = fft_mag(filtered_signal, fs)

# Filter transfer function
w, h = freqz(b, a)
w_hz = w * fs / (2 * np.pi)
H = np.abs(h)
if show_db:
    H = 20 * np.log10(H + 1e-12)

# -------------------- Layout --------------------
col_time, col_freq = st.columns(2)

# Time‑domain plot
with col_time:
    fig_t, ax_t = plt.subplots(figsize=(6, 3))
    ax_t.plot(t * 1e3, noisy_signal, label="Noisy", alpha=0.5, linewidth=1)
    ax_t.plot(t * 1e3, filtered_signal, label="Filtered", linewidth=2)
    ax_t.set_xlabel("Time (ms)")
    ax_t.set_ylabel("Amplitude")
    ax_t.set_title("Time Domain")
    ax_t.legend()
    ax_t.grid(True, linewidth=0.3)
    st.pyplot(fig_t)

# Frequency‑domain plot
with col_freq:
    fig_f, ax_f = plt.subplots(figsize=(6, 3))
    ax_f.plot(freqs_noisy, mag_noisy, label="Noisy", alpha=0.5, linewidth=1)
    ax_f.plot(freqs_noisy, mag_filtered, label="Filtered", linewidth=2)
    ax_f.plot(w_hz, H * max(mag_noisy) / max(H), label=f"{ftype.capitalize()} filter TF (scaled)",
              linestyle="--", linewidth=1)
    ax_f.set_xlabel("Frequency (Hz)")
    ax_f.set_ylabel("Magnitude" + (" (dB)" if show_db else ""))
    ax_f.set_title("Frequency Domain")
    ax_f.set_xlim(0, fs / 2)
    ax_f.legend()
    ax_f.grid(True, linewidth=0.3)
    st.pyplot(fig_f)

st.markdown(
    """
    **Tips**

    * Set **Low‑cut** to **0 Hz** for a pure *low‑pass* that keeps every low‑frequency component.  
    * Push **High‑cut** down to watch how more of the high‑frequency noise disappears.  
    * Or raise **Low‑cut** (with High‑cut = Nyquist) for a *high‑pass* effect.  
    * Leave both within the Nyquist range for a classic *band‑pass*.

    ```bash
    pip install streamlit numpy scipy matplotlib
    streamlit run filter_demo.py
    ```
    """
)
