import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt, freqz
import random

st.set_page_config(page_title="Awesome Peak Filtering Demo", layout="wide")

st.title("✨ Awesome Random Peak Filtering Demo ✨")
st.markdown("**A workshop demo for peak detection by Lucian Grzegorzewski (IFC Group)**")
st.markdown(
    """
    Generate random noisy signals with peaks and see how frequency filtering cleans them up!
    Adjust the parameters on the left, regenerate the signal, and apply filters.
    **Watch the magic happen** in the interactive plots below. 🤩
    """
)

# --- Helper Functions ---

def generate_peaks_only(t, fs, num_peaks, height_range, width_range_ms):
    """Generates a signal with random Gaussian peaks only (no noise)."""
    signal = np.zeros_like(t)
    duration = t[-1] if len(t) > 0 else 0
    min_height, max_height = height_range
    min_width_ms, max_width_ms = width_range_ms
    min_width = min_width_ms / 1000.0
    max_width = max_width_ms / 1000.0

    # Ensure peaks don't overlap too much at edges
    buffer = max_width * 3
    # Handle edge case where duration is too short or t is empty
    if duration <= 2 * buffer or duration == 0:
        st.warning(f"Duration ({duration*1000:.1f} ms) may be too short for the max peak width ({max_width_ms:.1f} ms). Peaks might be cut off or not generated.")
        # Generate a single peak in the middle if possible, or none if duration is 0
        peak_centers = [duration / 2] * num_peaks if duration > 0 else []
    else:
        peak_centers = np.random.uniform(buffer, duration - buffer, num_peaks)

    for center in peak_centers:
        height = random.uniform(min_height, max_height)
        width = random.uniform(min_width, max_width)
        # Using Gaussian peaks: width is FWHM, convert to std dev for formula
        std_dev = width / (2 * np.sqrt(2 * np.log(2))) # FWHM = 2*sqrt(2*ln2)*sigma
        if std_dev > 0: # Avoid division by zero if width is zero
             signal += height * np.exp(-0.5 * ((t - center) / std_dev) ** 2)

    return signal

def generate_random_signal(t, fs, num_peaks, height_range, width_range_ms, noise_level):
    """Generates a signal with random peaks and white noise."""
    signal = np.zeros_like(t)
    duration = t[-1]
    min_height, max_height = height_range
    min_width_ms, max_width_ms = width_range_ms
    min_width = min_width_ms / 1000.0
    max_width = max_width_ms / 1000.0

    # Ensure peaks don't overlap too much at edges
    buffer = max_width * 3
    if duration <= 2 * buffer:
        st.warning(f"Duration ({duration*1000:.1f} ms) is too short for the max peak width ({max_width_ms:.1f} ms). Peaks might be cut off.")
        peak_centers = [duration / 2] * num_peaks # Place all in center if duration too short
    else:
        peak_centers = np.random.uniform(buffer, duration - buffer, num_peaks)

    for center in peak_centers:
        height = random.uniform(min_height, max_height)
        width = random.uniform(min_width, max_width)
        # Using Gaussian peaks
        signal += height * np.exp(-0.5 * ((t - center) / (width / 2.355)) ** 2) # width is FWHM for Gaussian

    noise = noise_level * np.random.randn(len(t))
    noisy_signal = signal + noise
    return noisy_signal, t

def design_filter(lowcut, highcut, fs, order):
    """Designs a Butterworth filter."""
    nyq = 0.5 * fs
    ftype = "band"
    if lowcut <= 0 and highcut >= nyq: # No filter case
        return None, None, "none"
    elif lowcut <= 0:  # Low-pass
        wn = highcut / nyq
        b, a = butter(order, wn, btype="low")
        ftype = "low-pass"
    elif highcut >= nyq:  # High-pass
        wn = lowcut / nyq
        b, a = butter(order, wn, btype="high")
        ftype = "high-pass"
    else:  # Band-pass
        wn = [lowcut / nyq, highcut / nyq]
        b, a = butter(order, wn, btype="band")
        ftype = "band-pass"
    return b, a, ftype

def apply_filter(b, a, signal, order):
    """Applies the filter using filtfilt, checking signal length."""
    if b is None or a is None: # Handle no filter case
        return signal

    # Check required signal length for filtfilt
    min_len = 3 * (order + 1)
    if len(signal) <= min_len:
        st.warning(f"Signal length ({len(signal)}) is too short for the filter order ({order}). Required > {min_len}. Filtering skipped.")
        return signal # Return original signal if too short

    try:
        return filtfilt(b, a, signal)
    except ValueError as e:
        st.error(f"Filtering failed: {e}. Try adjusting filter parameters or signal length.")
        return signal # Return original if filtering fails

def fft_mag(x, fs, show_db):
    """Calculates the magnitude spectrum of a signal."""
    n = len(x)
    if n == 0:
        return np.array([]), np.array([])
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, 1 / fs)
    mag = np.abs(X)
    if show_db:
        # Avoid log(0) issues
        mag = 20 * np.log10(np.maximum(mag, 1e-12))
    return freqs, mag

def get_filter_response(b, a, fs, show_db):
    """Calculates the frequency response of the filter."""
    if b is None or a is None: # Handle no filter case
        return np.array([]), np.array([])
    w, h = freqz(b, a, worN=2048)
    w_hz = w * fs / (2 * np.pi)
    H = np.abs(h)
    if show_db:
        H = 20 * np.log10(np.maximum(H, 1e-12))
    return w_hz, H

def plot_time_domain(t, noisy_signal, filtered_signal):
    """Creates an interactive time-domain plot using Plotly."""
    fig = go.Figure()
    # Noisy signal: Light blue, thin, less opaque
    fig.add_trace(go.Scatter(x=t * 1e3, y=noisy_signal, mode='lines', name='Noisy',
                             line=dict(width=1, color='lightblue'), opacity=0.6))
    # Filtered signal: Orange, thicker, fully opaque
    fig.add_trace(go.Scatter(x=t * 1e3, y=filtered_signal, mode='lines', name='Filtered',
                             line=dict(width=2.5, color='orange')))
    fig.update_layout(
        title="Time Domain",
        xaxis_title="Time (ms)",
        yaxis_title="Amplitude",
        legend_title="Signal",
        margin=dict(l=20, r=20, t=40, b=20),
        height=350
    )
    fig.update_xaxes(rangemode='tozero')
    return fig

def plot_frequency_domain(freqs_noisy, mag_noisy, mag_filtered, w_hz, H, ftype, show_db, fs):
    """Creates an interactive frequency-domain plot using Plotly."""
    fig = go.Figure()
    yaxis_title = "Magnitude (dB)" if show_db else "Magnitude"

    # Noisy spectrum: Light blue, thin, less opaque
    fig.add_trace(go.Scatter(x=freqs_noisy, y=mag_noisy, mode='lines', name='Noisy',
                             line=dict(width=1, color='lightblue'), opacity=0.6))
    # Filtered spectrum: Orange, thicker, fully opaque
    fig.add_trace(go.Scatter(x=freqs_noisy, y=mag_filtered, mode='lines', name='Filtered',
                             line=dict(width=2.5, color='orange')))

    if len(w_hz) > 0 and len(H) > 0 : # Check if filter response exists
        # Scale filter response for visibility
        scale_factor = np.max(mag_noisy) / np.max(H) if np.max(H) > 1e-9 else 1
        fig.add_trace(go.Scatter(x=w_hz, y=H * scale_factor, mode='lines', name=f'{ftype.capitalize()} Filter TF (scaled)', line=dict(dash='dash', width=1.5)))

    fig.update_layout(
        title="Frequency Domain",
        xaxis_title="Frequency (Hz)",
        yaxis_title=yaxis_title,
        legend_title="Spectrum",
        xaxis_range=[0, fs / 2],
        margin=dict(l=20, r=20, t=40, b=20),
         height=350
    )
    return fig

# --- Initialize Session State ---
# Separate state for base signal (peaks only) and time vector
if 'base_signal' not in st.session_state:
    st.session_state.base_signal = None
    st.session_state.t = None
    st.session_state.fs = 10000 # Default fs
    st.session_state.duration_ms = 500 # Default duration

# -------------------- Sidebar Controls --------------------
st.sidebar.header("⚙️ Signal Generation")
fs_changed = False
new_fs = st.sidebar.number_input(
    "Sampling frequency (Hz)",
    min_value=1000,
    max_value=50000,
    value=st.session_state.fs,
    step=1000,
    key="fs_input"
)
if new_fs != st.session_state.fs:
    st.session_state.fs = new_fs
    fs_changed = True # Mark that fs changed

duration_ms_changed = False
new_duration_ms = st.sidebar.slider("Signal duration (ms)", 50, 2000, st.session_state.duration_ms, key="duration_slider")
if new_duration_ms != st.session_state.duration_ms:
    st.session_state.duration_ms = new_duration_ms
    duration_ms_changed = True
duration = st.session_state.duration_ms / 1000.0

num_peaks = st.sidebar.slider("Number of random peaks", 1, 50, 10, key="num_peaks")
peak_height_min, peak_height_max = st.sidebar.slider(
    "Peak height range", 0.1, 10.0, (1.0, 5.0), key="peak_height"
)
peak_width_min_ms, peak_width_max_ms = st.sidebar.slider(
    "Peak width range (ms, FWHM)", 0.5, 20.0, (1.0, 5.0), key="peak_width"
)
noise_level = st.sidebar.slider("Baseline noise level", 0.0, 5.0, 0.5, key="noise_level")

regen_button = st.sidebar.button("🔄 Regenerate Peaks", type="primary")

# Regenerate base signal (peaks) if button pressed, fs changed, duration changed, or no signal exists
# Noise is added later based on the slider, independent of this block
regenerate_peaks = regen_button or fs_changed or duration_ms_changed or st.session_state.base_signal is None

if regenerate_peaks:
    st.session_state.fs = new_fs # Ensure fs is up-to-date
    st.session_state.duration_ms = new_duration_ms # Ensure duration is up-to-date
    current_duration = st.session_state.duration_ms / 1000.0
    st.session_state.t = np.linspace(0, current_duration, int(current_duration * st.session_state.fs), endpoint=False)

    # Check if time vector generation resulted in non-empty array
    if st.session_state.t.size > 0:
        st.session_state.base_signal = generate_peaks_only(
            st.session_state.t,
            st.session_state.fs,
            num_peaks,
            (peak_height_min, peak_height_max),
            (peak_width_min_ms, peak_width_max_ms)
        )
        st.toast("New peaks generated!", icon="⛰️")
    else:
        # Handle case where time vector is empty (e.g., duration * fs is too small)
        st.session_state.base_signal = np.array([])
        st.warning("Could not generate signal, duration or sampling frequency might be too low.")


st.sidebar.header("🔧 Filter Parameters")
fs = st.session_state.fs # Use fs from session state
nyq = 0.5 * fs
max_freq = int(nyq) - 10 # Ensure cuts are below Nyquist

# Ensure sliders reset if fs changes and makes current values invalid
if 'lowcut' not in st.session_state or st.session_state.lowcut > max_freq:
    st.session_state.lowcut = 0
if 'highcut' not in st.session_state or st.session_state.highcut > max_freq:
    st.session_state.highcut = min(500, max_freq)

lowcut = st.sidebar.slider(
    "Low-cut (Hz, 0 = Low-pass)", 0, max_freq, st.session_state.lowcut, step=10, key="lowcut"
)
highcut = st.sidebar.slider(
    "High-cut (Hz)", lowcut + 10 if lowcut > 0 else 10 , max_freq, st.session_state.highcut, step=10, key="highcut"
)

order = st.sidebar.slider("Butterworth order", 1, 10, 4, key="order")
show_db = st.sidebar.checkbox("Show magnitude in dB", value=False, key="show_db")

# -------------------- Validation & Processing --------------------
st.header("📊 Filtering Results")

# Retrieve base signal and time from state
base_signal = st.session_state.base_signal
t = st.session_state.t

if base_signal is None or t is None:
    st.warning("Please generate a signal using the sidebar controls.")
    st.stop()

# Generate noise based on CURRENT slider value and add to BASE signal
# This happens every time, allowing noise slider to update independently
current_noise = noise_level * np.random.randn(len(t)) if len(t) > 0 else np.array([])
noisy_signal = base_signal + current_noise

# Design filter
b, a, ftype = design_filter(lowcut, highcut, fs, order)

# Apply filter (passing order for length check)
filtered_signal = apply_filter(b, a, noisy_signal, order)

# Calculate FFTs
freqs_noisy, mag_noisy = fft_mag(noisy_signal, fs, show_db)
_, mag_filtered = fft_mag(filtered_signal, fs, show_db)

# Get filter response
w_hz, H = get_filter_response(b, a, fs, show_db)

# -------------------- Layout & Plotting --------------------
col_time, col_freq = st.columns(2)

with col_time:
    st.plotly_chart(plot_time_domain(t, noisy_signal, filtered_signal), use_container_width=True)

with col_freq:
    st.plotly_chart(plot_frequency_domain(freqs_noisy, mag_noisy, mag_filtered, w_hz, H, ftype, show_db, fs), use_container_width=True)

st.header("💡 Tips & Info")
st.markdown(
    f"""
    * **Workshop:** Peak Detection Demo by Lucian Grzegorzewski (IFC Group)
    * **Sampling Frequency (Fs):** {fs} Hz. Determines the highest frequency captured (Nyquist = {nyq:.0f} Hz).
    * **Filter Type:** Currently `{ftype}`.
        * Set **Low-cut** to **0 Hz** for a *low-pass* filter.
        * Set **High-cut** near **Nyquist** ({nyq:.0f} Hz) for a *high-pass* filter.
        * Keep both within bounds for a *band-pass* filter.
    * The **Butterworth order** controls the filter's steepness. Higher order = sharper cutoff, but potentially more instability.
    * Use the **'Regenerate Peaks'** button to get new random peak data. Noise level updates instantly.
    * **Interactive Plots:** Zoom, pan, and hover over the plots (powered by Plotly) to explore the data!

    ---
    To run this app locally:
    ```bash
    pip install streamlit numpy scipy plotly
    streamlit run filter_demo.py
    ```
    """
)

# Add a footer or link
st.markdown("---")
st.markdown("Created with Streamlit & Plotly by Lucian Grzegorzewski (IFC Group).")
