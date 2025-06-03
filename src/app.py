import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import io

st.title("Interactive Curve Modifier")

st.markdown("""
Upload a CSV with pivot points (x,y columns).  
Define intervals to flatten as comma-separated x_start-x_end pairs (e.g. 4-5,6-7).  
See the plot, download modified data and plot image.
""")

# Upload pivot points CSV
uploaded_file = st.file_uploader("Upload pivot points CSV (with columns x,y)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if not {'x','y'}.issubset(df.columns):
        st.error("CSV must contain columns: x, y")
        st.stop()

    x_pivot = df['x'].values
    y_pivot = df['y'].values

    # Interpolation function
    try:
        f = interp1d(x_pivot, y_pivot, kind='cubic', fill_value="extrapolate")
    except Exception as e:
        st.error(f"Error creating interpolation: {e}")
        st.stop()

    # Generate dense x values
    x = np.linspace(x_pivot.min(), x_pivot.max(), 1000)
    y = f(x)

    # Flatten intervals input
    flatten_input = st.text_input(
        "Enter flatten intervals (comma-separated, e.g. 4-5,6-7):",
        value=""
    )

    # Process flatten intervals
    if flatten_input.strip():
        intervals = []
        try:
            parts = [s.strip() for s in flatten_input.split(",")]
            for p in parts:
                start, end = map(float, p.split("-"))
                if start >= end:
                    st.warning(f"Interval start {start} must be less than end {end}. Skipping.")
                    continue
                intervals.append((start, end))
        except Exception as e:
            st.error("Invalid flatten intervals format. Use format like: 4-5,6-7")
            st.stop()

        # Apply flattening by setting y values constant within intervals
        for start, end in intervals:
            mask = (x >= start) & (x <= end)
            if np.any(mask):
                flat_value = np.mean(y[(x >= start) & (x <= end)])
                y[mask] = flat_value

    # Add a demo spike (fixed between x=6 and 7)
    spike_center = 6.5
    spike_width = 0.05
    spike_height = 0.5
    spike = spike_height * np.exp(-((x - spike_center) ** 2) / (2 * spike_width ** 2))
    spike_mask = (x >= 6) & (x <= 7)
    y[spike_mask] += spike[spike_mask]

    # Add demo vibration (fixed between x=7 and 8)
    vib_mask = (x >= 7) & (x <= 8)
    amplitude = 0.1
    frequency = 20
    y[vib_mask] += amplitude * np.sin(frequency * np.pi * (x[vib_mask] - 7) / (8 - 7))

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y, label="Modified curve")
    ax.scatter(x_pivot, y_pivot, color='red', label='Pivot points')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Curve with Flattening, Spike & Vibration")
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)

    # Prepare data for download
    data_df = pd.DataFrame({"x": x, "y": y})

    csv_buffer = io.StringIO()
    data_df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()

    st.download_button(
        label="Download modified curve data (CSV)",
        data=csv_data,
        file_name="modified_curve.csv",
        mime="text/csv"
    )

    # Save plot to buffer for image download
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format="png")
    img_buffer.seek(0)

    st.download_button(
        label="Download plot image (PNG)",
        data=img_buffer,
        file_name="curve_plot.png",
        mime="image/png"
    )
else:
    st.info("Please upload a CSV file with pivot points to start.")
