import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO, BytesIO
import re
from scipy.interpolate import interp1d

st.title("Graph Modification with Custom Transformations")

st.sidebar.header("Input Pivot Points (format: [x,y], [x,y], ...)")
pivot_input = st.sidebar.text_area(
    "Enter pivot points here:",
    value="[0, 0], [2, 3], [4, 1], [6, 4], [8, 2]"
)

# Sliders for transformation parameters
vibration_amplitude = st.sidebar.slider("Vibration Amplitude", 0.0, 1.0, 0.2, 0.05)
peak_height = st.sidebar.slider("Peak Height", 0.0, 2.0, 1.0, 0.1)
flatten_level = st.sidebar.slider("Flatten Level (offset)", -1.0, 1.0, 0.0, 0.1)

# Parse pivot points
def parse_pivot_points(text):
    pattern = r"\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]"
    points = re.findall(pattern, text)
    return [(float(px), float(py)) for px, py in points]

try:
    pivot_points = parse_pivot_points(pivot_input)
    if len(pivot_points) < 2:
        st.sidebar.error("Please enter at least two pivot points.")
except Exception as e:
    st.sidebar.error(f"Error parsing pivot points: {e}")
    pivot_points = []

if pivot_points and len(pivot_points) >= 2:
    px, py = zip(*pivot_points)
    x_dense = np.linspace(min(px), max(px), 1000)
    f_interp = interp1d(px, py, kind='cubic')
    y_dense = f_interp(x_dense)

    # Display original graph
    st.subheader("Original Curve")
    fig1, ax1 = plt.subplots()
    ax1.plot(x_dense, y_dense, label="Original", color='blue', linewidth=2.5)
    ax1.scatter(px, py, color='red', label="Pivot Points")
    ax1.grid(True)
    ax1.legend()
    st.pyplot(fig1)

    # Table to define custom ranges and conditions
    st.subheader("Define Transformations")
    default_data = pd.DataFrame({
        "Start X": [2.0, 5.0],
        "End X": [4.0, 6.5],
        "Condition": ["Flatten", "Vibration"]
    })
    condition_options = ["Flatten", "Vibration", "Peak", "Invariant"]
    edited_data = st.data_editor(
        default_data,
        column_config={
            "Condition": st.column_config.SelectboxColumn("Condition", options=condition_options),
        },
        num_rows="dynamic",
        use_container_width=True
    )

    y_mod = y_dense.copy()

    # Apply transformations
    for _, row in edited_data.iterrows():
        x1, x2 = float(row["Start X"]), float(row["End X"])
        condition = row["Condition"]
        if x1 >= x2:
            continue

        idx1 = np.searchsorted(x_dense, x1)
        idx2 = np.searchsorted(x_dense, x2)
        segment_x = x_dense[idx1:idx2+1]
        segment_y = y_mod[idx1:idx2+1]

        if condition == "Flatten":
            mean_val = np.mean(segment_y) + flatten_level
            y_mod[idx1:idx2+1] = mean_val
        elif condition == "Vibration":
            vibration = vibration_amplitude * np.sin(20 * segment_x)
            y_mod[idx1:idx2+1] = segment_y + vibration
        elif condition == "Peak":
            mid = (segment_x[0] + segment_x[-1]) / 2
            sigma = (segment_x[-1] - segment_x[0]) / 6
            gauss = np.exp(-((segment_x - mid) ** 2) / (2 * sigma ** 2))
            y_mod[idx1:idx2+1] = segment_y + gauss * peak_height
        # Invariant: do nothing

    # Display modified curve only
    st.subheader("Modified Curve")
    fig2, ax2 = plt.subplots()
    ax2.plot(x_dense, y_mod, label="Modified", color='orange', linewidth=2.5)
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)

    # Prepare PNG download
    buf = BytesIO()
    fig2.savefig(buf, format="png")
    st.download_button(
        label="Download Modified Curve as PNG",
        data=buf.getvalue(),
        file_name="modified_curve.png",
        mime="image/png"
    )

    # Modified (x, y) table
    st.subheader("Modified (x, y) Values")
    df_mod = pd.DataFrame({"x": x_dense, "y": y_mod})
    st.dataframe(df_mod)

    # TXT download
    def convert_df_to_txt(df):
        buffer = StringIO()
        for _, row in df.iterrows():
            buffer.write(f"{row['x']}\t{row['y']}\n")
        return buffer.getvalue()

    txt_data = convert_df_to_txt(df_mod)
    st.download_button(
        label="Download Modified Data as TXT",
        data=txt_data,
        file_name='modified_data.txt',
        mime='text/plain'
    )
else:
    st.info("Please enter at least two valid pivot points.")
