import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO, BytesIO
from scipy.interpolate import interp1d

st.title("🎵 Phrase Curve Variation Synthesizer 🎵")

# File upload
st.sidebar.header("Upload File (.txt)")
uploaded_file = st.sidebar.file_uploader("Choose a TXT or CSV file", type=["txt", "csv"])

# Sidebar user inputs
vibration_amplitude_input = st.sidebar.text_input("Vibration Amplitude", value="0.2")
vibration_freq_input = st.sidebar.text_input("Vibration Frequency (higher = more waves)", value="20.0")
peak_height_input = st.sidebar.text_input("Peak Height", value="1.0")
flatten_level_input = st.sidebar.text_input("Flatten Level (offset)", value="0.0")
noise_level_input = st.sidebar.text_input("Noise Level (std dev)", value="0.05")

def safe_float(val, default):
    try:
        return float(val)
    except:
        return default

# Convert inputs
vibration_amplitude = safe_float(vibration_amplitude_input, 0.2)
vibration_frequency = safe_float(vibration_freq_input, 20.0)
peak_height = safe_float(peak_height_input, 1.0)
flatten_level = safe_float(flatten_level_input, 0.0)
noise_level = safe_float(noise_level_input, 0.05)

pivot_points = []

# File parsing
if uploaded_file is not None:
    try:
        df_raw = pd.read_csv(uploaded_file, sep=None, engine='python', header=None)
        if df_raw.shape[1] < 2:
            st.sidebar.error("The file must contain at least two columns (x and y).")
        else:
            df_raw.columns = ['x', 'y']
            pivot_points = list(zip(df_raw['x'], df_raw['y']))
            st.sidebar.success("File uploaded and parsed successfully.")
            st.subheader("Uploaded Pivot Points")
            st.dataframe(df_raw)
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")

# Curve generation and transformation
if pivot_points and len(pivot_points) >= 2:
    px, py = zip(*pivot_points)
    x_dense = np.linspace(min(px), max(px), 1000)
    f_interp = interp1d(px, py, kind='cubic')
    y_dense = f_interp(x_dense)

    # Display original curve
    st.subheader("Original Curve")
    fig1, ax1 = plt.subplots()
    ax1.plot(x_dense, y_dense, color='black', linewidth=2.5)
    ax1.grid(True)
    st.pyplot(fig1)

    # Editable transformation table
    st.subheader("Define Transformations")
    default_data = pd.DataFrame({
        "Start X": [2.0, 5.0],
        "End X": [4.0, 6.5],
        "Condition": ["Flatten", "Vibration, Peak"]
    })
    condition_options = ["Flatten", "Vibration", "Peak", "Noise", "Invert", "Invariant"]
    edited_data = st.data_editor(
        default_data,
        column_config={
            "Condition": st.column_config.TextColumn(
                "Condition (comma-separated)", help="E.g., Vibration, Peak"
            ),
        },
        num_rows="dynamic",
        use_container_width=True
    )

    y_mod = y_dense.copy()

    # Apply each transformation
    for _, row in edited_data.iterrows():
        x1, x2 = float(row["Start X"]), float(row["End X"])
        condition_str = row["Condition"]
        if x1 >= x2:
            continue

        idx1 = np.searchsorted(x_dense, x1)
        idx2 = np.searchsorted(x_dense, x2)
        segment_x = x_dense[idx1:idx2+1]
        segment_y = y_mod[idx1:idx2+1]

        conditions = [c.strip() for c in (condition_str or "").split(",") if c.strip()]


        for condition in conditions:
            if condition == "Flatten":
                mean_val = np.mean(segment_y) + flatten_level
                segment_y = np.full_like(segment_y, mean_val)
            elif condition == "Vibration":
                vibration = vibration_amplitude * np.sin(vibration_frequency * segment_x)
                segment_y += vibration
            elif condition == "Peak":
                mid = (segment_x[0] + segment_x[-1]) / 2
                sigma = (segment_x[-1] - segment_x[0]) / 6
                gauss = np.exp(-((segment_x - mid) ** 2) / (2 * sigma ** 2))
                segment_y += gauss * peak_height
            elif condition == "Noise":
                noise = np.random.normal(0, noise_level, size=segment_y.shape)
                segment_y += noise
            elif condition == "Invert":
                mean_val = np.mean(segment_y)
                segment_y = 2 * mean_val - segment_y
            # Invariant = do nothing

        y_mod[idx1:idx2+1] = segment_y

    # Plot modified curve
    st.subheader("Modified Curve")
    fig2, ax2 = plt.subplots()
    ax2.plot(x_dense, y_mod, label="Modified", color='black', linewidth=0.8)
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)

    # Download modified curve as PNG
    buf = BytesIO()
    fig2.savefig(buf, format="png")
    st.download_button(
        label="Download Modified Curve as PNG",
        data=buf.getvalue(),
        file_name="modified_curve.png",
        mime="image/png"
    )

    # Display modified data
    st.subheader("Modified (x, y) Values")
    df_mod = pd.DataFrame({"x": x_dense, "y": y_mod})
    st.dataframe(df_mod)

    # Download TXT
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
    st.info("Please upload a file with at least two pivot points.")
