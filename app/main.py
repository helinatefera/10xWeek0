import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from streamlit_pandas_profiling import st_profile_report
from utils import *

st.set_page_config(page_title="Simple Dashboard", layout="wide")
st.markdown(
    """
    <style>
    .stApp {
        background-color: white;
        color: black;
    }
     .st-emotion-cache-6qob1r{
        background-color: black;  /* Change this color as per your preference */
        color: white;
    }
    .stAppHeader{
        background-color: #D3D5D6;
    }
    .stMain{
        background-color: #E8E8E6;
    }
    .st-emotion-cache-p0pjm {
        color: #D8AB41;
    }
    .st-emotion-cache-kgpedg{
    color: #D8AB41;
    }
    .st-emotion-cache-1puwf6r{
    color: white;
    }
    .css-1d391kg .css-1v3fvcr {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    selected_option = st.radio(
        "Select an option:",
        [
            "Display head of CSV file",
            "Plot Histogram",
            "Summary Statistics",
            "Data Quality Check",
            "Correlation Analysis",
            "Wind Analysis",
            "Temperature Analysis",
        ],
    )

st.title("SRM Data Dashboard")

uploaded_file = st.file_uploader("Choose a CSV file to upload", type="csv")


if uploaded_file:
    if selected_option == "Display head of CSV file":
        st.header("Upload CSV and Plot Data")
        data = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Data Preview")
        st.data_editor(data, num_rows="dynamic")
        st.subheader("Graph of Uploaded Data")
        if st.checkbox("Show Plot"):
            if data.shape[1] >= 2:
                x_col = st.selectbox("Select X-axis column", data.columns)
                y_col = st.selectbox("Select Y-axis column", data.columns)

                fig, ax = plt.subplots(figsize=(8, 2))
                ax.plot(data[x_col], data[y_col], marker="o", color="b")
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_title(f"{y_col} vs {x_col}")
                st.pyplot(fig)
            else:
                st.warning(
                    "Uploaded data must have at least two columns for plotting."
                )


    elif selected_option == "Plot Histogram":
        data = pd.read_csv(uploaded_file)

        # Select column for histogram
        numeric_columns = data.select_dtypes(
            include=["float64", "int64"]
        ).columns.tolist()
        if not numeric_columns:
            st.warning("No numeric columns found in the dataset!")
        elif numeric_columns == []:
            st.warning("No numeric columns found in the dataset!")
        else:
            selected_column = st.radio(
                "Select a Column to Plot", options=numeric_columns, index=0
            )

            # Plot histogram
            st.subheader(f"Histogram of {selected_column}")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(
                data[selected_column], bins=20, color="skyblue", edgecolor="black"
            )
            ax.set_xlabel(selected_column)
            ax.set_ylabel("Frequency")
            ax.set_title(f"Distribution of {selected_column}")
            st.pyplot(fig)
    elif selected_option == "Summary Statistics":
        # Display dataset
        data = pd.read_csv(uploaded_file)

        # Summary statistics
        calculate_summary_statistics(data)
    elif selected_option == "Data Quality Check":
        # Display dataset
        data = pd.read_csv(uploaded_file)

        # Data Quality Check
        dataset_summary(data)
    elif selected_option == "Correlation Analysis":

        data = pd.read_csv(uploaded_file)

        # Correlation Analysis
        correlation_analysis(data)
    elif selected_option == "Wind Analysis":
        data = pd.read_csv(uploaded_file)
        wind_speed_col = st.selectbox(
            "Select Wind Speed Column",
            data.columns,
            index=data.columns.get_loc("WS"),
        )
        wind_direction_col = st.selectbox(
            "Select Wind Direction Column",
            data.columns,
            index=data.columns.get_loc("WD"),
        )

        bins = st.slider(
            "Number of Bins for Wind Speed",
            min_value=4,
            max_value=12,
            value=8,
            step=1,
        )

        if st.button("Generate Wind Rose"):
            plot_wind_rose(
                data,
                wind_speed_col=wind_speed_col,
                wind_direction_col=wind_direction_col,
                bins=bins,
            )
    elif selected_option == "Temperature Analysis":
        data = pd.read_csv(uploaded_file)
        rh_col = st.selectbox(
            "Select Relative Humidity Column",
            data.columns,
            index=data.columns.get_loc("RH"),
        )
        temp_cols = st.multiselect(
            "Select Temperature Columns", data.columns, default=["TModA", "TModB"]
        )
        solar_cols = st.multiselect(
            "Select Solar Radiation Columns",
            data.columns,
            default=["GHI", "DNI", "DHI"],
        )

        if st.button("Analyze"):
            temperature_humidity_analysis(
                data, rh_col=rh_col, temp_cols=temp_cols, solar_cols=solar_cols
            )

    else:
        st.info("Awaiting CSV file upload. Please upload your dataset.")
else:
    st.warning("Please upload a CSV file to start the analysis.") 