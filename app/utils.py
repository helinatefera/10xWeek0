import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from windrose import WindroseAxes
import streamlit as st

def plot_histogram(data, column, ax, title, xlabel):
    """
    Plots a histogram with KDE using Seaborn.

    Parameters:
    - data: DataFrame containing the data
    - column: Column name to plot
    - ax: Matplotlib Axes object to plot on
    - title: Title of the plot
    - xlabel: Label for the x-axis
    """
    sns.histplot(data[column], kde=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)

def plot_wind_rose(data, wind_speed_col='WS', wind_direction_col='WD', bins=8):
    """
    Plots a wind rose for the given wind speed and direction columns.
    
    Args:
        data (DataFrame): The input data containing wind speed and direction.
        wind_speed_col (str): The column name for wind speed.
        wind_direction_col (str): The column name for wind direction.
        bins (int): Number of bins for wind speed categorization.
    """
    # Extracting the necessary columns
    wind_speed = data[wind_speed_col]
    wind_direction = data[wind_direction_col]

    # Setting up the wind rose plot
    fig = plt.figure(figsize=(4, 4))
    ax = WindroseAxes.from_ax(fig=fig)
    ax.bar(
        wind_direction, wind_speed,
        normed=True,
        opening=0.8,
        edgecolor='white',
        bins=np.linspace(wind_speed.min(), wind_speed.max(), bins)
    )
    ax.set_legend(title="Wind Speed (m/s)", loc="best")
    ax.set_title("Wind Rose")
    
    # Displaying the plot in Streamlit
    st.pyplot(fig)

def temperature_humidity_analysis(data, rh_col="RH", temp_cols=["TModA", "TModB"], solar_cols=["GHI", "DNI", "DHI"]):
    """
    Analyzes the relationship between RH and temperature/solar radiation using heatmaps and joint plots.
    
    Args:
        data (DataFrame): The input data containing RH, temperature, and solar radiation columns.
        rh_col (str): The column name for relative humidity.
        temp_cols (list): List of temperature-related columns.
        solar_cols (list): List of solar radiation-related columns.
    """
    st.subheader("Temperature and Humidity Analysis (Alternative Visualizations)")
    
    # Heatmap: RH vs Temperature & Solar Radiation
    combined_cols = [rh_col] + temp_cols + solar_cols
    st.markdown("### Heatmap of Correlations")
    corr_matrix = data[combined_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)
    
    # Joint Plots for RH vs Each Selected Column
    st.markdown("### Detailed Joint Plots")
    for col in temp_cols + solar_cols:
        st.markdown(f"#### {rh_col} vs {col}")
        fig = sns.jointplot(data=data, x=rh_col, y=col, kind="reg", height=6, ratio=4, marginal_ticks=True, color="teal")
        st.pyplot(fig)

def calculate_summary_statistics(data):
    """
    Calculates summary statistics including mean, median, variance, and other relevant statistics
    for numeric columns in the dataset.
    
    Args:
        data (DataFrame): The input dataset.

    Returns:
        DataFrame: The summary statistics table.
    """
    st.subheader("Summary Statistics")
    
    # Select numeric columns
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()

    if not numeric_columns:
        st.warning("No numeric columns found in the dataset!")
    else:
        # Calculate basic statistics
        stats_df = data[numeric_columns].describe().T
        stats_df['median'] = data[numeric_columns].median()  # Add median
        stats_df['variance'] = data[numeric_columns].var()  # Add variance
        
        # Rename columns for better readability
        stats_df = stats_df.rename(columns={
            "mean": "Mean",
            "std": "Standard Deviation",
            "min": "Minimum",
            "25%": "25th Percentile",
            "50%": "50th Percentile",
            "75%": "75th Percentile",
            "max": "Maximum"
        })
        
        # Display statistics table
        st.write(stats_df)
        
    return stats_df

def dataset_summary(data):
    """
    Generates a summary of the dataset including information on data types, missing values, duplicates,
    negative or zero values, and basic statistics.

    Args:
        data (DataFrame): The input dataset.
    """
    # Dataset Information
    st.subheader("Dataset Information")
    buffer = data.info(buf=None)
    st.text(buffer)

    # Missing Values Check
    st.subheader("Missing Values")
    missing_values = data.isnull().sum()
    st.write(missing_values[missing_values > 0])
    if missing_values.sum() == 0:
        st.success("No missing values detected!")

    # Duplicate Rows Check
    st.subheader("Duplicate Rows")
    duplicate_count = data.duplicated().sum()
    st.write(f"Total duplicate rows: {duplicate_count}")
    if duplicate_count > 0:
        st.warning(f"{duplicate_count} duplicate rows found! Consider removing them.")

    # Negative or Zero Values Check
    st.subheader("Negative or Zero Values Check")
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    selected_columns = st.multiselect(
        "Select columns to check for negative or zero values",
        numeric_columns
    )
    if selected_columns:
        for col in selected_columns:
            negative_or_zero = data[data[col] <= 0]
            if not negative_or_zero.empty:
                st.warning(f"Column `{col}` has {len(negative_or_zero)} rows with negative or zero values.")
                st.write(negative_or_zero[[col]])
            else:
                st.success(f"No negative or zero values detected in `{col}`.")

    # Basic Statistics Summary
    st.subheader("Basic Statistics Summary")
    st.write(data.describe())
    
def correlation_analysis(data):
    """
    Performs correlation analysis, visualizes correlation matrix, creates pair plots for solar radiation
    and temperature measures, and provides scatter plots for wind conditions vs solar irradiance.
    
    Args:
        data (DataFrame): The input dataset.
    """
    # Select columns for correlation analysis
    solar_columns = ['GHI', 'DNI', 'DHI', 'TModA', 'TModB']
    wind_columns = ['WS', 'WSgust', 'WD']
    selected_columns = solar_columns + wind_columns

    # Correlation Matrix
    st.subheader("Correlation Matrix")
    corr_matrix = data[selected_columns].corr()
    st.write("Correlation values:")
    st.dataframe(corr_matrix)

    # Heatmap of the Correlation Matrix
    st.write("Heatmap of Correlations")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".1f", cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Pair Plot for Solar Radiation and Temperature Measures
    st.subheader("Pair Plot for Solar Radiation and Temperature")
    st.write("Investigate pairwise relationships among `GHI`, `DNI`, `DHI`, `TModA`, and `TModB`.")
    pairplot_fig = sns.pairplot(data[solar_columns], diag_kind="kde", corner=True)
    st.pyplot(pairplot_fig)

    # Scatter Plot for Wind Conditions and Solar Irradiance
    st.subheader("Scatter Plots for Wind Conditions and Solar Irradiance")
    x_var = st.selectbox("Select Wind Condition (X-axis)", wind_columns)
    y_var = st.selectbox("Select Solar Irradiance Component (Y-axis)", solar_columns)

    st.write(f"Scatter plot of `{x_var}` vs `{y_var}`.")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=data, x=x_var, y=y_var, ax=ax, alpha=0.7)
    ax.set_title(f"Scatter Plot: {x_var} vs {y_var}")
    st.pyplot(fig)
