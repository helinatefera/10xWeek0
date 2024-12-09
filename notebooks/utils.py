import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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


def describe_data(df, region_name):
    numeric_columns = [
        "GHI",
        "DNI",
        "DHI",
        "ModA",
        "ModB",
        "Tamb",
        "RH",
        "WS",
        "WSgust",
        "WSstdev",
        "WD",
        "WDstdev",
        "BP",
        "Cleaning",
        "Precipitation",
        "TModA",
        "TModB",
    ]
    print(f"\nDescriptive Statistics for {region_name}:\n")
    for col in numeric_columns:
        if col in df.columns:
            print(df[col].describe())
            print("-" * 20)
        else:
            print(f"Column '{col}' not found in {region_name} dataset.")


def analyze_data(df, region_name):
    print(f"\nAnalysis for {region_name}:\n")

    for col in ["GHI", "DNI", "DHI", "ModA", "ModB", "WS", "WSgust"]:
        if col in df.columns:
            # Missing Values
            missing_count = df[col].isnull().sum()
            print(f"Missing values in {col}: {missing_count}")

            # Outliers (using IQR method)
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            print(f"Number of outliers in {col}: {len(outliers)}")

            # Incorrect Entries (e.g., negative GHI, DNI, DHI)
            if col in ["GHI", "DNI", "DHI"]:
                negative_values = df[df[col] < 0]
                print(
                    f"Number of negative values in {col}: {len(negative_values)}"
                )

            # Further analysis for specific columns
            if col in ["ModA", "ModB"]:
                print(
                    f"Min {col}: {df[col].min()}, Max {col}: {df[col].max()}"
                )

            if col in ["WS", "WSgust"]:
                print(
                    f"Min {col}: {df[col].min()}, Max {col}: {df[col].max()}"
                )

        else:
            print(f"Column '{col}' not found in {region_name} dataset.")


def plot_timeseries(df, region_name):
    """Plots GHI, DNI, DHI, and Tamb over time for a given region."""

    plt.figure(figsize=(12, 6))

    # Check if the required columns exist in the DataFrame
    for col in ["GHI", "DNI", "DHI", "Tamb"]:
        if col in df.columns:
            plt.plot(df.index, df[col], label=col)
        else:
            print(f"Column '{col}' not found in {region_name} dataset.")

    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title(
        f"{region_name} - Solar Irradiance and Ambient Temperature Over Time"
    )
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_cleaning_impact(df, region_name):
    """Plots the impact of cleaning on ModA and ModB over time."""

    plt.figure(figsize=(12, 6))

    if (
        "Cleaning" in df.columns
        and "ModA" in df.columns
        and "ModB" in df.columns
        and "Timestamp" in df.columns
    ):
        try:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
            df = df.set_index("Timestamp")
        except ValueError:
            print(
                "Error converting 'Timestamp' column to datetime. Check the format."
            )
            return
        for cleaning_status in df["Cleaning"].unique():
            subset = df[df["Cleaning"] == cleaning_status]
            plt.plot(
                subset.index,
                subset["ModA"],
                label=f"ModA (Cleaning={cleaning_status})",
            )
            plt.plot(
                subset.index,
                subset["ModB"],
                label=f"ModB (Cleaning={cleaning_status})",
            )

        plt.xlabel("Time")
        plt.ylabel("Sensor Reading")
        plt.title(f"{region_name} - Impact of Cleaning on Sensor Readings")
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print(
            "Required columns ('Cleaning', 'ModA', 'ModB', 'Timestamp') not found in the DataFrame."
        )


def plot_correlation_matrix(df, title):
    """Plots a correlation matrix for the specified DataFrame."""

    # Select relevant columns for correlation analysis
    cols_to_correlate = ["GHI", "DNI", "DHI", "TModA", "TModB"]

    if not all(col in df.columns for col in cols_to_correlate):
        print("Not all required columns found in DataFrame.")
        return

    correlation_matrix = df[cols_to_correlate].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(title)
    plt.show()


def plot_wind_solar_relationship(df, region_name):
    """Plots scatter matrices to investigate the relationship between wind conditions and solar irradiance."""

    cols_to_plot = ["GHI", "DNI", "DHI", "WS", "WSgust", "WD"]

    if not all(col in df.columns for col in cols_to_plot):
        print("Not all required columns found in DataFrame.")
        return

    sns.pairplot(df[cols_to_plot], diag_kind="kde")  # kde for diagonal plots
    plt.suptitle(
        f"Scatter Matrix for {region_name} - Wind vs. Solar Irradiance", y=1.02
    )
    plt.show()


def plot_wind_rose(df, region_name):
    """Plots a wind rose for a given region's wind speed and direction."""

    if "WS" not in df.columns or "WD" not in df.columns:
        print("Columns 'WS' and 'WD' not found in DataFrame.")
        return

    N = 16
    direction = np.array(df["WD"])
    speed = np.array(df["WS"])

    direction_bins = np.linspace(0, 360, N + 1)
    speed_bins = np.linspace(0, speed.max(), 10)
    wind_rose_data = np.zeros((len(speed_bins), len(direction_bins) - 1))

    for i in range(len(speed)):
        dir_index = np.digitize(direction[i], direction_bins) - 1
        speed_index = np.digitize(speed[i], speed_bins) - 1

        if 0 <= dir_index < len(direction_bins) - 1 and 0 <= speed_index < len(
            speed_bins
        ):
            wind_rose_data[speed_index, dir_index] += 1

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i in range(len(speed_bins) - 1):
        values = np.concatenate((wind_rose_data[i, :], [wind_rose_data[i, 0]]))
        ax.plot(
            angles, values, label=f"{speed_bins[i]:.1f}-{speed_bins[i+1]:.1f}"
        )
        ax.fill(angles, values, alpha=0.5)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.arange(0, 360, 22.5))
    plt.title(f"Wind Rose for {region_name}")
    plt.legend(title="Wind Speed (m/s)", loc="center right")
    plt.show()


def analyze_rh_impact(df, region_name):
    """Analyzes the impact of relative humidity (RH) on temperature and solar radiation."""

    required_cols = ["RH", "Tamb", "GHI", "DNI", "DHI"]
    if not all(col in df.columns for col in required_cols):
        print(f"Not all required columns found in {region_name} dataset.")
        return

    plt.figure(figsize=(8, 6))
    plt.scatter(df["RH"], df["Tamb"], alpha=0.5)
    plt.xlabel("Relative Humidity (%)")
    plt.ylabel("Ambient Temperature (°C)")
    plt.title(f"{region_name}: RH vs. Ambient Temperature")
    plt.grid(True)
    plt.show()

    correlation_rh_tamb = df["RH"].corr(df["Tamb"])
    print(
        f"{region_name}: Correlation between RH and Tamb: {correlation_rh_tamb:.2f}"
    )

    solar_cols = ["GHI", "DNI", "DHI"]
    plt.figure(figsize=(12, 4))
    for i, col in enumerate(solar_cols):
        plt.subplot(1, 3, i + 1)
        plt.scatter(df["RH"], df[col], alpha=0.5)
        plt.xlabel("Relative Humidity (%)")
        plt.ylabel(col)
        plt.title(f"{region_name}: RH vs. {col}")
        plt.grid(True)

        correlation = df["RH"].corr(df[col])
        print(
            f"{region_name}: Correlation between RH and {col}: {correlation:.2f}"
        )

    plt.tight_layout()
    plt.show()


def plot_bubble_chart(df, x_col, y_col, bubble_col, region_name):
    required_cols = [x_col, y_col, bubble_col]
    if not all(col in df.columns for col in required_cols):
        print(f"Not all required columns found in the {region_name} dataset.")
        return

    plt.figure(figsize=(10, 6))
    plt.scatter(df[x_col], df[y_col], s=df[bubble_col] * 10, alpha=0.6)

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(
        f"{region_name}: {x_col} vs. {y_col} (Bubble Size: {bubble_col})"
    )
    plt.grid(True)
    plt.show()
