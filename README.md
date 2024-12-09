# Solar Farm Insight

## Project Overview
This project involves conducting Exploratory Data Analysis (EDA) for MoonLight Energy Solutions to identify the best locations for solar farm installations. The goal is to enhance sustainability and operational efficiency through insightful data analysis.

## Dataset Summary

### Solar Radiation Measurement Data
The data provided for this challenge has been extracted and aggregated from Solar Radiation Measurement records. Each entry in the dataset includes information on solar radiation levels, environmental factors like air temperature, humidity, wind speed, and precipitation, as well as measurements related to sensor cleaning and soiling events.

The dataset consists of the following columns:

- **Timestamp (yyyy-mm-dd hh:mm)**: The date and time when the observation was made.
- **GHI (W/m²)**: Global Horizontal Irradiance, representing the total solar radiation reaching a horizontal surface.
- **DNI (W/m²)**: Direct Normal Irradiance, showing the solar radiation received per square meter on a surface directly facing the sun.
- **DHI (W/m²)**: Diffuse Horizontal Irradiance, the amount of solar radiation reaching a horizontal surface but not directly from the sun.
- **ModA (W/m²)**: Irradiance measurement from Module A, similar to GHI.
- **ModB (W/m²)**: Irradiance measurement from Module B, similar to GHI.
- **Tamb (°C)**: Ambient temperature measured in degrees Celsius.
- **RH (%)**: The percentage of moisture in the air (Relative Humidity).
- **WS (m/s)**: Wind speed in meters per second.
- **WSgust (m/s)**: The peak wind gust speed measured in meters per second.
- **WSstdev (m/s)**: The standard deviation of wind speed, showing its variability.
- **WD (°N (to east))**: Wind direction, measured in degrees from north.
- **WDstdev**: The standard deviation of wind direction, indicating how much the wind direction varies.
- **BP (hPa)**: Barometric pressure measured in hectopascals.
- **Cleaning (1 or 0)**: Indicates whether the cleaning of modules or sensors took place.
- **Precipitation (mm/min)**: Precipitation rate recorded in millimeters per minute.
- **TModA (°C)**: Temperature of Module A in degrees Celsius.
- **TModB (°C)**: Temperature of Module B in degrees Celsius.
- **Comments**: A column dedicated to additional notes or observations.

This data is crucial for understanding the environmental conditions that influence solar energy production and identifying the most suitable locations for solar farm installations.

# Exploratory Data Analysis (EDA)

The goal of this Exploratory Data Analysis (EDA) is to understand the dataset, uncover patterns, trends, and relationships between the variables, and identify any potential data issues.

## Summary Statistics

### Central Tendency and Distribution:
- **Mean and Median**: For most variables, such as ambient temperature (Tamb) and relative humidity (RH), the mean and median values are relatively close, indicating that these variables follow a fairly symmetrical distribution.
- **Solar Irradiance Metrics**: The solar irradiance variables (GHI, DNI, DHI) show significant differences between the mean and the higher percentile values (75th and max), suggesting the presence of peaks during the observation period.

### Variability and Range:
- **Solar Metrics**: The high standard deviations in solar irradiance metrics (GHI, DNI, DHI) and module outputs (ModA, ModB) reflect large fluctuations. These fluctuations may be influenced by factors such as seasonal changes, time of day, or varying weather conditions.
- **Wind-related Variables**: The wind speed (WS), wind gusts (WSgust), and wind direction (WD) show notable variability, although less pronounced compared to the solar metrics.
- **Barometric Pressure (BP)**: This variable displays relatively low variability, suggesting a stable atmospheric pressure in the locations where data was collected.

### Outliers and Potential Anomalies:
- **Negative Values in Solar Irradiance**: Some negative values were identified in solar irradiance metrics (e.g., GHI, DNI, DHI). These likely represent nighttime readings or potential sensor errors, which will need to be addressed during data preprocessing.
- **Precipitation and Cleaning Metrics**: These columns exhibit very low mean values and a small number of significant entries, indicating that cleaning events and precipitation are rare occurrences.
- **Extreme Max Values**: Extreme high values for wind gusts and solar parameters may signal sporadic weather events or sensor anomalies, requiring further verification and analysis.

## Data Quality Check

### Missing Values:
- **Comments Column**: Across all datasets (Benin, Sierra Leone, Togo), the 'Comments' column contains 525,600 missing values, rendering it uninformative for further analysis. This column should be excluded unless imputation is considered.

### Negative Values:
- **Solar Irradiance Metrics**: Negative values were found in the solar irradiance columns (GHI, DNI, DHI), which are physically impossible since irradiance cannot be negative. These should be corrected or replaced with zeroes during preprocessing.

### Outliers:

- **Solar Irradiance (GHI, DNI, DHI)**: Significant outliers with extreme high values were detected, which may correspond to clear-sky conditions or sensor anomalies. A decision to keep or exclude these outliers should be made based on domain expertise.
  
- **Sensor Readings (ModA, ModB)**: Similar outliers were found in the module output data (ModA, ModB). These outliers could be due to environmental factors or measurement errors, and their validity should be confirmed through further investigation.

- **Wind Data (WS, WSgust)**: Wind speed and gust data also exhibit outliers, although these are less pronounced than those in the solar metrics. Extreme gust readings might be linked to rare weather events or sensor spikes.

### Zero Values:
- **Solar Irradiance (GHI, DNI, DHI)**: High frequencies of zero values were observed, particularly during nighttime periods when solar irradiance is expected to be absent. These zero values are valid and should remain in the dataset.

### Distribution Patterns:
- All distributions of the data exhibit **positive skewness**, with most readings clustered at the lower range and fewer extreme high values. This pattern is common in solar and wind data and may require normalization or transformation for specific types of analysis.

# Time Series Analysis

## Monthly Patterns of GHI, DNI, DHI, and Tamb

### Benin:
- **Solar Irradiance (GHI, DNI, DHI)**: Exhibits a clear seasonal pattern, with peaks during certain months (likely the dry seasons) and declines during other months (wet/cloudy seasons).
- **Temperature (Tamb)**: Remains relatively stable throughout the year, showing slight seasonal variations that align with expected climatic patterns.

### Sierra Leone:
- **Solar Irradiance (GHI, DNI, DHI)**: Shows similar seasonal trends but at generally lower levels compared to Benin, indicating a lower availability of solar resources.
- **Temperature (Tamb)**: Stable but slightly lower compared to Benin, reflecting regional climatic differences.

### Togo:
- **Solar Irradiance (GHI, DNI, DHI)**: Exhibits seasonal patterns similar to Benin but with slightly lower peaks in GHI and DNI.
- **Temperature (Tamb)**: Shows minimal fluctuation across the year, suggesting stable temperature conditions.

## Seasonal Decomposition of ModA and ModB

### Trend:
- **ModA and ModB**: All locations show a steady long-term trend in sensor readings, with fluctuations largely due to environmental factors such as changes in solar irradiance and temperature.

### Seasonality:
- **Strong Seasonal Components**: All datasets display strong seasonal components, closely correlating with the monthly patterns of solar irradiance and temperature.

### Residuals:
- **Significant Noise and Outliers**: The residual components highlight the presence of significant noise and outliers, which may be attributed to random environmental effects or sensor anomalies.

## Impact of Cleaning on Sensor Readings (ModA, ModB)

### Boxplots Analysis:
- **Post-Cleaning Variability**: Sensor readings after cleaning show reduced variability and fewer extreme outliers, indicating that cleaning improved data quality. Mean readings for ModA and ModB increase significantly after cleaning, suggesting that dust or debris negatively impacted sensor performance before cleaning.

### Trend Plots:
- **Pre- and Post-Cleaning Trends**: The time series trends for ModA and ModB show a noticeable distinction between periods before and after cleaning. The post-cleaning periods exhibit more consistent and elevated readings, emphasizing the positive impact of cleaning on sensor performance.

### Statistical Analysis:

- **Benin**: The T-test results show a significant difference between pre-cleaning and post-cleaning readings for both ModA and ModB (P-value: 1.92e-06), indicating that cleaning had a statistically significant positive effect.
  
- **Sierra Leone**: A significant increase in sensor readings post-cleaning is observed (P-value: 5.85e-07), confirming that cleaning substantially enhanced sensor performance.

- **Togo**: The effect of cleaning is most pronounced in Togo, with a P-value of 5.96e-60. This result strongly suggests that cleaning significantly improves sensor accuracy, highlighting the importance of regular maintenance.

## Correlation Analysis

### Solar Irradiance Components (GHI, DNI, DHI):
- **GHI and Sensor Modules (ModA, ModB)**: GHI has a very high correlation (~0.99) with both ModA and ModB, confirming that these sensors are primarily influenced by global horizontal irradiance.
- **DNI and Sensor Modules (ModA, ModB)**: DNI shows a slightly weaker correlation (~0.88) with ModA and ModB, suggesting that direct irradiance is not the sole factor driving sensor outputs.
- **DHI and Sensor Modules (ModA, ModB)**: DHI has the weakest correlation (~0.85) with ModA and ModB, indicating that diffuse radiation has a lesser influence compared to direct and global irradiance.

### Temperature Relationships (Tamb, TModA, TModB):
- **Solar Irradiance and Temperature**: Moderate correlations (~0.55-0.65) exist between the solar components (GHI, DNI, DHI) and temperature-related variables (Tamb, TModA, TModB). While temperature increases alongside irradiance, it does not directly drive the observed variations.
- **TModA and TModB**: A strong correlation (~0.99) is seen between TModA and TModB, reflecting their similar functionality and response to environmental conditions.

### Wind Conditions (WS, WSgust):
- **Wind Speed and Gusts**: Windspeed (WS) and gust values (WSgust) show weak correlations (~0.2-0.4) with solar irradiance metrics, as expected. Wind does not directly affect solar radiation but may influence cooling rates for the sensors.

### **Conclusion**

Benin has been identified as the optimal location for solar energy deployment, driven by its high levels of solar radiation and relatively stable climatic conditions. The analysis underscores the importance of regular sensor cleaning to maintain optimal performance and ensure accurate data collection. Additionally, seasonal trends, including variations in solar irradiance and temperature, should be considered when planning and maintaining solar systems to maximize efficiency and sustainability.
