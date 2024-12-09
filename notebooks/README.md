# **Solar Farm Insight**

Exploratory Data Analysis (`EDA`) for MoonLight Energy Solutions to identify optimal locations for solar farm installations, focusing on sustainability and operational efficiency through data-driven insights.

## **Dataset Overview**

### **Solar Radiation Measurement Data**
The data for this week's challenge is extracted and aggregated from Solar Radiation Measurement Data. Each row in the data contains the values for solar radiation, air temperature, relative humidity, barometric pressure, precipitation, wind speed, and wind direction, cleaned and soiled radiance sensor (soiling measurement) and cleaning events.

The structure of the data is as follows:
- **`Timestamp (yyyy-mm-dd hh:mm)`**: Date and time of each observation.
- **`GHI (W/m²)`**: Global Horizontal Irradiance, the total solar radiation received per square meter on a horizontal surface.
- **`DNI (W/m²)`**: Direct Normal Irradiance, the amount of solar radiation received per square meter on a surface perpendicular to the rays of the sun.
- **`DHI (W/m²)`**: Diffuse Horizontal Irradiance, solar radiation received per square meter on a horizontal surface that does not arrive on a direct path from the sun.
- **`ModA (W/m²)`**: Measurements from a module or sensor (A), similar to irradiance.
- **`ModB (W/m²)`**: Measurements from a module or sensor (B), similar to irradiance.
- **`Tamb (°C)`**: Ambient Temperature in degrees Celsius.
- **`RH (%)`**: Relative Humidity as a percentage of moisture in the air.
- **`WS (m/s)`**: Wind Speed in meters per second.
- **`WSgust (m/s)`**: Maximum Wind Gust Speed in meters per second.
- **`WSstdev (m/s)`**: Standard Deviation of Wind Speed, indicating variability.
- **`WD (°N (to east))`**: Wind Direction in degrees from north.
- **`WDstdev`**: Standard Deviation of Wind Direction, showing directional variability.
- **`BP (hPa)`**: Barometric Pressure in hectopascals.
- **`Cleaning (1 or 0)`**: Signifying whether cleaning (possibly of the modules or sensors) occurred.
- **`Precipitation (mm/min)`**: Precipitation rate measured in millimeters per minute.
- **`TModA (°C)`**: Temperature of Module A in degrees Celsius.
- **`TModB (°C)`**: Temperature of Module B in degrees Celsius.
- **`Comments`**: This column is designed for any additional notes.

## **Exploratory Data Analysis (`EDA`)**

Performing Exploratory Data Analysis (EDA) analysis to understand the dataset and identify patterns, trends, and relationships between variables.

### **Summary Statistics**

**Central Tendency and Distribution**: The mean and median values for most variables, such as temperature (`Tamb`) and relative humidity (RH), are relatively close, indicating a fairly symmetrical distribution. Solar irradiance metrics (`GHI`, `DNI`, `DHI`) show significant differences between the mean and higher percentile values (75th and max), highlighting the presence of peaks during the observation period.

**Variability and Range**: High standard deviations for irradiance metrics (`GHI`, `DNI`, `DHI`) and module outputs (`ModA`, `ModB`) indicate large fluctuations, which may be due to seasonal changes, time of day, or weather conditions. Wind-related variables, including wind speed (`WS`), wind gusts (`WSgust`), and wind direction (`WD`), also show notable variability, though less pronounced compared to solar metrics. Barometric pressure (`BP`) has relatively low variability, suggesting consistent atmospheric pressure in these locations.

**Outliers and Potential Anomalies**: Negative values in solar irradiance metrics (e.g., `GHI`, `DNI`, `DHI`) likely represent nighttime readings or potential sensor errors that need to be accounted for during analysis. Precipitation and cleaning metrics show very low means and counts of significant values, indicating rare occurrences. Extreme max values for wind gusts and solar parameters could signal sporadic weather events or measurement anomalies that might require further investigation.

### **Data Quality Check**

For all three locations—Benin, Sierra Leone, and Togo—the following observations were made from the analysis:

**Missing Values:**  The 'Comments' column across all datasets is entirely null (`525,600` missing values). This variable provides no meaningful information and should be excluded from further analysis unless imputation is justified.

**Negative Values:** Negative entries were identified in solar irradiance columns (`GHI`, `DNI`, `DHI`). These values are incorrect since irradiance cannot be negative. Potential causes could include sensor calibration errors or processing anomalies. These should be replaced with zeros or handled appropriately in preprocessing.

**Outliers:**

- **Solar Irradiance (`GHI`, `DNI`, `DHI`):** Significant outliers are present, with extreme high values compared to the interquartile range (IQR). These may represent clear-sky conditions or potential sensor errors. Careful consideration is required to retain or exclude these values based on domain knowledge.

- **Sensor Readings (`ModA`, `ModB`):** Outliers are observed in the module output data, with extreme values potentially caused by environmental factors or measurement anomalies. Verification of these outliers is necessary to ensure data integrity.

- **Wind Data (WS, `WSgust`):** Outliers exist for wind speed and gust values, although the distributions are more consistent compared to solar metrics. Extreme gust readings might correspond to rare weather events or sensor spikes.

- **Zero Values:** High frequencies of zero values were found in `GHI`, `DNI`, and `DHI`, particularly in periods with no sunlight (e.g., nighttime). These values are valid and should remain in the dataset.

All distributions exhibit positive skewness, with the majority of readings clustered at lower ranges and fewer high values. This pattern is consistent with typical solar and wind datasets but requires normalization or transformation for certain types of analysis.

### **Time Series Analysis**

#### **Monthly Patterns of `GHI`, `DNI`, `DHI`, and `Tamb`**
**Benin:** Solar irradiance (`GHI`, `DNI`, `DHI`) follows a clear seasonal pattern, peaking during specific months (likely dry seasons) and declining in others (wet/cloudy seasons). Temperature (`Tamb`) remains relatively stable but shows slight seasonal variation, aligning with expected climatic patterns.

**Sierra Leone:** `GHI`, `DNI`, and `DHI` exhibit similar seasonal trends but at generally lower levels compared to Benin, indicating less solar resource availability.`Tamb` is stable but marginally lower compared to Benin, reflecting differences in local climate.

**Togo:** Seasonal solar patterns closely resemble Benin, with slightly lower peaks in `GHI` and `DNI`.`Tamb` shows minimal fluctuation throughout the year, indicating consistent temperature conditions.

#### **Seasonal Decomposition of `ModA` and `ModB`**
**Trend:** - All locations exhibit a steady long-term trend in sensor readings (`ModA`, `ModB`), with fluctuations primarily due to environmental factors such as seasonal variation in solar irradiance and temperature.

**Seasonality:**- Strong seasonal components are observed in all datasets, correlating with the monthly patterns of solar irradiance and temperature.

**Residuals:** The residual components highlight significant noise and outliers, potentially due to random environmental effects or sensor anomalies.

#### **Impact of Cleaning on Sensor Readings (`ModA`, `ModB`)**

**Boxplots Analysis:** Sensor readings post-cleaning show reduced variability and fewer extreme outliers, indicating improved data quality. The mean readings for `ModA` and `ModB` increase significantly after cleaning, suggesting that dust or debris negatively affected sensor performance before cleaning.

**Trend Plots:** The time series trends of `ModA` and `ModB` clearly differentiate the periods before and after cleaning. Post-cleaning periods show more consistent and elevated readings, reinforcing the impact of cleaning on sensor performance.

**Statistical Analysis:**
   - **Benin:** The T-test results show a significant difference between pre-cleaning and post-cleaning readings for both `ModA` and `ModB` (P-value: `1.92e-06`), indicating that cleaning had a statistically significant positive effect.

   - **Sierra Leone:** A significant increase in mean sensor readings is observed post-cleaning (P-value: `5.85e-07`), confirming that cleaning greatly enhanced sensor performance.

   - **Togo:** The effect of cleaning is most pronounced here, with P-value: `5.96e-60`. This result strongly suggests that cleaning substantially improves sensor accuracy, highlighting the importance of regular maintenance.

### **Correlation Analysis**

**Solar Irradiance Components (`GHI`, `DNI`, `DHI`):** `GHI` is highly correlated with `ModA` and `ModB` (correlation coefficient ~0.99), confirming that these modules are primarily driven by global horizontal irradiance.  `DNI` has a slightly weaker correlation (~0.88) with `ModA` and `ModB`, suggesting that direct irradiance is not the sole determinant of sensor output. `DHI` shows the lowest correlation among solar components (~0.85), indicating a reduced influence of diffuse radiation compared to direct and global irradiance.

**Temperature Relationships (`Tamb`, `TModA`, `TModB`):** Moderate correlations (~0.55-0.65) between solar components (`GHI`, `DNI`, `DHI`) and `Tamb`/TMod metrics. This indicates that temperature increases alongside irradiance but is not a direct driver of the observed variation. The strong correlation between T`ModA` and TModB (~0.99) reflects their similar measurement functionality and shared response to environmental conditions.

**Wind Conditions (WS, `WSgust`):** Windspeed (`WS`) and gusts (`WSgust`) are weakly correlated with solar irradiance metrics (~0.2-0.4), which is expected since wind does not directly affect solar radiation but may influence cooling rates for sensors.

### **Conclusion**

Benin has been identified as the optimal location for solar energy deployment, driven by its high levels of solar radiation and relatively stable climatic conditions. The analysis underscores the importance of regular sensor cleaning to maintain optimal performance and ensure accurate data collection. Additionally, seasonal trends, including variations in solar irradiance and temperature, should be considered when planning and maintaining solar systems to maximize efficiency and sustainability.
