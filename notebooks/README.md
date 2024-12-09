# Solar Data Analysis for Benin, Sierra Leone, and Togo

This project analyzes solar energy data from three regions in West Africa: **Benin**, **Sierra Leone**, and **Togo**. The analysis aims to identify key trends, correlations, and anomalies in solar irradiance, temperature, humidity, and wind data to support decisions for solar energy generation and installation.

## Data Sources
The dataset consists of the following key columns:
- **GHI (Global Horizontal Irradiance)**: Measures the total solar radiation received per unit area on a horizontal surface.
- **DNI (Direct Normal Irradiance)**: Measures the amount of solar radiation received per unit area directly from the sun.
- **DHI (Diffuse Horizontal Irradiance)**: Measures the solar radiation scattered by atmospheric particles.
- **Tamb (Ambient Temperature)**: Measures the air temperature.
- **RH (Relative Humidity)**: Measures the amount of water vapor in the air relative to the maximum possible amount at a given temperature.
- **Wind Speed**: Measures the wind velocity at each data point.
  
## Steps Taken in Analysis

### 1. **Data Loading & Exploration**
   - Loaded and previewed the dataset for each region: **Benin**, **Sierra Leone**, and **Togo**.
   - Checked for missing values and anomalies in the dataset.
   
### 2. **Data Preprocessing**
   - Cleaned the dataset by handling missing values and dropped columns with all null values to ensure data quality.
   - Applied **Z-score analysis** to identify outliers in the data, flagging data points that significantly differ from the mean.
   
### 3. **Exploratory Data Analysis (EDA)**
   - **Correlation Analysis**: Calculated and visualized correlations between different variables (e.g., GHI, DNI, DHI, RH, and Tamb) to understand their relationships and impacts.
     - **Key Findings**: 
       - **Togo**: Weak negative correlation between RH and solar irradiance.
       - **Sierra Leone**: Strong negative correlation between RH and ambient temperature.
       - **Benin**: Moderate negative correlation between RH and all other variables.
   - **Wind Rose Plot**: Generated wind rose plots to understand wind speed and direction patterns in each region.
   - **Impact of RH**: Analyzed the relationship between relative humidity and key variables like ambient temperature and solar irradiance.
   
### 4. **Data Visualization**
   - **Time Series Plots**: Created line charts to visualize solar irradiance (GHI, DNI, DHI) and ambient temperature (Tamb) over time. This helps in understanding daily and seasonal variations.
   - **Correlation Matrix**: Plotted heatmaps to show correlations between various factors (GHI, DNI, DHI, Tamb, and RH).
   - **Wind Rose Plots**: Displayed wind speed and direction patterns to assess wind conditions' impact on solar installations.
   
### 5. **Z-Score Analysis**
   - Calculated Z-scores for all columns to identify and flag outliers.
   - Set a threshold of 3 to identify significantly different data points from the mean.

### 6. **Data Cleaning**
   - Dropped columns with all null values.
   - Ensured the dataset is ready for further analysis or modeling.

## Key Findings
- **Solar Irradiance Trends**: There are clear seasonal patterns in solar irradiance, with higher values during the dry seasons.
- **Humidity and Solar Irradiance**: A negative correlation between RH and solar irradiance indicates that higher humidity may reduce the amount of available solar energy.
- **Wind and Solar Efficiency**: Wind speed shows varying patterns across the regions, which can impact solar panel performance.
  
## Conclusion
This data analysis provides valuable insights into the relationship between environmental factors like humidity, temperature, and solar irradiance. The insights can be used for improving solar energy production efficiency, especially in regions like Benin, Sierra Leone, and Togo.

