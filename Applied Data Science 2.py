# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 15:06:02 2023

@author: Sanduni Sakunthala
"""
# import required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def process_file(file_path):
    
    """
    This function reads in the Climate_Change.csv file, filters data for specific indicators and years, and returns two transposed dataframes.

    Args:
    - file_path (str): file path for Climate_Change.csv file
    
    - df_new_country: a dataframe transposed by country, with country names as columns and years as rows.
    - df_new_year: a dataframe transposed by year, with countries as columns and indicators as rows.

    Returns:
    - df_new_country (pd.DataFrame): dataframe of filtered data for specific indicators, transposed
    - df_new_year (pd.DataFrame): dataframe of filtered data for specific years, transposed
    """

    # Read in Climate_Change.csv file and skip first 4 rows
    df_ClimateChange = pd.read_csv("Climate_Change.csv", skiprows=4)
    df_ClimateChange

    # Filter data for specific indicators and transpose the dataframe
    df_new_ClimateChange = df_ClimateChange[(df_ClimateChange["Indicator Name"] == "Urban population") | (
        df_ClimateChange["Indicator Name"] == "Forest area (sq. km)") | (df_ClimateChange["Indicator Name"] == "CO2 emissions (kt)")].reset_index(drop=True)
    df_new_country = df_new_ClimateChange.transpose()
    df_new_country.columns = df_new_country.iloc[0]

    # Rename columns and keep data from 36th to 75th columns
    df_new_country = df_new_country.iloc[0:, 36:75]

    # Drop any missing values
    df_new_country = df_new_country.dropna()

    # Filter data for specific years and join into one dataframe
    df_new_year = df_new_ClimateChange.iloc[34:72, :4].join(
        df_new_ClimateChange.iloc[34:75, 34:60]).reset_index(drop=True)

    # Drop any missing values
    df_new_year = df_new_year.dropna()
    # Return transposed dataframes
    return df_new_country, df_new_year


# Function caller
df_new_country, df_new_year = process_file("Climate_Change.csv")

# Create a multi-index dataframe and set columns to multi-index
df_new_country_multiindex = df_new_country.copy()
new_columns = df_new_country_multiindex.iloc[[0, 2]].apply(
    lambda x: '_'.join(x.astype(str)), axis=0)

# Set the columns as a MultiIndex
df_new_country_multiindex.columns = pd.MultiIndex.from_product([new_columns])
df_new_country_multiindex = df_new_country_multiindex.drop(
    df_new_country.index[:4])
print(df_new_country_multiindex)


# Filter data for Belgium and clean the dataframe
df_new_year_Belgium = df_new_country["Belgium"]
df_new_year_Belgium = df_new_year_Belgium.drop(
    ["Country Name", "Country Code", "Indicator Code"])
column_names = df_new_year_Belgium.iloc[0]
# take the data less the header row
df_new_year_Belgium = df_new_year_Belgium[1:]
df_new_year_Belgium.columns = column_names
df_new_year_Belgium = df_new_year_Belgium.astype(
    {'Urban population': 'float', 'CO2 emissions (kt)': 'float', 'Forest area (sq. km)': 'float', })
df_new_year_Belgium.describe()

# Filter data for Austria and clean the dataframe
df_new_year_Austria = df_new_country["Austria"]
df_new_year_Austria = df_new_year_Austria.drop(
    ["Country Name", "Country Code", "Indicator Code"])
column_names = df_new_year_Austria.iloc[0]
# take the data less the header row
df_new_year_Austria = df_new_year_Austria[1:]
df_new_year_Austria.columns = column_names
df_new_year_Austria = df_new_year_Austria.astype(
    {'Urban population': 'float', 'CO2 emissions (kt)': 'float', 'Forest area (sq. km)': 'float', })

df_new_year_Austria.describe()

# Filter data for Azerbaijan and clean the dataframe
df_new_year_Azerbaijan = df_new_country["Azerbaijan"]
df_new_year_Azerbaijan = df_new_year_Azerbaijan.drop(
    ["Country Name", "Country Code", "Indicator Code"])
column_names = df_new_year_Azerbaijan.iloc[0]
# take the data less the header row
df_new_year_Azerbaijan = df_new_year_Azerbaijan[1:]
df_new_year_Azerbaijan.columns = column_names
df_new_year_Azerbaijan = df_new_year_Azerbaijan.astype(
    {'Urban population': 'float', 'CO2 emissions (kt)': 'float', 'Forest area (sq. km)': 'float', })

# Print summary statistics of the Austria dataframe
df_new_year_Azerbaijan.describe()

# Filter data for Bangladesh and clean the dataframe
df_new_year_Bangladesh = df_new_country["Bangladesh"]
df_new_year_Bangladesh = df_new_year_Bangladesh.drop(
    ["Country Name", "Country Code", "Indicator Code"])
column_names = df_new_year_Bangladesh.iloc[0]
# take the data less the header row
df_new_year_Bangladesh = df_new_year_Bangladesh[1:]
df_new_year_Bangladesh.columns = column_names
df_new_year_Bangladesh = df_new_year_Bangladesh.astype(
    {'Urban population': 'float', 'CO2 emissions (kt)': 'float', 'Forest area (sq. km)': 'float', })
# Print summary statistics of the Austria dataframe
df_new_year_Bangladesh.describe()
print(df_new_year_Bangladesh)

# find the mode for urban population for each country
Belgium_mode = df_new_year_Belgium["Urban population"].mode()[0]
Belgium_mode_mask = df_new_year_Belgium["Urban population"] == Belgium_mode
df_new_year_Belgium.loc[Belgium_mode_mask].iloc[:, 0].to_frame()
print("Mode of Belgium:", Belgium_mode)

Austria_mode = df_new_year_Austria["Urban population"].mode()[0]
Austria_mode_mask = df_new_year_Austria["Urban population"] == Austria_mode
df_new_year_Austria.loc[Austria_mode_mask].iloc[:, 0].to_frame()
print("Mode of Austria:", Austria_mode)

Azerbaijan_mode = df_new_year_Azerbaijan["Urban population"].mode()[0]
Azerbaijan_mode_mask = df_new_year_Azerbaijan["Urban population"] == Azerbaijan_mode
df_new_year_Azerbaijan.loc[Azerbaijan_mode_mask].iloc[:, 0].to_frame()
print("Mode of Azerbaijan:", Azerbaijan_mode)

Bangladesh_mode = df_new_year_Bangladesh["Urban population"].mode()[0]
Bangladesh_mode_mask = df_new_year_Bangladesh["Urban population"] == Bangladesh_mode
df_new_year_Bangladesh.loc[Bangladesh_mode_mask].iloc[:, 0].to_frame()
print("Mode of Bangladesh:", Bangladesh_mode)


# Define the column names and their respective statistics to be calculated
column_stat = [['Sum', 'Urban population'], ['Sum', 'CO2 emissions (kt)'], ['Sum', 'Forest area (sq. km)'], ['Median', 'Urban population1'], ['Median', 'CO2 emissions (kt)1'], [
    'Median', 'Forest area (sq. km)1'], ['Mode', 'Urban population2'], ['Mode', 'CO2 emissions (kt)2'], ['Mode', 'Forest area (sq. km)2']]

# Calculate the desired statistics for each country and each column of interest, and store them in a dictionary called statistics
column_stat = [['Sum', 'Urban population'], ['Sum', 'CO2 emissions (kt)'], ['Sum', 'Forest area (sq. km)'], ['Median', 'Urban population1'], ['Median', 'CO2 emissions (kt)1'], [
    'Median', 'Forest area (sq. km)1'], ['Mode', 'Urban population2'], ['Mode', 'CO2 emissions (kt)2'], ['Mode', 'Forest area (sq. km)2']]
statistics = {
    "Urban population": {
        "Belgium": df_new_year_Belgium["Urban population"].sum(),
        "Austria": df_new_year_Austria["Urban population"].sum(),
        "Azerbaijan": df_new_year_Azerbaijan["Urban population"].sum(),
        "Bangladesh": df_new_year_Bangladesh["Urban population"].sum()
    },
    "CO2 emissions (kt)": {
        "Belgium": df_new_year_Belgium["CO2 emissions (kt)"].sum(),
        "Austria": df_new_year_Austria["CO2 emissions (kt)"].sum(),
        "Azerbaijan": df_new_year_Azerbaijan["CO2 emissions (kt)"].sum(),
        "Bangladesh": df_new_year_Bangladesh["CO2 emissions (kt)"].sum()
    },
    "Forest area (sq. km)": {
        "Belgium": df_new_year_Belgium["Forest area (sq. km)"].sum(),
        "Austria": df_new_year_Austria["Forest area (sq. km)"].sum(),
        "Azerbaijan": df_new_year_Azerbaijan["Forest area (sq. km)"].sum(),
        "Bangladesh": df_new_year_Bangladesh["Forest area (sq. km)"].sum()
    },


    "Urban population1": {
        "Belgium": df_new_year_Belgium["Urban population"].median(),
        "Austria": df_new_year_Austria["Urban population"].median(),
        "Azerbaijan": df_new_year_Azerbaijan["Urban population"].median(),
        "Bangladesh": df_new_year_Bangladesh["Urban population"].median()
    },
    "CO2 emissions (kt)1": {
        "Belgium": df_new_year_Belgium["CO2 emissions (kt)"].median(),
        "Austria": df_new_year_Austria["CO2 emissions (kt)"].median(),
        "Azerbaijan": df_new_year_Azerbaijan["CO2 emissions (kt)"].median(),
        "Bangladesh": df_new_year_Bangladesh["CO2 emissions (kt)"].median()

    },
    "Forest area (sq. km)1": {
        "Belgium": df_new_year_Belgium["Forest area (sq. km)"].median(),
        "Austria": df_new_year_Austria["Forest area (sq. km)"].median(),
        "Azerbaijan": df_new_year_Azerbaijan["Forest area (sq. km)"].median(),
        "Bangladesh": df_new_year_Bangladesh["Forest area (sq. km)"].median()
    },


    "Urban population2": {
        "Belgium": df_new_year_Belgium["Urban population"].mode(),
        "Austria": df_new_year_Austria["Urban population"].mode(),
        "Azerbaijan": df_new_year_Azerbaijan["Urban population"].mode(),
        "Bangladesh": df_new_year_Bangladesh["Urban population"].mode()
    },
    "CO2 emissions (kt)2": {
        "Belgium": df_new_year_Belgium["CO2 emissions (kt)"].mode(),
        "Austria": df_new_year_Austria["CO2 emissions (kt)"].mode(),
        "Azerbaijan": df_new_year_Azerbaijan["CO2 emissions (kt)"].mode(),
        "Bangladesh": df_new_year_Bangladesh["CO2 emissions (kt)"].mode()
    },

    "Forest area (sq. km)2": {
        "Belgium": df_new_year_Belgium["Forest area (sq. km)"].mode(),
        "Austria": df_new_year_Austria["Forest area (sq. km)"].mode(),
        "Azerbaijan": df_new_year_Azerbaijan["Forest area (sq. km)"].mode(),
        "Bangladesh": df_new_year_Bangladesh["Forest area (sq. km)"].mode()
    }

}

# create a pandas dataframe from the statistics data
df_statistics = pd.DataFrame(data=statistics)
# create a multi-level column index from column_stat and set it as the columns of df_statistics
df_statistics.columns = pd.MultiIndex.from_tuples(column_stat)

# print df_statistics dataframe
print(df_statistics)

# calculate the correlation coefficients between Forest area and CO2 emissions, Urban population and Forest area,
# and Urban population and CO2 emissions for Belgium

Belgium_Forest_co2_corr = df_new_year_Belgium['Forest area (sq. km)'].corr(
    df_new_year_Belgium['CO2 emissions (kt)'], method='pearson')
Belgium_population_forest_corr = df_new_year_Belgium['Urban population'].corr(
    df_new_year_Belgium['Forest area (sq. km)'], method='pearson')
Belgium_population_co2_corr = df_new_year_Belgium['Urban population'].corr(
    df_new_year_Belgium['CO2 emissions (kt)'], method='pearson')

# print the correlation coefficients for Belgium
print("Relationship between CO2 emissions and Forest area in Belgium :",
      Belgium_Forest_co2_corr)
print("Relationship between Urban population and Forest area in Belgium :",
      Belgium_population_forest_corr)
print("Relationship between Urban population and CO2 emissions in Belgium :",
      Belgium_population_co2_corr)

# calculate the correlation coefficients between Forest area and CO2 emissions, Urban population and Forest area,
# and Urban population and CO2 emissions for Austria

Austria_Forest_co2_corr = df_new_year_Austria['Forest area (sq. km)'].corr(
    df_new_year_Austria['CO2 emissions (kt)'], method='pearson')
Austria_population_forest_corr = df_new_year_Austria['Urban population'].corr(
    df_new_year_Austria['Forest area (sq. km)'], method='pearson')
Austria_population_co2_corr = df_new_year_Austria['Urban population'].corr(
    df_new_year_Austria['CO2 emissions (kt)'], method='pearson')

print("Relationship between CO2 emissions and Forest area in Austria :",
      Austria_Forest_co2_corr)
print("Relationship between Urban population and Forest area in Austria :",
      Austria_population_forest_corr)
print("Relationship between Urban population and CO2 emissions in Austria :",
      Austria_population_co2_corr)

# calculate the correlation coefficients between Forest area and CO2 emissions, Urban population and Forest area,
# and Urban population and CO2 emissions for Azerbaijan

Azerbaijan_Forest_co2_corr = df_new_year_Azerbaijan['Forest area (sq. km)'].corr(
    df_new_year_Azerbaijan['CO2 emissions (kt)'], method='pearson')
Azerbaijan_population_forest_corr = df_new_year_Azerbaijan['Urban population'].corr(
    df_new_year_Azerbaijan['Forest area (sq. km)'], method='pearson')
Azerbaijan_population_co2_corr = df_new_year_Azerbaijan['Urban population'].corr(
    df_new_year_Azerbaijan['CO2 emissions (kt)'], method='pearson')


Bangladesh_Forest_co2_corr = df_new_year_Bangladesh['Forest area (sq. km)'].corr(
    df_new_year_Bangladesh['CO2 emissions (kt)'], method='pearson')
Bangladesh_population_forest_corr = df_new_year_Bangladesh['Urban population'].corr(
    df_new_year_Bangladesh['Forest area (sq. km)'], method='pearson')
Bangladesh_population_co2_corr = df_new_year_Bangladesh['Urban population'].corr(
    df_new_year_Bangladesh['CO2 emissions (kt)'], method='pearson')

print("Relationship between CO2 emissions and Forest area in Bangladesh :",
      Bangladesh_Forest_co2_corr)
print("Relationship between Urban population and Forest area in Bangladesh :",
      Bangladesh_population_forest_corr)
print("Relationship between Urban population and CO2 emissions in Bangladesh :",
      Bangladesh_population_co2_corr)


# create a multi-level column index from column_cor and set it as the columns of df_corelations
column_cor = [['Belgium', 'Urban population'], ['Austria', 'Urban population1'], [
    'Azerbaijan', 'Urban population2'], ['Bangladesh', 'Urban population3']]
corelations = {
    "Urban population": {
        "CO2 emissions (kt)": Belgium_population_co2_corr,
        "Forest area (sq. km)": Belgium_population_forest_corr

    },
    "Urban population1": {
        "CO2 emissions (kt)": Austria_population_co2_corr,
        "Forest area (sq. km)": Austria_population_forest_corr
    },
    "Urban population2": {
        "CO2 emissions (kt)": Azerbaijan_population_co2_corr,
        "Forest area (sq. km)": Azerbaijan_population_forest_corr
    },
    "Urban population3": {
        "CO2 emissions (kt)": Bangladesh_population_co2_corr,
        "Forest area (sq. km)": Bangladesh_population_forest_corr
    }}

# Create a DataFrame with correlation coefficients and set column names as a MultiIndex
df_corelations = pd.DataFrame(data=corelations)
df_corelations.columns = pd.MultiIndex.from_tuples(column_cor)
print(df_corelations)

# Plot a line chart showing the urban population of different countries over time
plt.figure(figsize=(14, 6), facecolor="white")
plt.plot(df_new_country_multiindex.index,
         df_new_country_multiindex["Bangladesh_Urban population"], label="Bangladesh", linestyle=":", linewidth=4.0, color="#458B74")
plt.plot(df_new_country_multiindex.index,
         df_new_country_multiindex["Bahrain_Urban population"], label="Bahrain", linestyle="-", linewidth=4.0, color="#4daf4a")
plt.plot(df_new_country_multiindex.index,
         df_new_country_multiindex["Azerbaijan_Urban population"], label="Azerbaijan", linestyle="-", linewidth=4.0, color="#984ea3")
plt.plot(df_new_country_multiindex.index,
         df_new_country_multiindex["Austria_Urban population"], label="Austria", linestyle="-", linewidth=4.0, color="#ff7f00")
plt.plot(df_new_country_multiindex.index,
         df_new_country_multiindex["Antigua and Barbuda_Urban population"], label="Antigua and Barabuda", linestyle="-", linewidth=4.0, color="#ffff33")
plt.plot(df_new_country_multiindex.index,
         df_new_country_multiindex["Bulgaria_Urban population"], label="Bulgaria", linestyle=":", linewidth=4.0, color="#f781bf")
plt.plot(df_new_country_multiindex.index,
         df_new_country_multiindex["Belgium_Urban population"], label="Belgium", linestyle="-", linewidth=4.0, color="#a65628")

plt.xlabel("Year")
plt.ylabel("population")
plt.rcParams['font.size'] = 18
plt.rc('legend', fontsize=18)
plt.xticks(rotation=45)
plt.title("Urban population", fontweight='bold')
plt.xlim(min(df_new_country_multiindex.index,),
         max(df_new_country_multiindex.index,))
plt.ticklabel_format(axis='y', style='plain')

plt.legend(loc='center left', bbox_to_anchor=(1, 1))
plt.show()

plt.figure(figsize=(14, 6), facecolor="white")
plt.plot(df_new_country_multiindex.index,
         df_new_country_multiindex["Bangladesh_Forest area (sq. km)"], label="Bangladesh", linestyle=":", linewidth=4.0, color="#458B74")
plt.plot(df_new_country_multiindex.index,
         df_new_country_multiindex["Bahrain_Forest area (sq. km)"], label="Bahrain", linestyle="-", linewidth=4.0, color="#4daf4a")
plt.plot(df_new_country_multiindex.index,
         df_new_country_multiindex["Azerbaijan_Forest area (sq. km)"], label="Azerbaijan", linestyle="-", linewidth=4.0, color="#984ea3")
plt.plot(df_new_country_multiindex.index,
         df_new_country_multiindex["Austria_Forest area (sq. km)"], label="Austria", linestyle="-", linewidth=4.0, color="#ff7f00")
plt.plot(df_new_country_multiindex.index,
         df_new_country_multiindex["Antigua and Barbuda_Forest area (sq. km)"], label="Antigua and Barabuda", linestyle="-", linewidth=4.0, color="#ffff33")
plt.plot(df_new_country_multiindex.index,
         df_new_country_multiindex["Bulgaria_Forest area (sq. km)"], label="Bulgaria", linestyle=":", linewidth=4.0, color="#0000EE")
plt.plot(df_new_country_multiindex.index,
         df_new_country_multiindex["Belgium_Forest area (sq. km)"], label="Belgium", linestyle="-", linewidth=4.0, color="#a65628")

plt.xlabel("Year")
plt.ylabel("Forest area (sq. km)")
plt.rcParams['font.size'] = 18
plt.rc('legend', fontsize=18)
plt.xticks(rotation=45)
plt.title("Forest area", fontweight='bold')
plt.xlim(min(df_new_country_multiindex.index,),
         max(df_new_country_multiindex.index,))
plt.ticklabel_format(axis='y', style='plain')
plt.legend(loc='center left', bbox_to_anchor=(1, 1))
plt.show()

# Plot a scatter chart showing the urban population of different countries over time
plt.figure(figsize=(14, 6), facecolor="white")
plt.scatter(df_new_country_multiindex.index,
            df_new_country_multiindex["Bangladesh_Urban population"], label="Bangladesh", color="#458B74", marker="v")
plt.scatter(df_new_country_multiindex.index,
            df_new_country_multiindex["Bahrain_Urban population"], label="Bahrain", color="#e31a1c", marker="s")
plt.scatter(df_new_country_multiindex.index,
            df_new_country_multiindex["Austria_Urban population"], label="Austria", color="#ff7f00", marker="*")
plt.scatter(df_new_country_multiindex.index,
            df_new_country_multiindex["Antigua and Barbuda_Urban population"], label="Antigua and Barabuda", color="#b15928", marker="D")
plt.scatter(df_new_country_multiindex.index,
            df_new_country_multiindex["Bulgaria_Urban population"], label="Bulgaria", color="#0000EE", marker="^")
plt.scatter(df_new_country_multiindex.index,
            df_new_country_multiindex["Belgium_Urban population"], label="Belgium", linestyle="-", color="#a65628")


plt.xlabel("Year")
plt.ylabel("Population")
plt.xticks(rotation=45)
plt.title("Urban Population", fontweight='bold')
plt.xlim(min(df_new_country_multiindex.index),
         max(df_new_country_multiindex.index))
plt.ticklabel_format(axis='y', style='plain')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


plt.figure(figsize=(14, 6), facecolor="white")
plt.scatter(df_new_country_multiindex.index,
            df_new_country_multiindex["Bangladesh_CO2 emissions (kt)"], label="Bangladesh", color="#458B74", marker="v")
plt.scatter(df_new_country_multiindex.index,
            df_new_country_multiindex["Bahrain_CO2 emissions (kt)"], label="Bahrain", color="#e31a1c", marker="s")
plt.scatter(df_new_country_multiindex.index,
            df_new_country_multiindex["Austria_CO2 emissions (kt)"], label="Austria", color="#ff7f00", marker="*")
plt.scatter(df_new_country_multiindex.index,
            df_new_country_multiindex["Antigua and Barbuda_CO2 emissions (kt)"], label="Antigua and Barabuda", color="#b15928", marker="D")
plt.scatter(df_new_country_multiindex.index,
            df_new_country_multiindex["Bulgaria_CO2 emissions (kt)"], label="Bulgaria", color="#0000EE", marker="^")
plt.scatter(df_new_country_multiindex.index,
            df_new_country_multiindex["Belgium_CO2 emissions (kt)"], label="Belgium", linestyle="-", color="#a65628")


plt.xlabel("Year")
plt.ylabel("CO2 emissions (kt)")
plt.xticks(rotation=45)
plt.title("CO2 emissions (kt)", fontweight='bold')
plt.xlim(min(df_new_country_multiindex.index),
         max(df_new_country_multiindex.index))
plt.ticklabel_format(axis='y', style='plain')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

heatmap = plt.imshow(df_new_year_Belgium.corr(), cmap="PuBuGn")
plt.xticks(np.arange(len(df_new_year_Belgium.columns)),
           df_new_year_Belgium.columns, rotation=45)
plt.yticks(np.arange(len(df_new_year_Belgium.columns)),
           df_new_year_Belgium.columns)
plt.xlabel("Indicators")
plt.ylabel("Indicators")

# add colorbar legend to the heatmap
c_bar = plt.colorbar(heatmap)

# add a title to the plot
plt.title("Correlation between indicators - Belgium")

# show the plot
plt.show()

heatmap = plt.imshow(df_new_year_Belgium.corr(), cmap="RdYlBu")

plt.xticks(np.arange(len(df_new_year_Bangladesh.columns)),
           df_new_year_Bangladesh.columns, rotation=45)
plt.yticks(np.arange(len(df_new_year_Bangladesh.columns)),
           df_new_year_Bangladesh.columns)
plt.xlabel("Indicators")
plt.ylabel("Indicators")

# add colorbar legend to the heatmap
c_bar = plt.colorbar(heatmap)

# add a title to the plot
plt.title("Correlation between indicators - Bangladesh")

# show the plot
plt.show()
