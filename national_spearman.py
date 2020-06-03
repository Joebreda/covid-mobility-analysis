import pandas as pd 
import numpy as np 



def google_county_naming_2_covid_county_naming(county_name):
    stripped_county_name = str(county_name).replace("New York County", "New York City")
    stripped_county_name = str(stripped_county_name).replace(" County", "")
    stripped_county_name = str(stripped_county_name).replace(" Parish", "")
    stripped_county_name = str(stripped_county_name).replace(" Borough", "")
    stripped_county_name = str(stripped_county_name).replace("Doña Ana", "Dona Ana")
    return stripped_county_name


google_data_path = "data/Google_Global_Mobility_Report.csv"
google_df = pd.read_csv(google_data_path)
google_df = google_df.loc[google_df['country_region'] == "United States"]

covid_cases_path = 'data/time_series_covid19_confirmed_US.csv'
covid_cases_df = pd.read_csv(covid_cases_path)


print(google_df)
print(google_df.columns)
print(covid_cases_df)

print(len(google_df.sub_region_1.unique()))
print(google_df.sub_region_1.unique())





# correlate all county mobilities to all county cases
# test if the largest county in all 50 states had above a certain amount of cases normalized to the population size and call this treatment
# regress this on the social metrics to generate scores and match counties by scores and compare their intercounty distant to largest city distance
