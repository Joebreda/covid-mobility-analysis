import numpy as np
import csv
from numpy import linalg as LA
import matplotlib.pyplot as plt


#load and plot covid cases data
covid_data_dict = {}
skip_date_for_plot = 0  #skip some early dates that have 0 cases
county_key_word = "King, Washington, US"


with open('time_series_covid19_confirmed_US.csv','r') as csvfile:
    data = csv.reader(csvfile, delimiter = ',')
    header = next(data)
    for row in data:
    	covid_data_dict[row[10]] = np.array(row[11+skip_date_for_plot:], int)

date_string = [date[:-3] for date in header[11+skip_date_for_plot:]]

county_covid_cases = covid_data_dict[county_key_word]

xticks_date_index = range(0,len(county_covid_cases),5)
xticks_date = [date_string[i] for i in xticks_date_index]

fig = plt.figure(figsize=(24,8))
plt.plot(county_covid_cases)
plt.xticks(xticks_date_index, xticks_date, rotation=20) 
plt.xlabel("date")
plt.ylabel("confirmed cases")
plt.title("number of covid cases in " + county_key_word)
plt.show()







