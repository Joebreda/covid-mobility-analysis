import numpy as np
import csv
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz



def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y



#load and plot covid death data
covid_data_dict = {}
covid_skip_start_date = 24  #skip some early dates to have same date length with mobility data
covid_skip_end_date = -5  #end early to have same date length with mobility data
county_key_word = "King, Washington, US"


with open('time_series_covid19_deaths_US.csv','r') as csvfile:
    data = csv.reader(csvfile, delimiter = ',')
    header = next(data)
    for row in data:
    	covid_data_dict[row[10]] = np.array(row[12+covid_skip_start_date:covid_skip_end_date], int)

xticks_date_string = [date[:-3] for date in header[12+covid_skip_start_date:covid_skip_end_date]]
county_covid_cases = covid_data_dict[county_key_word]

xticks_date_index = range(0,len(county_covid_cases), 5)
xticks_date = [xticks_date_string[i] for i in xticks_date_index]

# fig = plt.figure(figsize=(24,8))
# plt.plot(county_covid_cases)
# plt.xticks(xticks_date_index, xticks_date, rotation=20) 
# plt.xlabel("date")
# plt.ylabel("death cases")
# plt.title("number of covid deaths in " + county_key_word)
# plt.show()

county_covid_cases_delta = np.diff(county_covid_cases)


# fig, axs = plt.subplots(2)

# fig.suptitle('Covid deaths')
# axs[0].plot(county_covid_cases_delta)
# axs[0].set_xticks(xticks_date_index, xticks_date)
# axs[0].set_xticklabels("date")
# axs[0].set_yticklabels("new death cases")
# # axs[0].title("number of new covid deaths in " + county_key_word)
# # axs[0].show()


#plot covid deaths diff

fig = plt.figure(figsize=(24,8))
plt.plot(county_covid_cases_delta)
plt.xticks(xticks_date_index, xticks_date, rotation=20) 
plt.xlabel("date")
plt.ylabel("new death cases")
plt.title("number of new covid deaths in " + county_key_word)
plt.show()


county_covid_cases_delta = butter_lowpass_filter(county_covid_cases_delta, 0.2,1)

# axs[1].plot(county_covid_cases_delta)
# axs[1].set_xticks(xticks_date_index, xticks_date)
# axs[1].set_xticklabels("date")
# axs[1].set_yticklabels("new death cases")
# axs[1].title("number of new covid deaths in " + county_key_word)
# axs[1].show()
# plt.setp(axs.get_xticklabels(), rotation=20, horizontalalignment='right')

fig = plt.figure(figsize=(24,8))
plt.plot(county_covid_cases_delta)
plt.xticks(xticks_date_index, xticks_date, rotation=20) 
plt.xlabel("date")
plt.ylabel("new death cases")
plt.title("filtered number of new covid deaths in " + county_key_word)
plt.show()



#load and plot mobility data
with open('mobility_king_county.csv','r') as csvfile:
    data = csv.reader(csvfile, delimiter = ',')
    mobility_data = list(data)


mobility_data = np.array(mobility_data)
mobility_data = mobility_data[:,5:11]
mobility_data = np.array(mobility_data, int)

recreation_mobility = mobility_data[:, 0]
grocery_mobility = mobility_data[:, 1]
park_mobility = mobility_data[:, 2]
transit_mobility = mobility_data[:, 3]
work_mobility = mobility_data[:, 4]
residential_mobility = mobility_data[:, 5]


residential_mobility = butter_lowpass_filter(residential_mobility, 0.2, 1)


fig = plt.figure(figsize=(24,8))
plt.plot(mobility_data)
plt.xticks(xticks_date_index, xticks_date, rotation=20) 
plt.xlabel("date")
plt.ylabel("mobility change")
plt.title("Mobility data in King County ")
plt.show()

fig = plt.figure(figsize=(24,8))
plt.plot(residential_mobility)
plt.xticks(xticks_date_index, xticks_date, rotation=20) 
plt.xlabel("date")
plt.ylabel("residential rate change")
plt.title("filtered residential rate in King County ")
plt.show()


#DTW
print(county_covid_cases_delta.shape)
print(mobility_data.shape)
from dtw import *
alignment = dtw(county_covid_cases_delta, residential_mobility, keep_internals=True)

alignment.plot(type="threeway")

dtw(county_covid_cases_delta, residential_mobility, keep_internals=True, step_pattern=rabinerJuangStepPattern(6, "c")).plot(type="twoway",offset=-2)

## See the recursion relation, as formula and diagram
print(rabinerJuangStepPattern(6,"c"))
rabinerJuangStepPattern(6,"c").plot()






