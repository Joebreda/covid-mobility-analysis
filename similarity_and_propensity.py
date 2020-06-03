import pandas as pd 
import numpy as np 
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# clean social metrics
social_metrics = pd.read_csv('data/county_demograohics.csv')
social_metrics['POP_ESTIMATE_2019'] = social_metrics['POP_ESTIMATE_2019'].str.replace(',','').astype(int)
social_metrics['area_name'] = social_metrics['area_name'].map(lambda x: x.split(',')[0])
social_metrics['Median_Household_Income_2018'] = social_metrics['Median_Household_Income_2018'].str.replace(',','').astype(int)
social_metrics['amount_land'] = pd.to_numeric(social_metrics['amount_land'])
gov = {'GOP': 0,'DEM': 1}
social_metrics['Government'] = [gov[item] for item in social_metrics['Government']] 
social_metrics['Government'] = social_metrics['Government'].astype(int)
social_metrics['density'] = social_metrics['POP_ESTIMATE_2019']/social_metrics['amount_land']
print(social_metrics.sort_values(by=['density'], ascending=False).head(20))


covid_cases_path = 'data/time_series_covid19_confirmed_US.csv'
covid_cases_df = pd.read_csv(covid_cases_path)

print(covid_cases_df)

# test if the largest county in all 50 states had above a certain amount of cases normalized to the population size and call this treatment
# regress this on the social metrics to generate scores and match counties by scores and compare their intercounty distant to largest city distance
#for state in social_metrics.
print(social_metrics.columns)
largest_city_by_state = {}
severity = []
total_cases_list = []
treatment = {}
for state in social_metrics.Stabr.unique():
    state_metrics = social_metrics.loc[social_metrics['Stabr'] == state]
    state_metrics = state_metrics.sort_values(by=['POP_ESTIMATE_2019'], ascending=False)
    name = state_metrics['area_name'].iloc[0]
    largest_city_by_state[state] = name
    fips = state_metrics['FIPStxt'].iloc[0]
    population = state_metrics['POP_ESTIMATE_2019'].iloc[0]
    total_cases = covid_cases_df[covid_cases_df['FIPS'] == fips].iloc[:,-1].values[0]
    severity.append(total_cases/population) 
    total_cases_list.append(total_cases)
    treatment[state] = total_cases
    treatment_threshold = 7000
    if total_cases > treatment_threshold:
        treatment[state] = 1
    else:
        treatment[state] = 0
# plot total cases in all 50 most populated counties per state to show knee at 7000 confirmed cases
'''
plt.plot(np.arange(len(total_cases_list)), sorted(total_cases_list))
plt.title('Number of Total Confirmed Cases in the Most Populated County by State')
plt.xlabel('Largest County Per State')
plt.ylabel('Number of Confirmed Cases')
plt.show()
'''
# plot totalcases/population and total cases sorted by severity
'''
fig, axs = plt.subplots(2, 1, figsize=(12,12))
severity, total_cases_list = (list(t) for t in zip(*sorted(zip(severity, total_cases_list))))
axs[0].plot(np.arange(len(severity)), severity)
axs[1].plot(np.arange(len(total_cases_list)), total_cases_list)
plt.show()
'''
social_metrics['treatment'] = [treatment[state] for state in social_metrics['Stabr']] 
print(social_metrics)



features = social_metrics[['Unemployment_rate_2019', 'Median_Household_Income_2018', 'Government', 'density']]
treatment_status = social_metrics[['treatment']]
names = social_metrics[['FIPStxt', 'area_name']]
print(features.values)
print(treatment_status.values)

clf = LogisticRegression(random_state=0).fit(features, treatment_status)
propensity_scores = clf.predict_proba(features)
propensity_scores = propensity_scores[:,0]

features_and_score = pd.concat([social_metrics, pd.DataFrame(propensity_scores, columns=['propensity_score'])], axis=1)
print(features_and_score)
features_and_score.to_csv('demographics_and_score.csv', index=False)


#plt.plot(social_metrics.index, sorted(social_metrics[['density']].values), 'o')
# density knee is around 0.3 
#plt.plot(social_metrics.index, sorted(social_metrics[['Unemployment_rate_2019']].values), 'o')
# one knee at 2.25 and another at 6
#plt.plot(social_metrics.index, sorted(social_metrics[['Median_Household_Income_2018']].values), 'o')
# one knee at 34,000 and another at 70,000
def plot_clustering(n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
    labels = kmeans.labels_.astype(float)

    fig, axs = plt.subplots(2, 2, figsize=(12,12))
    y, colors = (list(t) for t in zip(*sorted(zip(social_metrics[['density']].values, labels))))
    axs[0, 0].scatter(social_metrics.index, y, c=colors)
    axs[0, 0].set_title('Distribution of Clusters in Density Domain')
    y, colors = (list(t) for t in zip(*sorted(zip(social_metrics[['Unemployment_rate_2019']].values, labels))))
    axs[0, 1].scatter(social_metrics.index, y, c=colors)
    axs[0, 1].set_title('Distribution of Clusters in Unemployment Rate Domain')
    y, colors = (list(t) for t in zip(*sorted(zip(social_metrics[['Median_Household_Income_2018']].values, labels))))
    axs[1, 0].scatter(social_metrics.index, y, c=colors)
    axs[1, 0].set_title('Distribution of Clusters in Meadian Household Income Domain')
    y, colors = (list(t) for t in zip(*sorted(zip(social_metrics[['Government']].values, labels))))
    axs[1, 1].scatter(social_metrics.index, y, c=colors)
    axs[1, 1].set_title('Distribution of Clusters of Political Stance')

    for ax in axs.flat:
        ax.set(xlabel='Counties', ylabel='Metric')
    for ax in axs.flat:
        ax.label_outer()
    plt.show()


plot_clustering(3)