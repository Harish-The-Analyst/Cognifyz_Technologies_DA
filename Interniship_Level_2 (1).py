#!/usr/bin/env python
# coding: utf-8

# In[1]:


##### **Project Type**    - EDA
##### **Organization**    - Cognifyz Technologies
##### **Contribution**    - Individual level
##### **Member Name -** Harish Sharma
##### **Level -** 2


# In[1]:


# Import all the packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


# read the file
fp='C:\\Users\\hp\\OneDrive\\Desktop\\Cognify Internship\\Dataset  - Copy (2).csv'
idf=pd.read_csv(fp)
idf


# In[3]:


# first five rows
idf.head()


# In[4]:


# last five rows
idf.tail()


# In[5]:


# rows * columns
idf.size


# In[6]:


#(rows,columns)
idf.shape


# In[7]:


# seperate categorical and numerical column
cat=idf.select_dtypes(include=['object']).columns
num=idf.select_dtypes(exclude=['object']).columns
print('cat:',cat)
print('num:',num)


# In[8]:


# fill the missing values with mode because it is categorical column
idf['Cuisines']=idf['Cuisines'].fillna(idf['Cuisines'].mode()[0])


# In[9]:


# again check any missing values are available or not
idf.isnull().sum()


# ## Level:-2

# $Task:-1$
#     
# - **Analyze the distribution of aggregate
# ratings and determine the most common
# rating range.**

# In[10]:


# save the rating in one variable
rating=idf['Aggregate rating'].head()
rating


# In[11]:


# Histogram of Aggregate rating among the restaurents
plt.figure(figsize=(5,5))
plt.hist(rating,bins=5)
plt.xlabel("Rating Range")
plt.ylabel("Number of restaurents")
plt.title('Distribution of aggregate rating among the restaurents')
plt.show()


# In[12]:


rating_ranges=pd.cut(idf['Aggregate rating'],bins=[0,1,2,3,4,5], labels=['0-1','1-2','2-3','3-4','4-5'])


# In[13]:


rating_ranges.head()


# - Calculate the average number of votes
# received by restaurants.

# In[14]:


# Average votes received by the restaurent
avg_votes=idf['Votes'].mean()
avg_votes


# **Task:-2**
#     
# - Identify the most common combinations of
# cuisines in the dataset.

# In[15]:


# value counts of Cuisines
idf['Cuisines'].value_counts()


# In[16]:


# Cuisines 
idf['Cuisines']


# In[17]:


# Combinations of cuisines in the dataset
import itertools

idf['Cuisines'] = idf['Cuisines'].str.split(',')
unique_combinations = []
for i in idf['Cuisines']:
    unique_combinations.extend(set(combo) for combo in itertools.combinations(i, 2))
combination_counts = pd.Series(unique_combinations).value_counts()
print(combination_counts.head())




# - Determine if certain cuisine combinations
# tend to have higher ratings.

# In[18]:


idf=idf.dropna(subset=['Cuisines','Aggregate rating'])


# In[19]:


idf.head(2)


# In[20]:


# save the cuisines in a variable
rat=idf['Cuisines']
rat


# In[21]:


import pandas as pd

# Assuming 'idf' is your DataFrame
idf['Cuisines'] = idf['Cuisines'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)

# Display the updated DataFrame
print(idf['Cuisines'])



# In[22]:


avg_rating=idf.groupby('Cuisines')['Aggregate rating'].mean()
avg_rating


# In[23]:


# Average rating in descending order
avg_rating=avg_rating.sort_values(ascending=False)
avg_rating


# In[24]:


# Combination of Cuisines
print('The Cuisines Combination that have higher ratings:')
print(avg_rating.head())


# **Task:-3**
#     
# - Plot the locations of restaurants on a
# map using longitude and latitude
# coordinates.

# In[25]:


# import the packages
import plotly.express as px


# In[26]:


# plot the restaurents on the map
fig = px.scatter_mapbox(
     idf,
     lat='Latitude',
     lon='Longitude',
     hover_name='Restaurant Name',
     hover_data=['Cuisines'],
     color_discrete_sequence=['green'],
     zoom=5,
)


# In[27]:


fig.update_layout(
    mapbox_style="open-street-map",
    margin={"r": 0, "t": 0, "l": 0, "b":0},
)


# - Identify any patterns or clusters of
# restaurants in specific areas.

# In[28]:


# import the package
from sklearn.cluster import KMeans


# In[29]:


X=idf[['Latitude','Longitude']]
num_cluster=5


# In[30]:


# k mean clustering
kmeans=KMeans(n_clusters=num_cluster,n_init=10,random_state=42)
idf['cluster']=kmeans.fit_predict(X)


# In[31]:


# plot on the map
fig=px.scatter_mapbox(
    idf,
    lat='Latitude',
    lon='Longitude',
    hover_name='Restaurant Name',
    hover_data=['Cuisines','Country Code'],
    color='cluster',
    color_continuous_scale='reds',
    zoom=5,
)


# In[32]:


fig.update_layout(
    mapbox_style='carto-positron',
    margin={'r':0,"t":0,"l":0,'b':0}
)
fig.show()


# 
#     

# In[33]:


# clustering

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X = idf[['Latitude', 'Longitude']]

num_clusters = 5

kmeans = KMeans(n_clusters=num_clusters,n_init=10, random_state=42)
idf['Cluster'] = kmeans.fit_predict(X)

# Plotting the clusters
plt.scatter(idf['Longitude'], idf['Latitude'], c=idf['Cluster'], cmap='rainbow')
plt.title('Restaurant Clusters')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()


# - Identify if there are any restaurant chains
# present in the dataset

# In[34]:


idf.head(2)


# In[35]:


res_count=idf['Restaurant Name'].value_counts()


# In[36]:


potential_chains=res_count[res_count > 1].index


# In[37]:


print("Potential restaurant chains:")
for chain in potential_chains:
    print(f"-{chain}")


# - Analyze the ratings and popularity of
# different restaurant chains.

# In[38]:


idf=idf[idf['Aggregate rating'].notnull()]


# In[39]:


chain_stats=idf.groupby('Restaurant Name').agg({
    'Aggregate rating':'mean',
    'Votes':'sum',
    'Cuisines':'count'
}).reset_index()


# In[40]:


chain_stats.columns=['Restaurant Name','Average rating','Total Votes','Number of Location']


# In[41]:


chain_stats=chain_stats.sort_values(by='Total Votes',ascending=False)


# In[42]:


print("Restaurant Chain Rating and Popularity Analysis (Sorted by Total Votes):")
print(chain_stats.to_string(index=False,justify='center'))


# In[ ]:




