# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

df = pd.read_csv("http://cdn.sundog-soft.com/SelfDriving/CarVideos.csv")

#get a snapshot of the dataset
df.head()
#get the end
df.tail()
#rows, cols
df.shape
#rows*cols
df.size
#get number of rows
len(df)
#get columns
df.columns
#get a series
df['Time']
#get the first five rows of the time series
df['Time'][:5]
#get a specific time value
df['Time'][4]
#get a subset of the df, this is a new dataframe
df[['Vehicle', 'Time']]
#get first five rows
df[['Vehicle', 'Time']][:5]
df.sort_values(['Time'])
#returns a series that counts the number of items in the df
vehicle_count = df['Vehicle'].value_counts()
#plot
vehicle_count.plot(kind='bar')

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

img = mpimg.imread('image_lane_c.jpg')
plt.imshow(img)

df = pd.read_csv("http://media.sundog-soft.com/SelfDriving/FuelEfficiency.csv")
gear_counts = df['# Gears'].value_counts()
gear_counts.plot(kind='bar')

import seaborn as sns
sns.set()
gear_counts.plot(kind='bar')

sns.distplot(df['CombMPG'])


df2 = df[['Cylinders', 'CityMPG', 'HwyMPG', 'CombMPG']]
df2.head()

sns.pairplot(df2, hue='Cylinders', height=2.5);

sns.scatterplot(x="Eng Displ", y="CombMPG", data=df)

sns.jointplot(x="Eng Displ", y="CombMPG", data=df)

sns.lmplot(x="Eng Displ", y="CombMPG", data=df)

sns.set(rc={'figure.figsize':(15,5)})
ax=sns.boxplot(x='Mfr Name', y='CombMPG', data=df)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)

ax=sns.swarmplot(x='Mfr Name', y='CombMPG', data=df)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)

#categorical data
ax=sns.countplot(x='Mfr Name', data=df)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)

df3 = df.pivot_table(index='Cylinders', columns='Eng Displ', values='CombMPG', aggfunc='mean')
sns.heatmap(df3)


