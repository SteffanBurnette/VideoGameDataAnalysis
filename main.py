import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from pandas.core.dtypes.common import is_numeric_dtype
from sklearn import linear_model
import ast
from sklearn import tree

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

df = pd.read_csv('games.csv') #Assigns the game.csv file to the dataframe

df.info() #Prints the database info
df.head() #Returns first 5 rows
print("This is the shape of the dataset: ", df.shape)
#df.shape #Returns the number of rows(x) and columns(y)

print(df.nunique()) #Count number of distinct elements in specified row.

#Replaces all missing data in the Rating/Team/Summary
#Data set with values
df['Rating'] = df['Rating'].replace(np.nan, 0.0)
df['Team'] = df['Team'].replace(np.nan, "['Unknown Team']")
df['Summary'] = df['Summary'].replace(np.nan, 'Unknown Summary')


#The sort_index() method sorts the DataFrame by the index.
#Drops all duplicates in dataset
df = df.drop_duplicates().sort_index()

# create a datetime object for the datetime module
dt = datetime.now()
# convert the datetime object to a string
dt_str = dt.strftime('%b %d, %Y')
print(dt_str)


df.loc[df['Release Date'] == 'releases on TBD']

df['Release Date'] = df['Release Date'].str.replace('releases on TBD', dt_str )
df['Release Date'] = pd.to_datetime(df['Release Date'], format='%b %d, %Y')
# format the datetime object as a string in the desired format
df['Release Date'] = df['Release Date'].dt.strftime('%Y-%-m-%-d')


# convert the date column to a datetime object
df['Release Date'] = pd.to_datetime(df['Release Date'])
# get the day from the date column
df['Day'] = df['Release Date'].dt.day
df['Month'] = df['Release Date'].dt.strftime('%b')
df['Year'] = df['Release Date'].dt.year
df['Week day'] = df['Release Date'].dt.day_name()

###COnverts data t usable dtypes
#df['Times Listed'] = df['Times Listed'].str.replace('K', '').astype(float)
#df['Number of Reviews'] = df['Number of Reviews'].str.replace('K', '').astype(float)
df['Plays'] = df['Plays'].str.replace('K', '').astype(float)
df['Playing'] = df['Playing'].str.replace('K', '').astype(float)
#df['Backlogs'] = df['Backlogs'].str.replace('K', '').astype(float)
#df['Wishlist'] = df['Wishlist'].str.replace('K', '').astype(float)
#df['Team'] = df['Team'].apply(lambda x: ast.literal_eval(x))
#df['Genres'] = df['Genres'].apply(lambda x: ast.literal_eval(x))

#Outputs top 5 rows
print(df[['Title', 'Release Date', 'Rating', 'Day', 'Month', 'Year', 'Week day']].head())
#outputs last 5 rows
print(df[['Title', 'Release Date', 'Rating', 'Day', 'Month', 'Year', 'Week day']].tail())

#for x in df['Rating']:
 #   if x >= 4.0:
  #      print(df[['Title', 'Release Date', 'Rating', 'Day', 'Month', 'Year', 'Week day']])

print(df.describe())

# create a sample DataFrame with a column containing multiple values
df_rate = pd.DataFrame({
    'Release Date': df['Release Date'].tolist(),
    'Rating': df['Rating'].tolist()
})
# use the explode method to transform the 'Rating' column
df_rate = df_rate.explode('Rating')
print(df_rate)

#Plots the df_rate dataframe
#df_rate.plot()
#plt.show()

#Hard testing top 10 games/genres
top_10_games = ['Fortnite', 'Minecraft', 'Grand Theft Auto V', 'Counter-Strike: Global Offensive', 'Apex Legends', 'League of Legends', 'Call of Duty: Warzone', 'Valorant', 'Dota 2', 'Roblox']
top_10_genres = ['Action', 'Shooter', 'Sports', 'Role-Playing', 'Adventure', 'Strategy', 'Simulation', 'Fighting', 'Racing', 'Puzzle']


top_rating = df[['Title','Rating']].sort_values(by = 'Rating', ascending = False)
top_rating = top_rating.loc[top_rating['Title'].isin(top_10_games)]
top_rating = top_rating.drop_duplicates()

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

sns.histplot(ax = axes[0], data = df['Rating'])
sns.barplot(ax = axes[1], data = top_rating, x = 'Rating', y = 'Title', palette = 'Blues_d')

axes[0].set_title('Distribution of ratings', pad = 10, fontsize = 15)
axes[0].set_xlabel('Rating', labelpad = 20)
axes[0].set_ylabel('Frequency', labelpad = 20)

axes[1].set_title('Top rated games in 2021', pad = 10, fontsize = 15)
axes[1].set_xlabel('Rating', labelpad = 20)
axes[1].set_ylabel('Title', labelpad = 20)
plt.tight_layout()

plt.show()







#MACHINE LEARNING
#df.columns
# select prediction target
#y = df.Rating
# select features
#features = ['Times Listed', 'Number of Reviews', 'Plays', 'Playing', 'Backlogs', 'Wishlist']
#X = df[features]

# Define model. Specify a number for random_state to ensure the same results each run
#melbourne_model = DecisionTreeRegressor(random_state=1)
# Fit model
#melbourne_model.fit(X, y)

#print("Making predictions for the following 5 houses:")
#print(X.head())
#print("The predictions are")
#print(melbourne_model.predict(X.head()))


df_playRate = pd.DataFrame({
    'Release Date': df['Release Date'].tolist(),
    'Rating': df['Rating'].tolist(),
    'Plays': df['Plays'].tolist(),
    'Playing': df['Playing'].tolist(),
})



X = df_rate[['Rating']]
y = df_rate[['Release Date']]



regr = linear_model.LinearRegression()
regr.fit(X, y)

predictedRate = regr.predict([[4.0]])

print(predictedRate)
#ATTEMPT at training the dataset
plt.plot(X, y)
plt.show()

plt.plot(predictedRate)
plt.show()


train_x = X[:80]
train_y = y[:80]

test_x = X[80:]
test_y = y[80:]

plt.scatter(train_x, train_y)
plt.show()

plt.scatter(test_x, test_y)
plt.show()



ohe_cars = pd.get_dummies(df_rate[['Release Date']])

#print(ohe_cars.to_string())


#I believe that this is summing all the rating together
#An outputting the result next to the most popular
#dates that released the most popular games

#CAUSING AN ERROR NEED TO FIX
#platform = df.groupby('Release Date').sum()['Rating'].reset_index()
#platform = platform.sort_values('Rating', ascending=False).head(10)
#print(platform)
#figure2 = plt.bar(platform, Y='Release Date', title="Most popular release dates")
#figure2.show()

print("The max value: ")
print(df_rate.loc[df_rate['Rating'].idxmax()])

print("The other max value: ")
print(df.groupby(['Release Date','Title'])['Rating'].max().head(20))

#Attempt to get all the info from the highest rating games
Highest_Rating = df.groupby(['Release Date','Title'])['Rating'].max()

print(Highest_Rating)

#Outputs the Top 10 highest ratings
print('\n'+'\n'+"These are the ten HIGHEST ratings: ")
print(df['Rating'].nlargest(n=10))

#Outputs the top 10 lowest ratings
print('\n'+'\n'+"These are the ten LOWEST ratings: ")
print(df['Rating'].nsmallest(n=10))




#Outputs the Top 10 highest Values from multiple columns Rating/Plays/Playing:
print('\n'+'\n'+"These are the Top 10 highest Values from multiple columns: ")
print(df.nlargest(n=10, columns=['Rating']))


#Outputs the Top 10 highest Values from Release Date:
print('\n'+'\n'+"These are the Top 10 highest Values from Release Date: ")
print(df.nlargest(n=10, columns=['Release Date']))

#This will return the top values per column as a new DataFrame:

#This loop will place all the numeric values from the
# DataFrame into the array
dfs = []

for col in df.columns:
    top_values = []
    if is_numeric_dtype(df[col]):
        top_values = df[col].nlargest(n=10)
        dfs.append(pd.DataFrame({col: top_values}).reset_index(drop=True))
pd.concat(dfs, axis=1)

print("This is the new dataframe with the highest ratings only")
#print(dfs)

#####################################################################
# convert Plays column to numeric data type
df['Plays'] = pd.to_numeric(df['Plays'], errors='coerce')

# fill NaN values with 0
df['Plays'] = df['Plays'].fillna(0)

# convert Playing column to numeric data type
df['Playing'] = pd.to_numeric(df['Playing'], errors='coerce')
# fill NaN values with 0
df['Playing'] = df['Playing'].fillna(0)
######################################################################################

#df.plot(x=df['Release Date'], y=df.nlargest(n=10, columns=['Rating']), kind="bar", figsize=(10, 9))
#plt.show()

# calculates the mean rating for each team
ratings_date = df.groupby('Month')['Rating'].nlargest(10)
# select only top ten teams by average rating
top_ten_rating = ratings_date.nlargest(10)

print(top_ten_rating)

# calculates the mean rating for each team
#Displays the games with the most plays and their ratings
#Grouped the ratings columns by the most number of plays

plays_stats = df.groupby('Rating')['Plays'].nlargest(10)
# select only top ten teams by average rating
top_ten_plays = plays_stats.nlargest(10)
print(top_ten_plays)


playing_stats = df.groupby('Rating')['Playing'].nlargest
#playing_stats = df.groupby('Playing')['Rating'].nlargest
top_ten_playing = playing_stats
print(top_ten_playing)

#TEST TEST TEST TEST TEST TEST TEST TEST TEST




avg_rating_by_team = df.groupby("Team")["Rating"].mean()

# Get the top 10 teams by average rating
top_teams = avg_rating_by_team.nlargest(10).index.tolist()

# Filter the dataset to include only games by the top 10 teams
top_teams_games = df[df["Team"].isin(top_teams)]

# Create a dictionary mapping each team to a unique color
# team_colors = {team: f"C{i}" for i, team in enumerate(top_teams)}
# Create a timeline plot of the top 10 teams' games
fig, ax = plt.subplots(figsize=(12, 8))

sns.scatterplot(data=top_teams_games, x="Release Date",
                y="Team", hue="Title", s=150, alpha=0.8, ax=ax)

# Set the title and axis labels
ax.set_title(
    "Top 10 Teams by Average Rating and Their Games Timeline", fontsize=18)
ax.set_xlabel("Release Date", fontsize=14)
ax.set_ylabel("Team", fontsize=14)
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=12)

# Add a grid to the plot
ax.grid(axis="y", alpha=0.3)

# Show the plot
plt.show()
######################################################
#ALTER

#top_rating_games = df[df["Release Date"].isin(top_ten_rating)]

#avg_rating_by_team = df.groupby("Team")["Rating"].mean()

# Get the top 10 teams by average rating
#top_teams = avg_rating_by_team.nlargest(10).index.tolist()

# Filter the dataset to include only games by the top 10 teams
#top_teams_games = df[df["Team"].isin(top_teams)]

# Create a dictionary mapping each team to a unique color
# team_colors = {team: f"C{i}" for i, team in enumerate(top_teams)}
# Create a timeline plot of the top 10 teams' games
#fig, ax = plt.subplots(figsize=(12, 8))
#sns.scatterplot(data=top_rating_games, x="Rating",
 #               y="Month", hue="Title", s=150, alpha=0.8, ax=ax)

# Set the title and axis labels
#ax.set_title(
 #   "Top 10 RATINGSSS by Average Rating and Their Games Timeline", fontsize=18)
#ax.set_xlabel("Release Date", fontsize=14)
#ax.set_ylabel("Rating", fontsize=14)
#ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=12)

# Add a grid to the plot
#ax.grid(axis="y", alpha=0.3)

# Show the plot
#plt.show()




#######################################################

ratedate = df.groupby('Release Date')['Rating'].nlargest(10)

top_dates = ratedate.nlargest(10)
print("The top dates: ")
print(top_dates)
print(top_dates.isnull())

stats = df.groupby('Rating')['Release Date']
datedate =stats.nlargest(10)
print(datedate)


df.plot.line(y=['Rating', 'Plays', 'Playing'], figsize=(15,10))
plt.show()

top_ten_rating.plot.line(y=['Rating', 'Plays', 'Playing'], figsize=(15,10))
plt.show()

# calculates the mean rating for each team

#df[[ 'Plays']].hist(figsize=(14, 9),bins=16,linewidth='1',edgecolor='k',grid=False)
#plt.show()





##***************MY GRAPHS************************
###########################################################################
#*******************************************************************
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

#GETS THE TOP 10 GAME TITLES AND THEIR RATINGS
#And shows their release dates in the legend

top_10_game_of_all_time = df[['Title', 'Rating', 'Release Date']].sort_values(by = 'Rating', ascending = False).nlargest(10, columns='Rating')

print(top_10_game_of_all_time)

fig, axes = plt.subplots(1, figsize=(16, 5))
sns.barplot(ax=axes, data=top_10_game_of_all_time, x='Rating', y='Title', palette='GnBu')
axes.set_title('Top rated Games', pad=10, fontsize=15)
axes.set_xlabel('Rating', labelpad=20)
axes.set_ylabel('Title', labelpad=20)
plt.legend(top_10_game_of_all_time['Release Date'])
plt.show()

#BOTTOM 10 PERFORMING GAMES
#Drops the ratings that are equal to 0
#The reason why i did this is because almost all of the
#0 rated games have not been released yet so there are no ratings/reviews

df.drop(df[df['Rating'] == 0].index, inplace=True)
#Had to specifically drop these two columns since the
#.drop_duplicates(inplace=True) method was not dropping
#them for some reason. These are duplicate columns.
df.drop(547, inplace=True)
df.drop(761, inplace=True)

top_10_worst_game_of_all_time = df[['Title', 'Rating', 'Release Date']].sort_values(by = 'Rating', ascending = False).nsmallest(10, columns='Rating')

print(top_10_worst_game_of_all_time)

fig, axes = plt.subplots(1, figsize=(16, 6))
sns.barplot(ax=axes, data=top_10_worst_game_of_all_time, x='Title', y='Rating', palette='YlOrBr')
axes.set_title('Least rated Games', pad=10, fontsize=15)
axes.set_xlabel('Title', labelpad=100)
axes.set_ylabel('Rating', labelpad=100)
plt.legend(top_10_worst_game_of_all_time['Release Date'])
fig.tight_layout()
plt.show()



####GRAPHS THE GAMES WITH THE MOST PLAYS
#THE DATA POINTS ARE THE TOP 10 MOST PLAYED
#GAMES (not the top 10 highest rated games)
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
top_played_game_of_all_time = df[['Title', 'Rating', 'Plays']].sort_values(by = 'Plays', ascending = False).nlargest(10, columns='Plays')


fig, ax = plt.subplots(figsize=(12, 8))

sns.scatterplot(data=top_played_game_of_all_time, x="Rating",
                y="Plays", hue="Title", s=150, alpha=0.8, ax=ax)

# Set the title and axis labels
ax.set_title(
    "Top 10 Most Played Games and Their Ratings", fontsize=18)
ax.set_xlabel("Rating", fontsize=14)
ax.set_ylabel("Plays", fontsize=14)
ax.legend(bbox_to_anchor=(0.8, 0.8), loc="upper left", fontsize=12)

# Add a grid to the plot
ax.grid(axis="y", alpha=0.3)

# Show the plot
plt.show()

print(top_played_game_of_all_time)

##################################
#Graphs the games with the most people that are still playing them
#Seems like some duplicates are still in the dataframe
#will need to readjust
top_playing_game_of_all_time = df[['Title', 'Rating', 'Playing']].sort_values(by = 'Playing', ascending = False).nlargest(10, columns='Playing')


fig, ax = plt.subplots(figsize=(12, 8))

sns.scatterplot(data=top_playing_game_of_all_time, x="Rating",
                y="Playing", hue="Title", s=150, alpha=0.8, ax=ax)

# Set the title and axis labels
ax.set_title(
    "Top 10 Games That are still being played and their Ratings", fontsize=18)
ax.set_xlabel("Rating", fontsize=14)
ax.set_ylabel("Playing", fontsize=14)
ax.legend(bbox_to_anchor=(0.8, 0.8), loc="upper left", fontsize=12)

# Add a grid to the plot
ax.grid(axis="y", alpha=0.3)

# Show the plot
plt.show()

print(top_playing_game_of_all_time)



# https://www.kaggle.com/datasets/arnabchaki/popular-video-games-1980-2023