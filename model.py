import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pickle 

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# reading the dataset
df = pd.read_csv('players_22.csv')
df.head(5)

df.shape

# checking for missing values
df.isnull().any()

#lets handle the above missing values
col = []
for c in df.columns:
  missing_value = np.mean(df[c].isnull()) * 100
  if missing_value > 60:
    print('{} - {}%'.format(c, round(missing_value)))
    col.append(c)

print("\n We need to drop these columns: \n \n", col)

#now dropping the columns with missing values
df.drop(columns=['club_loaned_from', 'nation_team_id', 
                 'nation_position', 'nation_jersey_number', 
                 'player_tags', 'goalkeeping_speed', 
                 'nation_logo_url'], inplace=True)

#extracting the columns whose dtype is number and ignoring the ones with object dtype
dfCols = []
for c in df.columns:
  if df[c].dtype == 'int64':
    dfCols.append(c)

print(dfCols)

# lets look at the relation between player overall [rating] and wages
plt.figure(figsize=(6,5))
ax = sns.scatterplot(x = df['overall'], y=df['wage_eur'])
plt.xlabel("Overall")
plt.ylabel("Wage EUR")
plt.title("Overall and Wage", fontsize=18)
plt.show()

# how will the above observation look if we filtered the Intl reputation
plt.figure(figsize=(7, 5))
ax = sns.scatterplot(x =df['overall'], y = df['wage_eur'], hue = df['international_reputation'])
plt.xlabel("Overall") 
plt.ylabel("Wage EUR")
plt.title("Overall & wage", fontsize = 18)
plt.show()


# now lets look at the player reputation and value
fig, ax = plt.subplots(figsize=(7,5))
plt.scatter(x=df['international_reputation'], y=df['value_eur'] )
plt.xlabel("International Reputation") 
plt.ylabel("Value in EUR")
plt.title("Reputation & Value in EUR", fontsize = 15)

plt.show()

# now lets check player reputation and wages


fig, ax = plt.subplots(figsize=(7,5))
plt.scatter(x=df['international_reputation'], y=df['wage_eur'] )
plt.xlabel("International Reputation") 
plt.ylabel("Wage in EUR")
plt.title("Reputation & wages in EUR", fontsize = 15)
plt.show()

# these are the top 15 players by the overall metric
top_15 = df.nlargest(15, 'overall')

# finally lets check if overall score correlates with earnings, we will limit to only the top 15 players




fig, ax = plt.subplots(figsize=(8,5))

plt.scatter(top_15['potential'], top_15['wage_eur'] )
plt.text(top_15.iloc[0]['potential'], top_15.iloc[0]['wage_eur'], top_15.iloc[0]['short_name'])
# plt.text(top_15.iloc[1]['potential'], top_15.iloc[1]['wage_eur'], top_15.iloc[1]['short_name']) for better view
plt.text(top_15.iloc[2]['potential'], top_15.iloc[2]['wage_eur'], top_15.iloc[2]['short_name'])
# plt.text(top_15.iloc[3]['potential'], top_15.iloc[3]['wage_eur'], top_15.iloc[3]['short_name'])
plt.text(top_15.iloc[4]['potential'], top_15.iloc[4]['wage_eur'], top_15.iloc[4]['short_name'])
plt.text(top_15.iloc[5]['potential'], top_15.iloc[5]['wage_eur'], top_15.iloc[5]['short_name'])
plt.text(top_15.iloc[6]['potential'], top_15.iloc[6]['wage_eur'], top_15.iloc[6]['short_name'])
plt.text(top_15.iloc[7]['potential'], top_15.iloc[7]['wage_eur'], top_15.iloc[7]['short_name'])
plt.text(top_15.iloc[8]['potential'], top_15.iloc[8]['wage_eur'], top_15.iloc[8]['short_name'])
plt.text(top_15.iloc[9]['potential'], top_15.iloc[9]['wage_eur'], top_15.iloc[9]['short_name'])

ax.set_title("Potential vs Wages of top 15")
ax.set_ylabel('Wages in Eur')
ax.set_xlabel('Potential')

plt.show()

# lets check if mentality composure relates with the player rating
fig, ax = plt.subplots(figsize=(8,5))

plt.scatter(top_15['overall'], top_15['mentality_composure'])

plt.text(top_15.iloc[0]['overall'], top_15.iloc[0]['mentality_composure'], top_15.iloc[0]['short_name'])
plt.text(top_15.iloc[1]['overall'], top_15.iloc[1]['mentality_composure'], top_15.iloc[1]['short_name'])
plt.text(top_15.iloc[2]['overall'], top_15.iloc[2]['mentality_composure'], top_15.iloc[2]['short_name'])
plt.text(top_15.iloc[3]['overall'], top_15.iloc[3]['mentality_composure'], top_15.iloc[3]['short_name'])
plt.text(top_15.iloc[4]['overall'], top_15.iloc[4]['mentality_composure'], top_15.iloc[4]['short_name'])
plt.text(top_15.iloc[5]['overall'], top_15.iloc[5]['mentality_composure'], top_15.iloc[5]['short_name'])
plt.text(top_15.iloc[6]['overall'], top_15.iloc[6]['mentality_composure'], top_15.iloc[6]['short_name'])
plt.text(top_15.iloc[7]['overall'], top_15.iloc[7]['mentality_composure'], top_15.iloc[7]['short_name'])
plt.text(top_15.iloc[8]['overall'], top_15.iloc[8]['mentality_composure'], top_15.iloc[8]['short_name'])
plt.text(top_15.iloc[9]['overall'], top_15.iloc[9]['mentality_composure'], top_15.iloc[9]['short_name'])

ax.set_title("Overall Rating vs Mentality Composure")
ax.set_ylabel('Mentality Composure Rating')
ax.set_xlabel('Overall Rating')

plt.show()

corr_matrix = df.corr()
corr_matrix

# see how the dataset correlates to the ocerall rating
corr_matrix['overall'].sort_values(ascending= False)


# df.drop(columns=['sofifa_id', 
#         'player_url', 
#         'short_name', 
#         'long_name', 'player_positions',
#         'player_face_url', 'club_logo_url',
#         'club_flag_url'	, 'nation_flag_url',
#         'club_name', 'league_name',
#         'club_position', 'dob'
#         ], inplace=True)


## creating a new df with only numerical dtypes
newDf = df[['potential', 'value_eur', 'wage_eur', 
            'age',  'height_cm', 'weight_kg', 
            'pace',
          'shooting',
          'passing',
          'dribbling',
          'defending',
          'physic',
          'attacking_crossing',
          'attacking_finishing',
          'attacking_heading_accuracy',
          'attacking_short_passing',
          'attacking_volleys',
          'skill_dribbling',
          'skill_curve',
          'skill_fk_accuracy',
          'skill_long_passing',
          'skill_ball_control',
          'movement_acceleration',
          'movement_sprint_speed',
          'movement_agility',
          'movement_reactions',
          'movement_balance',
          'power_shot_power',
          'power_jumping',
          'power_stamina',
          'power_strength',
          'power_long_shots',
          'mentality_aggression',
          'mentality_interceptions',
          'mentality_positioning',
          'mentality_vision',
          'mentality_penalties',
          'mentality_composure',
          'defending_marking_awareness',
          'defending_standing_tackle',
          'defending_sliding_tackle',
          
            ]].copy()

newDf.dtypes

# changing all the dtypes to float64
newDf = newDf.astype(np.float64)


#dropping all the null values
# newDf = newDf.notna()
# newDf.isnull().any()

#dropping all the null values
# newDf = newDf.notna()
newDf = newDf.dropna()
newDf.isnull().any()

train_set, test_set= train_test_split(newDf, test_size= 0.2, random_state= 42)

y_train = train_set['overall'].copy()
x_train = train_set.drop(['overall'], axis=1)

#building the models
# labels = df['overall']
forest = RandomForestRegressor(random_state=42)
# forest.fit(newDf, labels)
forest.fit(x_train, y_train)

y_test = test_set['overall'].copy()
x_test = test_set.drop(['overall'], axis=1)
forestPred = forest.predict(x_test)
forest_mse = mean_squared_error(y_test, forestPred)
forest_rmse = np.sqrt(forest_mse)
forest_mse, forest_rmse

# lets tune the model by adding the parameters to learn from
forest_new = RandomForestRegressor(random_state=42, max_features=20)
forest_new.fit(x_test, y_test)

#checking the perfomance of the above model
newForestPred = forest_new.predict(x_test)
newForest_mse = mean_squared_error(y_test, newForestPred)
newForest_rmse = np.sqrt(newForest_mse)
newForest_mse, newForest_rmse


#creating the pickle file of our model
pickle.dump(forest_new, open("model.pkl", "wb"))

