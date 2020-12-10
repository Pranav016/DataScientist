# Reading the test file having the data of footballers and target being their
# overall skill score ( 1-100 ).
import pandas as pd
from datascientist.feature_selection.filter_based_selection import ChiSquare

player_df = pd.read_csv("datascientist/feature_selection/test/CSV/data.csv")

# Taking only those columns which have numerical or categorical values since
# feature selection with Pearson Correlation can be performed on numerical
# data.
numcols = ['Overall', 'Crossing', 'Finishing', 'ShortPassing', 'Dribbling',
           'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed',
           'Agility',  'Stamina', 'Volleys', 'FKAccuracy', 'Reactions',
           'Balance', 'ShotPower', 'Strength', 'LongShots', 'Aggression',
           'Interceptions']
catcols = ['Preferred Foot', 'Position', 'Body Type', 'Nationality',
           'Weak Foot']
player_df = player_df[numcols+catcols]

# encoding categorical values with one-hot encoding.
traindf = pd.concat([player_df[numcols],
                     pd.get_dummies(player_df[catcols])], axis=1)
features = traindf.columns

# dropping rows with Nan values
traindf = traindf.dropna()
traindf = pd.DataFrame(traindf, columns=features)

# Separating features(X) and target(y).
y = traindf['Overall']
X = traindf.copy()
X = X.drop(['Overall'], axis=1)

Col_sel = ChiSquare(X, y)

# using chi2_score method with different parameter values.
score1 = Col_sel.chi2_score()

score2 = Col_sel.chi2_score(sort=True)

score3 = Col_sel.chi2_score(sort=True, reset_index=True)

# using top_chi2_featurenames method with different parameter values.
topfeatname1 = Col_sel.top_chi2_featurenames()

topfeatname2 = Col_sel.top_chi2_featurenames(feat_num=15)

topfeatname3 = Col_sel.top_chi2_featurenames(feat_num=30)

# using top_chi2_features method with different parameter values.
X_mod1 = Col_sel.top_chi2_features()

X_mod2 = Col_sel.top_chi2_features(feat_num=15)

X_mod3 = Col_sel.top_chi2_features(feat_num=30)
