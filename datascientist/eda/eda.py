import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy

data = pd.read_csv('')

def eda(df):
    print("Exploratory Data Analysis for Data:\n")
    print("Head of Data: ", df.head())
    print("===========================================================================")
    print("Types present in data frame: ", df.dtypes)
    print("===========================================================================")
    print("Correlation between different columns of Dataframe: ", df.corr())
    print("===========================================================================")
    print("Null values: ", df.isnull().sum())
    print("===========================================================================")
    sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
    sns.pairplot(df)
    
eda_(data)