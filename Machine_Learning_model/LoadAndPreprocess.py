import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import TargetEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

def load_and_analysis():

    df = pd.read_csv("House Price Prediction Dataset.csv")

    df.corr(numeric_only=True)['Price'].sort_values(ascending=False)
    df.info()
    df.isna().sum()

    Q1 = df['Price'].quantile(0.25)
    Q3 = df['Price'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter rows that are outliers
    outliers = df[(df['Price'] < lower_bound) | (df['Price'] > upper_bound)]
    outliers

    return df

def data_process(df):

    df_train = df[df['YearBuilt'].between(1900,2000)]
    df_eval = df[df['YearBuilt'].between(2001,2015)]
    df_test = df[df['YearBuilt'].between(2016,2023)]

    condition_counts = df_train['Condition'].value_counts()
    df_train['condition_frequency'] = df_train['Condition'].map(condition_counts)
    df_eval['condition_frequency'] = df_eval['Condition'].map(condition_counts)
    df_test['condition_frequency'] = df_test['Condition'].map(condition_counts)


    ohe = OneHotEncoder(drop='first', sparse_output=False)
    df_train['Garage_Yes'] = ohe.fit_transform(df_train[['Garage']]).ravel()
    df_eval['Garage_Yes'] = ohe.transform(df_eval[['Garage']]).ravel()
    df_test['Garage_Yes'] = ohe.transform(df_test[['Garage']]).ravel()

    #meanPrice = df_train.groupby('Location')['Price'].mean()
    #globalMean = df_train['Price'].mean()
    #cols = ['Location']
    #te = TargetEncoder(target_type='continuous', random_state=42)
    #df_train['Location_encoded'] = te.fit_transform(df_train[cols],df_train['Price'])

    location_count = df_train['Location'].value_counts()

    df_train['Location_encoded'] = df_train['Location'].map(location_count)
    df_eval['Location_encoded'] = df_eval['Location'].map(location_count).fillna(0)
    df_test['Location_encoded'] = df_test['Location'].map(location_count).fillna(0)

    df_train_processed = df_train.drop(columns=['Location','Condition','Garage','Id'])
    df_eval_processed = df_eval.drop(columns=['Location','Condition','Garage','Id'])
    df_test_processed = df_test.drop(columns=['Location','Condition','Garage','Id'])

    x_train = df_train_processed.drop('Price',axis=1)
    y_train = df_train_processed['Price']

    x_eval = df_eval_processed.drop('Price',axis=1)
    y_eval = df_eval_processed['Price']

    x_test = df_test_processed.drop('Price',axis=1)
    y_test = df_test_processed['Price']

    poly_feat = PolynomialFeatures(degree=2,include_bias=False)
    x_train_poly = poly_feat.fit_transform(x_train)
    x_eval_poly = poly_feat.transform(x_eval)
    x_test_poly = poly_feat.transform(x_test)

    sc = StandardScaler()
    sc_x_train = sc.fit_transform(x_train_poly)
    sc_x_eval = sc.transform(x_eval_poly)
    sc_x_test = sc.transform(x_test_poly)

    encoding_artifacts={
        "garage_ohe": ohe,
        "condition_counts": condition_counts.to_dict(),
        "location_counts": location_count.to_dict(),
        "poly_feat": poly_feat
    }

    df_final_artifacts = df_train_processed.drop("Price", axis=1)

    return sc_x_train, y_train, sc_x_eval, y_eval, sc_x_test, y_test, sc, encoding_artifacts, df_final_artifacts












