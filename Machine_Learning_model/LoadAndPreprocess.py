import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures


def load_and_analysis():

    df = pd.read_csv("house_price_dataset_india_12k.csv")
    df.head(2)
    df.describe()
    df.corr(numeric_only=True).sort_values("Market_Price_INR", ascending=False)
    df.info()
    df.isna().sum()
    df=df.drop_duplicates()

    Q1 = df['Market_Price_INR'].quantile(0.25)
    Q3 = df['Market_Price_INR'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter rows that are outliers
    outliers = df[(df['Market_Price_INR'] < lower_bound) | (df['Market_Price_INR'] > upper_bound)]
    outliers["Market_Price_INR"].sort_values(ascending=False)
    df = df[df["Market_Price_INR"]<30000000]

    return df

def data_process(df):

    x = df.drop("Market_Price_INR", axis=1)
    y = df["Market_Price_INR"]
    
    x_train_eval,x_test,y_train_eval,y_test = train_test_split(x,y,test_size=0.15,random_state=42)
    x_train,x_eval,y_train,y_eval = train_test_split(x_train_eval,y_train_eval,test_size=0.15,random_state=42)

    price_per_sqft_mean = x_train.groupby("City")["Price_per_sqft_INR"].mean()
    global_mean = x_train["Price_per_sqft_INR"].mean()

    x_train["City_Encoded"] = x_train["City"].map(price_per_sqft_mean)
    x_eval["City_Encoded"] = x_eval["City"].map(price_per_sqft_mean).fillna(global_mean)
    x_test["City_Encoded"] = x_test["City"].map(price_per_sqft_mean).fillna(global_mean)

    df['Locality_Tier'].value_counts()
    preprocessor = ColumnTransformer(
        transformers=[
            (
                'ohe',
                OneHotEncoder(drop='first',handle_unknown='ignore', sparse_output=False),
                ['Locality_Tier','Furnishing'],
            )
        ],
        remainder='passthrough',  # Keeps all your numeric columns (area, bedrooms, etc.)!
    )

    # 3. Tell Scikit-learn to output a pandas DataFrame instead of a NumPy array
    preprocessor.set_output(transform='pandas')

    x_train=preprocessor.fit_transform(x_train)
    x_eval=preprocessor.transform(x_eval)
    x_test=preprocessor.transform(x_test)

    x_train = x_train.drop(columns=["remainder__City"])
    x_eval = x_eval.drop(columns=["remainder__City"])
    x_test = x_test.drop(columns=["remainder__City"])

    x_train = x_train.drop(columns=["remainder__House_ID"])
    x_eval = x_eval.drop(columns=["remainder__House_ID"])
    x_test = x_test.drop(columns=["remainder__House_ID"])


    poly_conv = PolynomialFeatures(degree=2,include_bias=False)
    sc_x_train_poly = poly_conv.fit_transform(x_train)
    sc_x_eval_poly = poly_conv.transform(x_eval)
    sc_x_test_poly = poly_conv.transform(x_test)

    encoding_artifacts={
        "price_per_sqft_mean": price_per_sqft_mean.to_dict(),
        "global_mean": global_mean,
        "preprocessor": preprocessor,
        "poly_feat": poly_conv
    }

    df_latest = x_train

    return sc_x_train_poly, y_train, sc_x_eval_poly, y_eval, sc_x_test_poly, y_test, encoding_artifacts,df_latest












