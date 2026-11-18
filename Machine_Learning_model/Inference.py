from pathlib import Path
from Machine_Learning_model.LoadAndPreprocess import load_and_analysis
from Machine_Learning_model.LoadAndPreprocess import data_process
from Machine_Learning_model.TrainingAndEvaluation import train_evaluation
import pandas as pd
import joblib

MODEL_PATH = Path(__file__).resolve().parent / "house_price_model.joblib"

if __name__ == "__main__":

    df=load_and_analysis()
    x_train, y_train, x_eval, y_eval, x_test, y_test, encoding_artifacts,df_latest=data_process(df)
    model=train_evaluation(x_train, y_train, x_eval, y_eval, x_test, y_test)

    artifacts ={
        "model": model,
        "feature_columns": list(df_latest.columns),
        **encoding_artifacts
    }

    joblib.dump(artifacts, MODEL_PATH)
    print("Artifacts saved successfully.")

def preprocess_single_house(df,artifacts):

    df = df.copy()

    df['City_Encoded'] = (
        df['City'].map(artifacts['price_per_sqft_mean']).fillna(artifacts['global_mean'])
    )

    df = artifacts['preprocessor'].transform(df)

    df_processed = artifacts['poly_feat'].transform(df[artifacts['feature_columns']])

    return df_processed

