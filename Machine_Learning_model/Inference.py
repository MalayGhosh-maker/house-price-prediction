from Machine_Learning_model.LoadAndPreprocess import load_and_analysis
from Machine_Learning_model.LoadAndPreprocess import data_process
from Machine_Learning_model.TrainingAndEvaluation import train_evaluation
import pandas as pd
import joblib

if __name__ == "__main__":

    df=load_and_analysis()
    sc_x_train, y_train, sc_x_eval, y_eval, sc_x_test, y_test,sc,encoding_artifacts, df_final_artifacts=data_process(df)
    model=train_evaluation(sc_x_train, y_train, sc_x_eval, y_eval, sc_x_test, y_test)

    artifacts ={
        "model": model,
        "scaler": sc,
        "feature_columns": list(df_final_artifacts.columns),
        **encoding_artifacts
    }

    joblib.dump(artifacts, "house_price_model.joblib")
    print("Acrtifacts saved successfully.")

def preprocess_single_house(df,artifacts):

    df = df.copy()
    df['Garage_Yes'] = artifacts['garage_ohe'].transform(df[['Garage']]).ravel()
    df['condition_frequency'] = (
        df['Condition'].map(artifacts['condition_counts']).fillna(0)
    )
    df['Location_encoded'] = (
        df['Location'].map(artifacts['location_counts']).fillna(0)
    )

    df_processed = artifacts['poly_feat'].transform(df[artifacts['feature_columns']])

    return artifacts['scaler'].transform(df_processed)

