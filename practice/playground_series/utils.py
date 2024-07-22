import os
import pandas as pd

def create_model_submission(df_test, pipeline, model_name):
    file_name = f"{model_name}_submission.csv"
    current_dir = os.getcwd()
    file_path = os.path.join(current_dir, file_name)
    y_pred = pipeline.predict(df_test)
    submission = pd.DataFrame(y_pred, index=df_test.index, columns=["Target"])
    submission.to_csv(file_path)