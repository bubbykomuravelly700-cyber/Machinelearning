import pandas as pd
import numpy as np

def main():
    df = pd.read_excel(r"C:\Users\rishvik\Downloads\Lab Session Data.xlsx",sheet_name="thyroid0387_UCI")

    numeric_cols = df.select_dtypes(include=np.number)

    df[numeric_cols.columns] = (numeric_cols - numeric_cols.min()) / (numeric_cols.max() - numeric_cols.min())

    print("Normalization completed successfully")
    print(df.head())

main()
