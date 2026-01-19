import pandas as pd

def main():
    df = pd.read_excel(r"C:\Users\rishvik\Downloads\Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")

    print("Data Types:\n", df.dtypes)
    print("Missing Values:\n", df.isnull().sum())
    print("Numeric Summary:\n", df.describe())

main()
