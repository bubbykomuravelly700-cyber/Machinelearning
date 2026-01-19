import pandas as pd

def main():
    df = pd.read_excel(r"C:\Users\rishvik\Downloads\Lab Session Data.xlsx",sheet_name="thyroid0387_UCI")
    print(df.isnull().sum())

main()
