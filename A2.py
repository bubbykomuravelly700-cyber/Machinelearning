import pandas as pd

def clas_custom(df):
    return ["RICH" if p > 200 else "POOR" for p in df["Payment (Rs)"]]

def main():
    df = pd.read_excel(r"C:\Users\rishvik\Downloads\Lab Session Data.xlsx", sheet_name="Purchase data")
    df["Status"] = clas_custom(df)
    print(df[["Payment (Rs)", "Status"]])

main()
