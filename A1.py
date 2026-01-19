import pandas as pd
import numpy as np

def load_purchase_data(file_path):
    df = pd.read_excel(file_path, sheet_name="Purchase data")
    X = df[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values
    y = df["Payment (Rs)"].values.reshape(-1, 1)
    return X, y

def calculate_rank(matrix):
    return np.linalg.matrix_rank(matrix)

def calculate_cost(X, y):
    X_pinv = np.linalg.pinv(X)
    cost = X_pinv @ y
    return cost

def main():
    X, y = load_purchase_data(r"C:\Users\rishvik\Downloads\Lab Session Data.xlsx")
    rank = calculate_rank(X)
    cost = calculate_cost(X, y)

    print("Dimensionality:", X.shape[1])
    print("Number of vectors:", X.shape[0])
    print("Rank of feature matrix:", rank)
    print("Cost of Candies, Mangoes, Milk:", cost.flatten())

main()
