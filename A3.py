import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

def m_mean(data):
    return sum(data) / len(data)

def m_variance(data):
    mean = m_mean(data)
    return sum((x - mean) ** 2 for x in data) / len(data)

def exec_time(func, data):
    times = []
    for _ in range(10):
        start = time.time()
        func(data)
        times.append(time.time() - start)
    return sum(times) / len(times)

def main():
    df = pd.read_excel(r"C:\Users\rishvik\Downloads\Lab Session Data.xlsx", sheet_name="IRCTC Stock Price")
    price = df["Price"].values

    print("Mean (NumPy):", np.mean(price))
    print("Variance (NumPy):", np.var(price))
    print("Mean (Manual):", m_mean(price))
    print("Variance (Manual):", m_variance(price))

    print("Manual Mean Time:", exec_time(m_mean, price))
    print("NumPy Mean Time:", exec_time(np.mean, price))

    wednesday_mean = df[df["Day"] == "Wednesday"]["Price"].mean()
    april_mean = df[df["Month"] == "Apr"]["Price"].mean()

    print("Wednesday Mean:", wednesday_mean)
    print("April Mean:", april_mean)

    loss_prob = len(df[df["Chg%"] < 0]) / len(df)
    print("Probability of Loss:", loss_prob)

    wed_profit = df[(df["Day"] == "Wednesday") & (df["Chg%"] > 0)]
    print("Profit on Wednesday Probability:", len(wed_profit) / len(df))

    plt.scatter(df["Day"], df["Chg%"])
    plt.xlabel("Day")
    plt.ylabel("Change %")
    plt.show()

main()
