import numpy as np

def cosine_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def main():
    v1 = np.array([2, 1, 0, 3])
    v2 = np.array([1, 0, 2, 1])
    print("Similarity:", cosine_sim(v1, v2))

main()
