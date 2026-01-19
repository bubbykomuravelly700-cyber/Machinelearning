import numpy as np
import matplotlib.pyplot as plt

def main():
    simi_matrix = np.random.rand(20, 20)

    plt.imshow(simi_matrix)
    plt.colorbar()
    plt.title("Similarity Heatmap (JC / SMC / Cosine)")
    plt.show()

main()
