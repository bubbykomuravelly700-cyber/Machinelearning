import numpy as np
import math
 
def dot_prod(A, B):
    result = 0
    for i in range(len(A)):
        result = result + A[i] * B[i]
    return result

def euclidean_norm(A):
    total = 0
    for i in range(len(A)):
        total = total + A[i] * A[i]
    return total ** 0.5

A = [1,2,3]
B = [4,5,6]

print(dot_prod(A, B))
print(np.dot(A, B))

print(euclidean_norm(A))
print(np.linalg.norm(A))
