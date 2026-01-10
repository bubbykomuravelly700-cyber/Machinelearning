import random
from collections import Counter
#1
def vowelsconsonents(text):
    vowels = "aeiouAEIOU"
    v = 0
    c = 0
    for char in text:
        if char.isalpha():
            if char in vowels:
                v += 1
            else:
                c += 1

    return v,c

#2
def multmatrices(matrix_a,matrix_b):
    rows_a = len(matrix_a)
    cols_a = len(matrix_a[0])
    rows_b = len(matrix_b)
    cols_b = len(matrix_b[0])
    if cols_a != rows_b:
       return None
    result = [[0 for _ in range(cols_b)] for _ in range(rows_a)]

    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]
    return result

#3
def commonelements(list1,list2):
    return len(set(list1) & set(list2))

#4
def transpose_matrix(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    return [[matrix[j][i] for j in range(rows)] for i in range(cols)]

#5
def statistics_random_numbers():
    numbers = [random.randint(100, 150) for _ in range(100)]
    mean_value = sum(numbers) / len(numbers)
    sorted_numbers = sorted(numbers)
    n = len(sorted_numbers)
    if n % 2 == 0:
        median_value = (sorted_numbers[n // 2 - 1] + sorted_numbers[n // 2]) / 2
    else:
        median_value = sorted_numbers[n // 2]
    frequency = Counter(numbers)
    mode_value = frequency.most_common(1)[0][0]
    return  mean_value, median_value, mode_value

if __name__ == "__main__":
#1
    text = input("enter a string : ")
    v,c = vowelsconsonents(text)
    print("number of vowels :", v)
    print("number of consonants :", c)

#2
    matrix_a = [[5,8],[5,6]]
    matrix_b = [[4,6],[6,8]]
    product = multmatrices(matrix_a,matrix_b)

    print("matrix multiplication : ")
    for row in product:
        print(row)

#3
    list1 = [5,6,3,4,7,8,3]
    list2 = [4,2,6,8,4,7,5]
    print("number of common elements :", commonelements(list1,list2))

#4
    matrix = [[3,4,5],[5,7,8]]
    print("transpose of matrix:")
    for row in transpose_matrix(matrix):
        print(row)

#5
    mean_val,median_val,mode_val = statistics_random_numbers()
    print("mean:",mean_val)
    print("median:",median_val)
    print("mode:",mode_val)
