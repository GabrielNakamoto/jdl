def matmul(a, b):
    assert len(a[0]) == len(b)
    m, n = len(a), len(a[0])
    p = len(b[0])

    c = [[0] * p] * m
    for i in range(m):
        for j in range(p):
            for k in range(n):
                c[i][j]+=a[i][k]*b[k][j]
    return c


a = [
    [1, 3, 4],
    [-10, 5, 6]
]

b = [
    [0, -20],
    [25, 10],
    [-3.4, 8.5]
]


# (2, 3) @ (3, 2) = (2, 2)
c = matmul(a, b)
print(c)
