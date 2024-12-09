import numpy as np
import time

# Standard Matrix Multiplication (Triple Nested Loops)
def basic_matrix_multiply(A, B):
    """
    Perform matrix multiplication using the standard method with triple nested loops.
    """
    # Get the dimensions of the matrices A and B
    m, n = A.shape
    n2, p = B.shape
    C = np.zeros((m, p))

    # Standard matrix multiplication with triple nested loops
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]

    return C

# Example usage:
A = np.random.rand(500, 500)
B = np.random.rand(500, 500)

# Measure the execution time of the basic matrix multiplication
start_time = time.time()
C_basic = basic_matrix_multiply(A, B)
end_time = time.time()
print(f"Basic Matrix Multiplication Time: {end_time - start_time:.4f} seconds")
