# Optimized Matrix Multiplication Using Blocking (Tiling)
def blocked_matrix_multiply(A, B, block_size=50):
    """
    Perform matrix multiplication using the blocking technique for memory optimization.
    """
    # Get the dimensions of the matrices A and B
    m, n = A.shape
    n2, p = B.shape
    C = np.zeros((m, p))

    # Loop over blocks
    for ii in range(0, m, block_size):
        for jj in range(0, p, block_size):
            for kk in range(0, n, block_size):
                # Perform the block multiplication
                for i in range(ii, min(ii + block_size, m)):
                    for j in range(jj, min(jj + block_size, p)):
                        for k in range(kk, min(kk + block_size, n)):
                            C[i, j] += A[i, k] * B[k, j]

    return C

# Example usage:
A = np.random.rand(500, 500)
B = np.random.rand(500, 500)

# Measure the execution time of the optimized matrix multiplication with blocking
start_time = time.time()
C_blocked = blocked_matrix_multiply(A, B, block_size=50)
end_time = time.time()
print(f"Blocked Matrix Multiplication Time: {end_time - start_time:.4f} seconds")
