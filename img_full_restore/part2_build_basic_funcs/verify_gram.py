import scipy.sparse
import numpy as np
import os

file_path = "part2_build_basic_funcs/gram_matrix.npz"

if not os.path.exists(file_path):
    print(f"Error: File {file_path} not found.")
    exit(1)

G = scipy.sparse.load_npz(file_path)

print(f"Shape: {G.shape}")
print(f"Non-zero elements: {G.nnz}")
print(f"Sparsity: {G.nnz / (G.shape[0]*G.shape[1]):.4f}")

# Check symmetry
diff = (G - G.T).nnz
print(f"Symmetry check (G - G.T).nnz: {diff}")
if diff == 0:
    print("Matrix is symmetric.")
else:
    print("Matrix is NOT symmetric.")

# Check positive diagonal
diag = G.diagonal()
min_diag = np.min(diag)
max_diag = np.max(diag)
print(f"Min diagonal element: {min_diag}")
print(f"Max diagonal element: {max_diag}")
if min_diag > 0:
    print("Diagonal elements are positive.")
else:
    print("Diagonal elements contain non-positive values.")
