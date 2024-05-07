import numpy as np
import sparse
import tensorly as tl
from tensorly.decomposition import parafac
from scipy.sparse.linalg import lsqr
from scipy.sparse import csr_matrix, vstack

def initialize_sparse_factors(tensor_shape, rank):
    """ Initialize factor matrices randomly for a sparse tensor """
    factors = []
    for mode in range(len(tensor_shape)):
        shape = (tensor_shape[mode], rank)
        factor = np.random.rand(*shape)
        factors.append(factor)
    return factors

def sparse_khatri_rao(matrices):
    """ Compute the Khatri-Rao product of matrices with sparse matrices in mind """
    result = csr_matrix(matrices[0])
    for mat in matrices[1:]:
        mat_csr = csr_matrix(mat)
        expanded_result = vstack([result] * mat_csr.shape[0])
        # Adjust the multiplication step to ensure consistency
        tiled_mat = csr_matrix(np.tile(mat_csr.toarray(), (result.shape[0], 1)))
        result = expanded_result.multiply(tiled_mat)
    return result

def unfold_coo(tensor, mode):
    """ Unfold a COO sparse tensor along a given mode """
    shape = tensor.shape
    other_modes = [i for i in range(len(shape)) if i != mode]
    
    # Calculate the new shape after unfolding
    unfolded_rows = shape[mode]
    unfolded_cols = np.prod([shape[i] for i in other_modes])
    
    # Create an empty sparse matrix to fill with values
    unfolded_matrix = csr_matrix((unfolded_rows, unfolded_cols))
    
    # Populate the matrix with non-zero values from the tensor
    row_indices = tensor.coords[mode]
    col_indices = np.zeros(len(tensor.data), dtype=int)
    
    # Calculate the flat index for the other modes
    multiplier = 1
    for i in reversed(other_modes):
        col_indices += tensor.coords[i] * multiplier
        multiplier *= shape[i]
    
    unfolded_matrix = csr_matrix((tensor.data, (row_indices, col_indices)), shape=(unfolded_rows, unfolded_cols))
    
    return unfolded_matrix

def update_sparse_factor(tensor, factors, mode):
    """ Update a factor matrix for sparse tensors using least squares """
    other_modes = [m for m in range(len(factors)) if m != mode]
    kr_product = sparse_khatri_rao([factors[m] for m in other_modes])
    
    unfolded_tensor = unfold_coo(tensor, mode)

    # Check compatibility
    print(f'kr_product shape: {kr_product.shape}')
    print(f'unfolded_tensor.T shape: {unfolded_tensor.T.shape}')

    # Solve the least squares problem using a sparse solver
    solution = lsqr(kr_product, unfolded_tensor.T)
    new_factor = solution[0].reshape((tensor.shape[mode], -1))
    return new_factor

def cp_als_coo(tensor, rank, max_iter=1000, tolerance=1e-5):
    """ CP decomposition via ALS for sparse tensors in COO format """
    factors = initialize_sparse_factors(tensor.shape, rank)
    for iteration in range(max_iter):
        for mode in range(len(factors)):
            factors[mode] = update_sparse_factor(tensor, factors, mode)
        # Optional: Check for convergence by calculating the norm of the reconstruction error
        if iteration % 10 == 0:
            error = 0  # Placeholder error calculation logic
            print(f'Iteration {iteration}, Reconstruction Error: {error}')
            if error < tolerance:
                break
    return factors