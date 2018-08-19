import numpy as np


def compress_image(image, num_values):
    """Compress an image using SVD and keeping the top `num_values` singular values.

    Args:
        image: numpy array of shape (H, W)
        num_values: number of singular values to keep

    Returns:
        compressed_image: numpy array of shape (H, W) containing the compressed image
        compressed_size: size of the compressed image
    """
    compressed_image = None
    compressed_size = 0

    # YOUR CODE HERE
    # Steps:
    #     1. Get SVD of the image
    #     2. Only keep the top `num_values` singular values, and compute `compressed_image`
    #     3. Compute the compressed size
    
    # 1. Get SVD of 1 channel image
    #    The value of s has been sorted descending
    U, s, V = np.linalg.svd(image, full_matrices = True)
    
    # 2. Keep top num_values singular value and vector
    U_c = U[:, 0:num_values]
    S_c = np.diag(s[0:num_values])
    V_c = V[0:num_values, :]
    
    
    # 3. Reconstruct the compressed image
    compressed_image = np.dot(U_c, np.dot(S_c, V_c))
    
    H_c, W_c = compressed_image.shape
    
    compressed_size = U_c.size + V_c.size + num_values
    
    # END YOUR CODE

    assert compressed_image.shape == image.shape, \
           "Compressed image and original image don't have the same shape"

    assert compressed_size > 0, "Don't forget to compute compressed_size"

    return compressed_image, compressed_size
