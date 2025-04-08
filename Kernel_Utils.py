import numpy as np


def generate_linear_decreasing_matrix(n, m, max_val, min_val):
    """
    Generates a matrix with linearly decreasing values from the center to the edges.

    The matrix is of size n x m. Values decrease from the center (max_val) to the edges (min_val)
    based on the maximum distance from the center along rows and columns.

    Args:
        n (int): Number of rows.
        m (int): Number of columns.
        max_val (float): Maximum value at the center.
        min_val (float): Minimum value at the edges.

    Returns:
        numpy.ndarray: A matrix of shape (n, m) with linearly decreasing values.
    """
    # Create an n x m matrix filled with zeros
    matrix = np.zeros((n, m))

    # Calculate the center indices of the matrix
    center_row = n // 2
    center_col = m // 2

    # Calculate the amplitude of decrease from max_val to min_val
    amplitude = max_val - min_val

    # Determine the decrease factor per row and per column
    row_factor = amplitude / max(center_row, n - center_row - 1)
    col_factor = amplitude / max(center_col, m - center_col - 1)

    # Assign values to the matrix based on the distance from the center
    for i in range(n):
        for j in range(m):
            row_distance = abs(i - center_row)
            col_distance = abs(j - center_col)
            # Use the maximum of row and column distance
            distance = max(row_distance, col_distance)
            # Compute the decrease based on the smaller factor to maintain uniform decrease
            decrease = distance * min(row_factor, col_factor)
            matrix[i, j] = max_val - decrease

    return matrix


def apply_single_kernel_in_tile_i_j(W, H, resource, i, j, K):
    """
    Applies a kernel to a specific tile of the resource matrix, centered at (i, j).

    This function performs a convolution-like operation using the kernel K on the resource
    matrix at position (i, j). It handles boundaries to ensure indices remain valid.

    Args:
        W (int): Total number of rows in the resource matrix.
        H (int): Total number of columns in the resource matrix.
        resource (numpy.ndarray): The matrix on which the kernel is applied.
        i (int): Row index of the central tile.
        j (int): Column index of the central tile.
        K (list of lists or numpy.ndarray): The kernel matrix to be applied.

    Returns:
        float: The computed value after applying the kernel to the tile.
    """
    val = 0.0
    # a and b represent the half-dimensions of the kernel
    a = len(K) // 2
    b = len(K[0]) // 2

    # Process the upper part of the kernel (including the central row)
    for ii in range(a + 1):
        # Ensure the row index is within bounds (above the central position)
        if i - (a - ii) >= 0:
            # Process the left side of the kernel
            for jj in range(b + 1):
                if j - (b - jj) >= 0:
                    val += (K[ii][jj] * resource[i - (a - ii)][j - (b - jj)])
            # Process the right side of the kernel
            for jj in range(b + 1, len(K)):
                if j + (jj - b) < H:
                    val += (K[ii][jj] * resource[i - (a - ii)][j + (jj - b)])

    # Process the lower part of the kernel
    for ii in range(a + 1, len(K)):
        if i + (ii - a) < W:
            # Left side for lower part
            for jj in range(b + 1):
                if j - (b - jj) >= 0:
                    val += (K[ii][jj] * resource[i + (ii - a)][j - (b - jj)])
            # Right side for lower part
            for jj in range(b + 1, len(K[0])):
                if j + (jj - b) < H:
                    val += (K[ii][jj] * resource[i + (ii - a)][j + (jj - b)])
    return val


def apply_kernel_with_pop_weight(resource, K, population):
    """
    Applies a kernel to each element of the resource matrix and weights the result by the corresponding population.

    For each cell in the resource matrix, this function applies the kernel using 'apply_single_kernel_in_tile_i_j'
    and then multiplies the result by the population value at that cell.

    Args:
        resource (numpy.ndarray): The resource matrix to process.
        K (list of lists or numpy.ndarray): The kernel to apply.
        population (numpy.ndarray): The population matrix (same shape as resource).

    Returns:
        numpy.ndarray: A matrix of the same shape as resource with weighted kernel results.
    """
    W = int(len(resource))
    H = int(len(resource[0]))
    results_matrix = np.zeros((W, H))
    for i in range(W):
        for j in range(H):
            results_matrix[i][j] = apply_single_kernel_in_tile_i_j(W, H, resource, i, j, K) * population[i][j]
    return results_matrix


def apply_kernel_no_weight(resource, K):
    """
    Applies a kernel to each element of the resource matrix without population weighting.

    The function processes the resource matrix by applying the kernel on each cell using
    'apply_single_kernel_in_tile_i_j'.

    Args:
        resource (numpy.ndarray): The resource matrix to process.
        K (list of lists or numpy.ndarray): The kernel to apply.

    Returns:
        numpy.ndarray: A matrix with the same dimensions as resource containing the kernel results.
    """
    W = int(len(resource))
    H = int(len(resource[0]))
    results_matrix = np.zeros((W, H))
    for i in range(W):
        for j in range(H):
            results_matrix[i][j] = apply_single_kernel_in_tile_i_j(W, H, resource, i, j, K)
    return results_matrix


def apply_kernel_normalized_pop_weight(resource, K, population):
    """
    Applies a kernel to each element of the resource matrix and weights the result by a normalized population value.

    Each population value is normalized to the range [0, 1] using the minimum and maximum population values
    in the entire matrix. The kernel is applied and the resulting value is multiplied by the normalized population.

    Args:
        resource (numpy.ndarray): The resource matrix to process.
        K (list of lists or numpy.ndarray): The kernel to apply.
        population (numpy.ndarray): The population matrix (same shape as resource).

    Returns:
        numpy.ndarray: A matrix with weighted kernel results based on normalized population values.
    """
    W = int(len(resource))
    H = int(len(resource[0]))
    pop_max = np.max(population)
    pop_min = np.min(population)
    results_matrix = np.zeros((W, H))
    for i in range(W):
        for j in range(H):
            temp = apply_single_kernel_in_tile_i_j(W, H, resource, i, j, K)
            pop_norm = normalize(population[i][j], pop_min, pop_max, 0.0, 1.0)
            results_matrix[i][j] = temp * pop_norm
    return results_matrix


def create_kernels_given_size(size):
    """
    Generates and prints a kernel matrix of a given size.

    The kernel is generated with values linearly decreasing from the center (1) to the edges (0.1).
    The matrix is printed in a formatted style.

    Args:
        size (int): The number of rows and columns of the kernel matrix.
    """
    min_val = 0.1
    max_val = 1
    kernel = generate_linear_decreasing_matrix(size, size, max_val, min_val)

    print(str(size) + ": [")
    for row in kernel:
        print("[", end="")
        for c in row:
            print(str(c) + ", ", end="")
        print(" ],")
    print("]")


def normalize(value, min_value, max_value, new_min=0.0, new_max=1.0):
    """
    Normalizes a value from the range [min_value, max_value] to the range [new_min, new_max].

    Args:
        value (float): The original value to normalize.
        min_value (float): The minimum value of the original range.
        max_value (float): The maximum value of the original range.
        new_min (float, optional): The minimum value for the normalized range (default 0.0).
        new_max (float, optional): The maximum value for the normalized range (default 1.0).

    Returns:
        float: The value normalized to the new range.
    """
    return ((value - min_value) / (max_value - min_value)) * (new_max - new_min) + new_min


if __name__ == '__main__':
    # Generate and print a kernel of size 121x121
    create_kernels_given_size(121)
