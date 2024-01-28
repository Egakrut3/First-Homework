import numpy as np


def prod_non_zero_diag(x):
    """Compute product of nonzero elements from matrix diagonal.

    input:
    x -- 2-d numpy array
    output:
    product -- integer number


    Vectorized implementation.
    """

    return np.prod(np.diag(x), where = (np.diag(x) != 0))


def are_multisets_equal(x, y):
    """Return True if both vectors create equal multisets.

    input:
    x, y -- 1-d numpy arrays
    output:
    True if multisets are equal, False otherwise -- boolean

    Vectorized implementation.
    """
    
    return np.array_equal(np.sort(x, kind = "heapsort"), np.sort(y, kind = "heapsort"))


def max_after_zero(x):
    """Find max element after zero in array.

    input:
    x -- 1-d numpy array
    output:
    maximum element after zero -- integer number

    Vectorized implementation.
    """

    return np.max(x, initial = -100000, where = np.concatenate(([False], x[:-1] == 0)))


def convert_image(img, coefs):
    """Sum up image channels with weights from coefs array

    input:
    img -- 3-d numpy array (H x W x 3)
    coefs -- 1-d numpy array (length 3)
    output:
    img -- 2-d numpy array

    Vectorized implementation.
    """

    pass


def run_length_encoding(x):
    """Make run-length encoding.

    input:
    x -- 1-d numpy array
    output:
    elements, counters -- integer iterables

    Vectorized implementation.
    """

    index_arr = np.nonzero(np.concatenate(([True], x[1:] != x[:-1])))[0]
    return (x[index_arr], np.diff(np.concatenate((index_arr, [len(x)]))))


def pairwise_distance(x, y):
    """Return pairwise object distance.

    input:
    x, y -- 2d numpy arrays
    output:
    distance array -- 2d numpy array

    Vctorized implementation.
    """
    
    x_square = np.dot(np.atleast_2d(np.sum(np.square(x), axis = 1)).transpose(), np.atleast_2d(np.ones(len(y))))
    y_square = np.dot(np.atleast_2d(np.sum(np.square(y), axis = 1)).transpose(), np.atleast_2d(np.ones(len(x)))).transpose()
    return np.sqrt(x_square - 2 * np.dot(x, y) + y_square)
