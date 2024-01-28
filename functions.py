def prod_non_zero_diag(x):
    """Compute product of nonzero elements from matrix diagonal.

    input:
    x -- 2-d numpy array
    output:
    product -- integer number


    Not vectorized implementation.
    """
    
    ans = 1
    for i in range(min(len(x), len(x[0]))):
        if x[i][i] != 0:
            ans *= x[i][i]
    return ans


def are_multisets_equal(x, y):
    """Return True if both vectors create equal multisets.

    input:
    x, y -- 1-d numpy arrays
    output:
    True if multisets are equal, False otherwise -- boolean

    Not vectorized implementation.
    """
    
    return sorted(x) == sorted(y)


def max_after_zero(x):
    """Find max element after zero in array.

    input:
    x -- 1-d numpy array
    output:
    maximum element after zero -- integer number

    Not vectorized implementation.
    """
    
    ans = None
    for i in range(len(x) - 1):
        if x[i] == 0:
            if ans == None:
                ans = x[i + 1]
            else:
                ans = max(ans, x[i + 1])
    return ans


def convert_image(img, coefs):
    """Sum up image channels with weights from coefs array

    input:
    img -- 3-d numpy array (H x W x 3)
    coefs -- 1-d numpy array (length 3)
    output:
    img -- 2-d numpy array

    Not vectorized implementation.
    """

    pass


def run_length_encoding(x):
    """Make run-length encoding.

    input:
    x -- 1-d numpy array
    output:
    elements, counters -- integer iterables

    Not vectorized implementation.
    """

    value_arr = []
    count_arr = []
    cur_cnt = 1
    for i in range(len(x) - 1):
        if x[i] != x[i + 1]:
            value_arr.append(x[i])
            count_arr.append(cur_cnt)
            cur_cnt = 1
        else:
            cur_cnt += 1
    value_arr.append(x[-1])
    count_arr.append(cur_cnt)
    return (value_arr, count_arr)


def pairwise_distance(x, y):
    """Return pairwise object distance.

    input:
    x, y -- 2d numpy arrays
    output:
    distance array -- 2d numpy array

    Not vectorized implementation.
    """

    pass
