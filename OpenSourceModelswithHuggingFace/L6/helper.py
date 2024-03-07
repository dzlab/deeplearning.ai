def get_idx(dataset,n):
    i = 0
    n = 1 # choose the n-th example
    example = None
    for example in dataset:
        if i == n:
            break
        i += 1
    return example