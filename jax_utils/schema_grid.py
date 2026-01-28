from itertools import product


def get_positions(amount, dim):
    # Returns a list of tuples representing all N-dimensional coordinates
    # from 0 to amount-1
    return list(product(range(amount), repeat=dim))