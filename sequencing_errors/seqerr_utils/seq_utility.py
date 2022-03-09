import numpy as np
import scipy.spatial.distance as spd

seq_dtype = np.uint16

nt_to_num = {"A": 0, "C": 1, "G": 2, "T": 3}
num_to_nt = {n: nt for nt, n in nt_to_num.items()}


def nt_to_array(nt_str):
    """Converts a string of nucleotides into a numpy array
    of integers"""
    num_seq = [nt_to_num[x] for x in nt_str]
    return np.array(num_seq, dtype=seq_dtype)


def nt_to_matrix(nt_list):
    """Converts a list of nucleotide sequences into a numpy matrix"""
    N = len(nt_list)
    L = len(nt_list[0])
    matrix = np.zeros((N, L), dtype=seq_dtype)
    for n, seq in enumerate(nt_list):
        matrix[n] = nt_to_array(seq)
    return matrix


def matrix_to_nt(seq_matrix):
    """converts back a matrix of integers into a list of nucleotide
    sequences."""
    seq_list = []
    for line in seq_matrix:
        seq_list.append("".join([num_to_nt[x] for x in line]))
    return seq_list
