import numpy as np
import scipy.stats as sps

from .seq_utility import nt_to_matrix


def greedy_seqremoval(seqs, counts, eps):
    """
    Args:
    - seqs: a list of nucleotide sequences
    - counts: their number of counts
    - eps: estimated single-site reading error probability

    Returns:
    - a list of probabilities for each sequences, that errors could cause
        a number of observations equal or bigger than the number of counts.
    """
    # - logarithm of the same quantity, sometimes more precise.

    B = 4  # size of the alphabet, 4 nucleotides

    # sequences numerical matrix
    S = nt_to_matrix(seqs)
    # dictionary tuple(sequence) : sequence count
    cdict = {tuple(s): c for s, c in zip(S, counts)}
    N, L = S.shape

    # probability that errors could cause an equal or bigger number of observations
    P_err = np.zeros_like(counts, dtype=float)
    # same quantity but logarithm
    LogP_err = np.zeros_like(counts, dtype=float)

    # probability of making a particular mistake
    p = (eps / 3.0) * (1.0 - eps) ** (L - 1)
    # probability of reading a sequence correctly
    p_correct = (1.0 - eps) ** L

    for nseq, s in enumerate(S):
        # count n. neighbors
        nn = 0
        for l in range(L):
            for b in range(B - 1):
                # all single-site mutations
                s[l] = (s[l] + 1) % B
                key = tuple(s)
                if key in cdict:
                    # count counts of these neighbouring sequences
                    nn += cdict[key]
            # reset to original value
            s[l] = (s[l] + 1) % B

        if nn > 0:
            s_counts = counts[nseq]
            # correct nn with the error probability
            nn_tilde = int(nn / p_correct)
            # probability that errors could cause an equal or bigger number of observations
            P_err[nseq] = sps.binom.sf(s_counts - 1, nn_tilde, p)
            # LogP_err[nseq] = sps.binom.logsf(s_counts-1, nn_tilde, p)

    # return P_err, LogP_err
    return P_err
