import numpy as np
import pandas as pd
from Bio import SeqIO


def import_data(filename):
    """Extracts sequences and counts from the fasta data file,
    and returns a pandas dataframe."""

    with open(filename, "r") as f:
        reads = list(SeqIO.parse(f, format="fasta"))

    seqs = [str(r.seq) for r in reads]
    counts = [int(r.id.split("-")[1]) for r in reads]

    df = pd.DataFrame({"seq": seqs, "counts": counts})
    df = df.set_index("seq")

    return df


def project_on_side(df, side):
    """Given a pandas dataframe with sequences and counts, it
    group sequences and counts according to only the right or left loop, and
    returns a dataframe with cumulative count."""

    assert (side == "left") or (side == "right"), "side must be left or right"

    beg, end = (None, 20) if side == "left" else (20, None)
    df = df.reset_index()
    df.seq = df.seq.apply(lambda s: s[beg:end])
    df = df.groupby("seq")["counts"].sum().sort_values(ascending=False)
    return pd.DataFrame(df)
