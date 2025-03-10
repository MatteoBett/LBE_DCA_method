import os, sys, re
from typing import Tuple

from RNA import RNA
from Bio import SeqIO

from dca.dataset.loader import DatasetDCA


""" Set the temperature for all analysis """
RNA.cvar.temperature = 25


def make_consensus(infile_path : str) -> Tuple[str, float]:
    msa = DatasetDCA(infile_path).msa
    return RNA.alifold(msa)
    
def make_mfe(seq : str) -> str:
    return RNA.fold(seq)

def ss_distance(seq_gen : str, refseq : str) -> float:
    return RNA.compare_structure(seq_gen, refseq)

def walk_seq(infile_path : str, genseqs_path : str, indel_mode : bool = False):
    consensus_ref, mfe_ref = make_consensus(infile_path)
    for record in SeqIO.parse(genseqs_path, 'fasta'):
        tmp_struct, tmp_mfe = make_mfe(str(record.seq))
        if len(record.description) < 5:
            continue
        yield record.seq, (mfe_ref, tmp_mfe), record.description


