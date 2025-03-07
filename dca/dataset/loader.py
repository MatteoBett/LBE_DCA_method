#########################################################
#                        std Lib                        #
#########################################################
import os, sys, argparse, shutil

#########################################################
#                      Dependencies                     #
#########################################################
import Bio.SeqRecord
import torch
import Bio

#########################################################
#                      Own modules                      #
#########################################################

from typing import Any, List, Literal
from pathlib import Path
import os, re
from collections import Counter

from torch.utils.data import Dataset
import torch
from Bio import SeqIO
import numpy as np
import matplotlib.pyplot as plt


class DatasetDCA(Dataset):
    """Dataset class for handling multi-sequence alignments data."""
    def __init__(
        self,
        path_data: str | Path,
        chains_file : str | Path,
        params_file : str | Path,
        alphabet: str = '-AUCG',
        device : str = "cuda"
    ):
        """Initialize the dataset.

        Args:
            path_data (str | Path): Path to multi sequence alignment in fasta format.
        """
        self.path_data = Path(path_data)
        self.device = device
        self.chains_file = Path(chains_file)
        self.params_file = Path(params_file)
        self.alphabet = alphabet

        self.msa = []
        self.mat = []
        for record in SeqIO.parse(self.path_data, 'fasta'):
            self.msa.append(str(record.seq))
            self.mat.append(encode_sequence(record.seq))
        
        self.mat = torch.tensor(self.mat, device=device, dtype=torch.float32)
        self.nseq, self.nnuc, self.nval = self.mat.shape
        """
            nseq = number of sequences in the MSA
            nnuc = sequences lengths in the MSA (i.e., number of nucleotides)
            nval = number of different nucleotides elements        
        """

        self.params = {
            "fields" : torch.zeros((self.nnuc, self.nval), device=device, dtype=torch.float32),
            "couplings": torch.zeros((self.nnuc, self.nval, self.nnuc, self.nval), device=device, dtype=torch.float32),
            "gaps_bias": torch.zeros((self.nnuc, 1), device=device, dtype=torch.float32),
            "gaps_lr" : torch.tensor([0.001], device=device, dtype=torch.float32),
            "all_params" : torch.zeros((self.nnuc, 1), device=device, dtype=torch.float32)
        }

        self.mask = torch.zeros(size=(self.nnuc, self.nval, self.nnuc, self.nval), device=device, dtype=torch.bool)

    def __len__(self):
        return len(self.msa)
    
    def produce_chains(self, nchains : int) -> torch.Tensor:
        """
        Samples sequences according to the single point frequency of the
        currently loaded dataset.
        """
        chains = torch.multinomial(torch.exp(self.params["fields"]), num_samples=nchains, replacement=True).to(device=self.device).T
        return torch.tensor([encode_sequence(seq) for seq in chains], device=self.device, dtype=torch.float32)
    
def encode_sequence(seq : Bio.SeqRecord.Seq | torch.Tensor):
    if isinstance(seq, Bio.SeqRecord.Seq):
        dico = {
            '-':0,
            'A':1,
            'U':2,
            'C':3,
            'G':4
            }
        
        new = []
        for nuc in seq:
            oh = [0]*5
            oh[dico[nuc]] = 1
            new.append(oh)

    if isinstance(seq, torch.Tensor):
        new = []
        for nuc in seq:
            oh = [0]*5
            oh[nuc] = 1
            new.append(oh)        
    return new


def family_stream(family_dir : str):
    """ Yield the output of load_msa function for each family directory """
    for family_file in os.listdir(family_dir):
        yield family_file, os.path.join(family_dir, family_file, f"{family_file}.fasta")