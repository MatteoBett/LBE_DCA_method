#########################################################
#                        std Lib                        #
#########################################################
import os
from typing import Any, List, Literal
from pathlib import Path

#########################################################
#                      Dependencies                     #
#########################################################
import Bio.SeqRecord
import torch
import Bio
import numpy as np
from torch.utils.data import Dataset
import torch
from Bio import SeqIO

#########################################################
#                      Own modules                      #
#########################################################



class DatasetDCA(Dataset):
    """Dataset class for handling multi-sequence alignments data."""
    def __init__(
        self,
        path_data: str | Path,
        params_file : str | Path | None = None,
        chains_file : str | Path | None = None,
        alphabet: str = '-AUCG',
        device : str = "cuda"
    ):
        """
        Initialize the dataset.
        """
        self.path_data = Path(path_data)
        self.device = device
        self.chains_file = Path(chains_file) if chains_file is not None else None
        self.params_file = Path(params_file) if params_file is not None else None

        self.alphabet = alphabet
        self.dtype = torch.float32

        self.msa = []
        self.mat = []
        for record in SeqIO.parse(self.path_data, 'fasta'):
            self.msa.append(str(record.seq))
            self.mat.append(encode_sequence(record.seq))
        
        self.mat = one_hot(mat=self.mat, device=self.device, num_classes=len(alphabet))
        self.nseq, self.nnuc, self.nval = self.mat.shape

        """
            nseq = number of sequences in the MSA
            nnuc = sequences lengths in the MSA (i.e., number of nucleotides)
            nval = number of different nucleotides elements        
        """

        self.params = {
            "fields" : torch.zeros((self.nnuc, self.nval), device=device, dtype=self.dtype),
            "couplings": torch.zeros((self.nnuc, self.nval, self.nnuc, self.nval), device=device, dtype=self.dtype),
            "gaps_bias": torch.zeros((self.nnuc, 1), device=device, dtype=self.dtype),
            "gaps_lr" : torch.tensor([0.001], device=device, dtype=self.dtype),
            "all_params" : torch.zeros((self.nnuc, 1), device=device, dtype=self.dtype)
        }

        self.mask = torch.ones(size=(self.nnuc, self.nval, self.nnuc, self.nval), dtype=torch.bool, device=device)
        self.mask[torch.arange(self.nnuc), :, torch.arange(self.nnuc), :] = 0

    def __len__(self):
        return len(self.msa)
        
    def produce_chains(self, nchains : int) -> torch.Tensor:
        """
        Samples sequences according to the single point frequency of the
        currently loaded dataset.
        """
        chains = torch.multinomial(torch.exp(self.params["fields"]), num_samples=nchains, replacement=True).to(device=self.device).T
        return one_hot(chains, device=self.device, dtype=torch.float32)

    def get_indels_info(self):
        new = np.matrix([list(elt) for elt in self.msa]).T.tolist()

        count_gaps = {index : col.count('-') + col.count('*') for index, col in enumerate(new)}
        mean = np.mean(list(count_gaps.values()))
        
        return count_gaps, mean
    
def encode_sequence(seq : Bio.SeqRecord.Seq | torch.Tensor):
    if isinstance(seq, Bio.SeqRecord.Seq):
        dico = {'-':0,'A':1,'U':2,'C':3,'G':4}
        return [dico[nuc] for nuc in seq]    
    


def family_stream(family_dir : str):
    """ Yield the output of load_msa function for each family directory """
    for family_file in os.listdir(family_dir):
        yield family_file, os.path.join(family_dir, family_file, f"{family_file}.fasta")




@torch.jit.script
def _one_hot(x: torch.Tensor, num_classes: int = -1, dtype: torch.dtype = torch.float32):
   
    if x.dim() != 2:
        raise ValueError("Input tensor x must be 2D")
    
    if num_classes < 0:
        num_classes = x.max() + 1
    res = torch.zeros(x.shape[0], x.shape[1], num_classes, device=x.device, dtype=dtype)
    tmp = torch.meshgrid(
        torch.arange(x.shape[0], device=x.device),
        torch.arange(x.shape[1], device=x.device),
        indexing="ij",
    )
    index = (tmp[0], tmp[1], x)
    values = torch.ones(x.shape[0], x.shape[1], device=x.device, dtype=dtype)
    res.index_put_(index, values)
    
    return res


def one_hot(mat: List[List[int]] | torch.Tensor, device : str,  num_classes: int = -1, dtype: torch.dtype = torch.float32):
    """
    A fast one-hot encoding function faster than the PyTorch one working with torch.int32 and returning a float Tensor.
    Works only for 2D tensors.
    """
    if isinstance(mat, List):
        mat = torch.tensor(mat, device=device, dtype=torch.int32)
    return _one_hot(mat, num_classes, dtype)