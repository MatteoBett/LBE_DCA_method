#########################################################
#                        std Lib                        #
#########################################################
from typing import Dict, List
import json
#########################################################
#                      Dependencies                     #
#########################################################
import torch
import numpy as np
import pandas as pd
#########################################################
#                      Own modules                      #
#########################################################


def chains_to_fasta(chains_path  : str,
                  chains : torch.Tensor,
                  alphabet : str,
                  headers : List[str] | None = None):
    """ Update the chains sampled during the DCA """

    chains=chains.argmax(dim=-1)
    assert chains.ndim == 2, "Chains must be a 2D Tensor to be saved"

    chains = chains.cpu().numpy()

    with open(chains_path, 'w') as chains_file:
        for index, chain in enumerate(chains):
            if headers is not None:
                chains_file.write(f"{headers[index]}\n")
            else:
                chains_file.write(f">chain_{index}\n")
            for elt in chain:
                chains_file.write(f"{alphabet[elt]}")
            chains_file.write('\n')
    

def save_params(params_file : str,
                params : Dict[str, torch.Tensor]):
    """
    Saves the parameters of the model in a file."
    """
    with open(params_file, 'w') as writer:
        json.dump({k : v.cpu().tolist() for k, v in params.items()}, writer)
    
def load_params(params_file : str, 
                device : str = 'cuda:0',
                dtype : torch.dtype = torch.float32):
    """
    Load the saved parameters of a previously trained model
    """
    with open(params_file) as trained_model:
        params = json.loads(trained_model.read())
    
    return {k : torch.tensor(v, device=device, dtype=dtype) for k,v in params.items()}