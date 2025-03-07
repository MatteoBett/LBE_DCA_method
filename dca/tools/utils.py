#########################################################
#                        std Lib                        #
#########################################################
from typing import Dict
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


def update_chains(chains_path  : str,
                  chains : torch.Tensor,
                  alphabet : str):
    """ Update the chains sampled during the DCA """

    chains=chains.argmax(dim=-1)
    assert chains.ndim == 2, "Chains must be a 2D Tensor to be saved"

    chains = chains.cpu().numpy()

    with open(chains_path, 'w') as chains_file:
        for index, chain in enumerate(chains):
            chains_file.write(f">chain_{index}\n")
            for elt in chain:
                chains_file.write(f"{alphabet[elt]}")
            chains_file.write('\n')
    

def save_params(params_file : str,
                params : Dict[str, torch.Tensor],
                mask: torch.Tensor,
                alphabet : str):
    """
    Saves the parameters of the model in a file."
    """
    mask = mask.cpu().numpy()
    params = {k : v.cpu().numpy() for k, v in params.items()}
    
    
    L, q, *_ = mask.shape
    idx0 = np.arange(L * q).reshape(L * q) // q
    idx1 = np.arange(L * q).reshape(L * q) % q
    idx1_aa = np.vectorize(lambda n, tokens : tokens[n], excluded=["tokens"])(idx1, alphabet).astype(str)
    df_h = pd.DataFrame(
        {
            "param" : np.full(L * q, "h"),
            "index" : idx0,
            "nuc" : idx1_aa,
            "val" : params["bias"].flatten(),
        }
    )
    
    maskt = mask.transpose(0, 2, 1, 3) # Transpose mask and coupling matrix from (L, q, L, q) to (L, L, q, q)
    Jt = params["coupling_matrix"].transpose(0, 2, 1, 3)
    idx0, idx1, idx2, idx3 = maskt.nonzero()
    idx2_aa = np.vectorize(lambda n, tokens : tokens[n], excluded=["tokens"])(idx2, alphabet).astype(str)
    idx3_aa = np.vectorize(lambda n, tokens : tokens[n], excluded=["tokens"])(idx3, alphabet).astype(str)
    J_val = Jt[idx0, idx1, idx2, idx3]
    df_J = pd.DataFrame(
        {
            "param" : np.full(len(J_val), "J").tolist(),
            "idx0" : idx0,
            "idx1" : idx1,
            "idx2" : idx2_aa,
            "idx3" : idx3_aa,
            "val" : J_val,
        }
    )

    df_b = pd.DataFrame(
        {
            "param" : np.full(L, "b"),
            "index" : np.arange(L, dtype=np.int32),
            "val" : params["all_params"].flatten(),
        }
    )
    
    df_J.to_csv(params_file, sep=" ", header=False, index=False)
    df_h.to_csv(params_file, sep=" ", header=False, index=False, mode="a")
    df_b.to_csv(params_file, sep=" ", header=False, index=False, mode='a')
    