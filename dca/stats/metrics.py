#########################################################
#                        std Lib                        #
#########################################################
from typing import Dict

#########################################################
#                      Dependencies                     #
#########################################################
import torch
from tqdm import tqdm

#########################################################
#                      Own modules                      #
#########################################################

def compute_log_likelihood(
    fi: torch.Tensor,
    fij: torch.Tensor,
    params: Dict[str, torch.Tensor],
    logZ: float,
) -> float:
    
    mean_energy_data = - torch.sum(fi * params["fields"]) - 0.5 * torch.sum(fij * params["couplings"])
    return - mean_energy_data - logZ


def compute_Dkl(fij: torch.Tensor,
                pij: torch.Tensor
                ) -> torch.Tensor:
    """
    Computes the Kullback-Leibler divergence matrix of all the possible couplings.
    """
    L = fij.shape[0]
    Dkl = fij * (torch.log(fij) - torch.log(pij)) + (1. - fij) * (torch.log(1. - fij) - torch.log(1. - pij))
    Dkl[torch.arange(L), :, torch.arange(L), :] = -float("inf")
    
    return Dkl


def activate_graph(
    mask: torch.Tensor,
    fij: torch.Tensor,
    pij: torch.Tensor,
    nactivate: int,
) -> torch.Tensor: 
    """
    Updates the interaction graph by activating a maximum of nactivate couplings, based on the
    Kullback-Leibler divergence between the two points frequency for the generated chains
    respect to the training data.
    """
    Dkl = compute_Dkl(fij=fij, pij=pij)
    Dkl_flat_sorted, _ = torch.sort(Dkl.flatten(), descending=True)

    Dkl_th = Dkl_flat_sorted[2 * nactivate]
    mask = torch.where(Dkl > Dkl_th, torch.tensor(1, device=Dkl.device, dtype=Dkl.dtype), mask)
    
    return mask

