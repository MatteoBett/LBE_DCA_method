#########################################################
#                        std Lib                        #
#########################################################
import os, sys, argparse, shutil
from typing import Callable, Dict, Tuple

#########################################################
#                      Dependencies                     #
#########################################################
import torch

#########################################################
#                      Own modules                      #
#########################################################
import dca.dataset.loader as loader


def gibbs_sampling(chains : torch.Tensor, 
                   params : Dict[str, torch.Tensor], 
                   nsweeps : int = 10,
                   beta : float = 1.0):
    
    nseq, nnuc, nval = chains.shape
    
    for _ in torch.arange(nsweeps):
        residue_idxs = torch.randperm(nnuc)
        for i in residue_idxs:
            couplings_residue = params["couplings"][i].view(nval, nnuc * nval)

            # Update the chains
            logit_residue = params["fields"][i].unsqueeze(0) + chains.reshape(nseq, nnuc * nval) @ couplings_residue.T # (N, q)
            make_proba = torch.softmax(beta*logit_residue, -1)

            sampled = torch.multinomial(make_proba, 1)
            chains[:, i, :] = torch.tensor(loader.encode_sequence(sampled), device=chains.device).to(logit_residue.dtype).squeeze(1)

    return chains

def get_single_point_freq(mat : torch.Tensor):
    nseqs, _, _ = mat.shape
    freq = mat.sum(dim=0) / nseqs
    return freq

@torch.jit.script
def get_two_point_freq(mat : torch.Tensor) -> torch.Tensor:
    nseq, nnuc, nval = mat.shape
    data_oh = mat.reshape(nseq, nnuc * nval)
    fij = (data_oh.T @ data_oh) / nseq
    return fij.reshape(nnuc, nval, nnuc, nval)

def calc_freq(mat : torch.Tensor):
    f_single = get_single_point_freq(mat=mat)
    f_double = get_two_point_freq(mat=mat)
    return f_single, f_double

@torch.jit.script
def compute_gradient_centred(fi: torch.Tensor,
                             fij: torch.Tensor,
                             pi: torch.Tensor,
                             pij: torch.Tensor,
                             ) -> Dict[str, torch.Tensor]:
    grad = {}
    
    C_data = fij - torch.einsum("ij,kl->ijkl", fi, fi)
    C_model = pij - torch.einsum("ij,kl->ijkl", pi, pi)
    grad["couplings"] = C_data - C_model
    grad["fields"] = fi - pi - torch.einsum("iajb,jb->ia", grad["couplings"], fi)
    
    return grad

def update_params(fi: torch.Tensor, 
                  fij: torch.Tensor,
                  pi: torch.Tensor,
                  pij: torch.Tensor,
                  params: Dict[str, torch.Tensor],
                  mask: torch.Tensor,
                  lr: float,
                  ) -> Dict[str, torch.Tensor]:
    """
    Updates the parameters of the model.
    """
    grad = compute_gradient_centred(fi=fi, fij=fij, pi=pi, pij=pij)

    with torch.no_grad():
        for key in params:
            if key == "gaps_bias" or key == "gaps_lr" or key == "all_params":
                continue
            params[key] += lr * grad[key]
        params["couplings"] *= mask # Remove autocorrelations
        params["fields"][:, 0] += params["gaps_bias"][:, 0]
        params["gaps_lr"] = params["gaps_lr"]*(1-0.0033)
        params["all_params"][:, 0] += params["gaps_bias"][:, 0] 

    return params

def _get_slope(x, y):
    """ 
    Get the slope of the curve obtained from the points resulting
    from the association of the x and y input vectors.
    """
    n = len(x)
    num = n * (x @ y) - y.sum() * x.sum()
    den = n * (x @ x) - torch.square(x.sum())
    return torch.abs(num / den)

def extract_Cij_from_freq(
                        fij: torch.Tensor,
                        pij: torch.Tensor,
                        fi: torch.Tensor,
                        pi: torch.Tensor,
                        mask: torch.Tensor | None = None,
                        ) -> Tuple[float, float]:
    """
    Extracts the lower triangular part of the covariance matrices of the data and chains starting from the frequencies.
    """
    L = fi.shape[0]
        
    # Compute the covariance matrices
    cov_data = fij - torch.einsum('ij,kl->ijkl', fi, fi)
    cov_chains = pij - torch.einsum('ij,kl->ijkl', pi, pi)
    
    # Only use a subset of couplings if a mask is provided
    if mask is not None:
        cov_data = torch.where(mask, cov_data, torch.tensor(0.0, device=cov_data.device, dtype=cov_data.dtype))
        cov_chains = torch.where(mask, cov_chains, torch.tensor(0.0, device=cov_chains.device, dtype=cov_chains.dtype))
    

    # Extract only the entries of half the matrix and out of the diagonal blocks
    idx_row, idx_col = torch.tril_indices(L, L, offset=-1)
    fij_extract = cov_data[idx_row, :, idx_col, :].reshape(-1)
    pij_extract = cov_chains[idx_row, :, idx_col, :].reshape(-1)
    return fij_extract, pij_extract


def two_points_correlation(fij : torch.Tensor, 
                           pij : torch.Tensor, 
                           fi : torch.Tensor, 
                           pi : torch.Tensor,
                           mask: torch.Tensor | None = None
                           ):
    """
    Computes the Pearson coefficient and the slope between the two-point frequencies of data and chains.
    """
    fij_extract, pij_extract = extract_Cij_from_freq(fij, pij, fi, pi, mask)
    stack = torch.stack([fij_extract, pij_extract])
    pearson = torch.corrcoef(stack)[0, 1].item()
    
    slope = _get_slope(fij_extract, pij_extract).item()
    
    return pearson, slope