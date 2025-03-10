#########################################################
#                        std Lib                        #
#########################################################
from typing import Dict, Tuple

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


def _compute_energy_sequence(
    x: torch.Tensor,
    params: Dict[str, torch.Tensor],
) -> torch.Tensor:
    L, q = params["fields"].shape
    x_oh = x.ravel()
    
    bias_oh = params["fields"].ravel()
    couplings_oh = params["couplings"].view(L * q, L * q)

    fields = - x_oh @ bias_oh
    couplings = - 0.5 * x_oh @ (couplings_oh @ x_oh)

    energy = fields + couplings
    return energy


def compute_energy(
    X: torch.Tensor,
    params: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Compute the DCA energy of the sequences in X.
    """
    
    if X.dim() != 3:
        raise ValueError("Input tensor X must be 3-dimensional of size (_, L, q)")

    return torch.vmap(_compute_energy_sequence, in_dims=(0, None))(X, params)
