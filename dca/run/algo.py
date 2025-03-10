#########################################################
#                        std Lib                        #
#########################################################
import os, sys, argparse, shutil
from typing import Callable, Dict, Tuple

#########################################################
#                      Dependencies                     #
#########################################################
import torch
from tqdm import tqdm
#########################################################
#                      Own modules                      #
#########################################################
import dca.dataset.loader as loader
import dca.run.algo as algo

""" Point frequency calculation """

@torch.jit.script
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

def calc_freq(mat : torch.Tensor) -> Tuple[torch.Tensor]:
    f_single = get_single_point_freq(mat=mat)
    f_double = get_two_point_freq(mat=mat)
    return f_single, f_double

""" Parameters Update """

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
        params["fields"][:, 0] += params["gaps_bias"][:, 0]
        if params["gaps_lr"] > 0.00001:
            params["gaps_lr"] = params["gaps_lr"]*0.95
        params["all_params"][:, 0] += params["gaps_bias"][:, 0] 

    return params


def compute_gap_gradient(target_dist : torch.Tensor,
                         dist_sample : torch.Tensor,
                         params : Dict[str, torch.Tensor],
                         device : str = 'cuda:0'
                         ) -> Dict[str, torch.Tensor]:
    """
    Computes the gradient of the bias applied to the gaps frequency and adjust it 
    toward a target distribution of gaps corresponding to a mean frequency of gaps in the sequence.
    """ 
    target_dist = target_dist.to(device=device)
    dist_sample = dist_sample.to(device=device)
    loss = target_dist - dist_sample

    new_bias = params["gaps_lr"] * loss # positive result
    params["gaps_bias"][:, 0] += new_bias[:, 0]
    return params


""" Sampling """

def gibbs_sampling(chains : torch.Tensor, 
                   params : Dict[str, torch.Tensor], 
                   gaps_target_dist : torch.Tensor | None = None,
                   bias_flag : bool | None = False,
                   nsweeps : int = 10,
                   beta : float = 1.0):
    
    nseq, nnuc, nval = chains.shape
    for _ in torch.arange(nsweeps):
        residue_idxs = torch.randperm(nnuc)
        alloc_bias = torch.zeros((nnuc, 1), device='cuda:0')

        for i in residue_idxs:
            couplings_residue = params["couplings"][i].view(nval, nnuc * nval)
            
            logit_residue = params["fields"][i].unsqueeze(0) + chains.reshape(nseq, nnuc * nval) @ couplings_residue.T
            make_proba = torch.softmax(beta*logit_residue, -1)
            
            sampled = torch.multinomial(make_proba, 1)
            chains[:, i, :] = loader.one_hot(sampled, device=chains.device, num_classes=nval).to(logit_residue.dtype).squeeze(1)

            alloc_bias[i] = make_proba[:, 0].mean()
        
        if bias_flag:
            params = compute_gap_gradient(
            target_dist=gaps_target_dist,
            dist_sample=alloc_bias,
            params=params
        )
        
    return chains, params


@torch.jit.script
def get_mean_seqid(
    a1: torch.Tensor,
    a2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the mean and the standard deviation of the mean sequence identity between two sets of one-hot encoded sequences.
    """
    a1 = a1.view(a1.shape[0], -1)
    a2 = a2.view(a2.shape[0], -1)
    overlaps = (a1 * a2).sum(1)
    mean_overlap = overlaps.mean()
    std_overlap = overlaps.std() / torch.sqrt(overlaps.shape[0])
    
    return mean_overlap, std_overlap


def compute_mixing_time(
    data: torch.Tensor,
    params: Dict[str, torch.Tensor],
    n_max_sweeps: int,
    beta: float,
) -> Dict[str, list]:
    """
    Computes the mixing time using the t and t/2 method. The sampling will halt when the mixing time is reached or
    the limit of `n_max_sweeps` sweeps is reached.
    """

    torch.manual_seed(0)
    
    L, _ = params["fields"].shape
    # Initialize chains at random
    sample_t = data
    # Copy sample_t to a new variable sample_t_half
    sample_t_half = sample_t.clone()

    # Initialize variables
    results = {
        "seqid_t": [],
        "std_seqid_t": [],
        "seqid_t_t_half": [],
        "std_seqid_t_t_half": [],
        "t_half": [],
    }

    # Loop through sweeps
    pbar = tqdm(
        total=n_max_sweeps,
        colour="red",
        dynamic_ncols=True,
        leave=False,
        ascii="-#",
    )
    pbar.set_description("Iterating until the mixing time is reached")
        
    for i in range(1, n_max_sweeps + 1):
        pbar.update(1)
        # Set the seed to i
        torch.manual_seed(i)
        # Perform a sweep on sample_t
        sample_t, params = algo.gibbs_sampling(chains=sample_t, params=params, nsweeps=1, beta=beta)

        if i % 2 == 0:
            # Set the seed to i/2
            torch.manual_seed(i // 2)
            # Perform a sweep on sample_t_half
            sample_t_half, params = algo.gibbs_sampling(chains=sample_t_half, params=params, nsweeps=1, beta=beta)

            # Calculate the average distance between sample_t and itself shuffled
            perm = torch.randperm(len(sample_t))
            seqid_t, std_seqid_t = get_mean_seqid(sample_t, sample_t[perm])
            seqid_t, std_seqid_t = seqid_t / L, std_seqid_t / L

            # Calculate the average distance between sample_t and sample_t_half
            seqid_t_t_half, std_seqid_t_t_half = get_mean_seqid(sample_t, sample_t_half)
            seqid_t_t_half, std_seqid_t_t_half = seqid_t_t_half / L, std_seqid_t_t_half / L

            # Store the results
            results["seqid_t"].append(seqid_t.item())
            results["std_seqid_t"].append(std_seqid_t.item())
            results["seqid_t_t_half"].append(seqid_t_t_half.item())
            results["std_seqid_t_t_half"].append(std_seqid_t_t_half.item())
            results["t_half"].append(i // 2)

            # Check if they have crossed
            if torch.abs(seqid_t - seqid_t_t_half) / torch.sqrt(std_seqid_t**2 + std_seqid_t_t_half**2) < 0.1:
                break

        if i == n_max_sweeps:
            print(f"Mixing time not reached within {n_max_sweeps // 2} sweeps.")
            
    pbar.close()

    return results