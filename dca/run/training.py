#########################################################
#                        std Lib                        #
#########################################################
import time
from typing import Dict

#########################################################
#                      Dependencies                     #
#########################################################
import torch
from tqdm import tqdm

#########################################################
#                      Own modules                      #
#########################################################
import dca.dataset.loader as loader
import dca.stats.metrics as metrics
import dca.run.algo as algo
import dca.tools.utils as utils

def train_graph(chains : torch.Tensor,
                dataset : loader.DatasetDCA,
                f_single : torch.Tensor,
                f_double : torch.Tensor,
                target_pearson : float,
                log_weights : torch.Tensor,
                progress_bar : bool = True,
                max_epochs : int = 50
                ):
    
    device = f_single.device
    dtype = f_single.dtype
    L, q = f_single.shape
    time_start = time.time() 

    logZ = (torch.logsumexp(log_weights, dim=0) - torch.log(torch.tensor(len(chains), device=device, dtype=dtype))).item()
    log_likelihood = metrics.compute_log_likelihood(fi=f_single, fij=f_double, params=dataset.params, logZ=logZ) 

    p_single, p_double = algo.calc_freq(chains)
    pearson, slope = metrics.two_points_correlation(fij=f_double,pij=p_double,fi=f_single, pi=p_single)
    
    epochs = 0   

    if progress_bar: 
        pbar = tqdm(
            initial=max(0, float(pearson)),
            total=target_pearson,
            colour="red",
            dynamic_ncols=True,
            leave=False,
            ascii="-#",
            bar_format="{desc} {percentage:.2f}%[{bar}] Pearson: {n:.3f}/{total_fmt} [{elapsed}]"
        )
        pbar.set_description(f"Epochs: {epochs:3f} - Gap avg freq: {p_single[:,0].mean():.3f} - train LL: {log_likelihood:.2f}")

    while epochs < max_epochs:
        dataset.params = algo.update_params(
            fi=f_single,
            fij=f_double,
            pi=p_single,
            pij=p_double,
            params=dataset.params,
            mask=dataset.mask,
            lr=0.05,
        )

        # Update the Markov chains
        chains = algo.gibbs_sampling(chains=chains, 
                                        params=dataset.params)

        epochs += 1
        p_single, p_double = algo.calc_freq(mat=chains)            

        pearson, slope = metrics.two_points_correlation(fij=f_double, pij=p_double, fi=f_single, pi=p_single)
        logZ = (torch.logsumexp(log_weights, dim=0) - torch.log(torch.tensor(len(chains), device=device, dtype=dtype))).item()
        log_likelihood = metrics.compute_log_likelihood(fi=f_single, fij=f_double, params=dataset.params, logZ=logZ)

        if progress_bar:
            pbar.n = min(max(0, float(pearson)), target_pearson)
            pbar.set_description(f"Epochs: {epochs} - Gap avg freq: {p_single[:,0].mean():.3f} - LL: {log_likelihood:.2f}")

        if epochs % 5 == 0:
            utils.chains_to_fasta(chains_path=dataset.chains_file, chains=chains, alphabet=dataset.alphabet)
            utils.save_params(params_file=dataset.params_file, params=dataset.params)

    print("\n")
    return chains