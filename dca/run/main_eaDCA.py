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
import dca.run.training as training
import dca.run.algo as algo

def main(infile_path : str,
         chains_file : str,
         params_file : str,
         nchains : int = 1000,
         target_pearson : float = 0.8):
    """ 
    Main function running the Direct Coupling Analysis.
    """

    dataset = loader.DatasetDCA(infile_path, chains_file, params_file)
    f_single, f_double = algo.calc_freq(mat=dataset.mat)
    dataset.params['fields'] = torch.log(f_single)

    chains = dataset.produce_chains(nchains=nchains)

    p_single, p_double = algo.calc_freq(chains)
    print("pi_train gap freq: ", p_single[:, 0].mean())

    begin_training(
        chains=chains,
        dataset = dataset,
        f_single=f_single,
        f_double=f_double,
        p_single=p_single,
        p_double=p_double,
        target_pearson=target_pearson
    )


def begin_training(chains : torch.Tensor,
                   dataset: loader.DatasetDCA,
                   f_single : torch.Tensor, 
                   f_double : torch.Tensor,
                   p_single : torch.Tensor,
                   p_double : torch.Tensor,
                   target_pearson : float,
                   factivate : float = 0.001
                   ):
    
    device = f_single.device
    dtype = f_single.dtype

    pearson = max(0, float(algo.two_points_correlation(fij=f_double, pij=p_double, fi=f_single, pi=p_single)[0]))
    nchains, nnuc, nval = chains.shape
    log_weights = torch.zeros(size=(nchains,), device=chains.device, dtype=chains.dtype)
    logZ = (torch.logsumexp(log_weights, dim=0) - torch.log(torch.tensor(len(chains), device=device, dtype=dtype))).item()

    nactive = dataset.mask.sum()
    
    # Training loop
    time_start = time.time()
    log_likelihood = metrics.compute_log_likelihood(fi=f_single, fij=f_double, params=dataset.params, logZ=logZ)
        
    pbar = tqdm(initial=max(0, float(pearson)), total=0.8, colour="red", dynamic_ncols=True, ascii="-#",
                bar_format="{desc}: {percentage:.2f}%[{bar}] Pearson: {n:.3f}/{total_fmt} [{elapsed}]")
    pbar.set_description(f"Gap avg freq: {p_single[:,0].mean():.3f} - New active couplings: {0} - LL: {log_likelihood:.3f}")

    while pearson < target_pearson:
        nactive_old = nactive
        pij_Dkl = algo.get_two_point_freq(mat=chains)

        nactivate = int(((nnuc**2 * nval**2) - dataset.mask.sum().item()) * factivate)
        mask = metrics.activate_graph(mask=dataset.mask,
                              fij=f_double,
                              pij=pij_Dkl,
                              nactivate=nactivate,
                            )
        
        # New number of active couplings
        nactive = dataset.mask.sum()
        chains, params, log_weights = training.train_graph(chains=chains,
                                                           dataset=dataset,
                                                           f_single=f_single,
                                                           f_double=f_double,
                                                           target_pearson=target_pearson,
                                                           log_weights=log_weights,
        )

        graph_upd += 1
        
        p_single = algo.get_single_point_freq(data=chains)
        p_double = algo.get_two_point_freq(data=chains)
        
        pearson, slope = algo.two_points_correlation(fij=f_double, pij=p_double, fi=f_single, pi=p_single)
        logZ = (torch.logsumexp(log_weights, dim=0) - torch.log(torch.tensor(len(chains), device=device, dtype=dtype))).item()
        log_likelihood = metrics.compute_log_likelihood(fi=f_single, fij=f_double, params=params, logZ=logZ)