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
         target_pearson : float,
         nchains : int = 1000
         ):
    """ 
    Main function running the Direct Coupling Analysis.
    """

    dataset = loader.DatasetDCA(path_data=infile_path, chains_file=chains_file, params_file=params_file)
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
                   ):
    
    device = f_single.device
    dtype = f_single.dtype

    pearson = max(0, float(metrics.two_points_correlation(fij=f_double, pij=p_double, fi=f_single, pi=p_single)[0]))
    nchains, nnuc, nval = chains.shape
    log_weights = torch.zeros(size=(nchains,), device=chains.device, dtype=chains.dtype)
    logZ = (torch.logsumexp(log_weights, dim=0) - torch.log(torch.tensor(len(chains), device=device, dtype=dtype))).item()
        
    chains = training.train_graph(chains=chains,
                                    dataset=dataset,
                                    f_single=f_single,
                                    f_double=f_double,
                                    target_pearson=target_pearson,
                                    log_weights=log_weights,
                                    )
    
    p_single, p_double = algo.calc_freq(mat=chains)
    
    pearson, slope = metrics.two_points_correlation(fij=f_double, pij=p_double, fi=f_single, pi=p_single)
    print("\n")

