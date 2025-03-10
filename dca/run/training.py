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

def get_target_gap_distribution(frac_target : float, 
                                data_distrib : torch.Tensor, 
                                distrib : torch.Tensor, 
                                passed : list = []) -> torch.Tensor:
    """ Determines the target frequency distribution of gaps given an overall mean """
    if frac_target <= 0:
        return distrib

    else:
        for index, val in enumerate(data_distrib):
            target_val = val*(frac_target/data_distrib.mean())
            if target_val > 1 and index not in passed:
                target_val = 1
                passed.append(index)
            
            new_val = distrib[index] + target_val
            if new_val > 1:
                passed.append(index)
                distrib[index] = 1
            else:
                distrib[index] = new_val

        unused_frac = frac_target-distrib.mean()
        return get_target_gap_distribution(frac_target=unused_frac,
                                           data_distrib=data_distrib,
                                           distrib=distrib,
                                           passed=passed)




def train_graph(chains : torch.Tensor,
                dataset : loader.DatasetDCA,
                f_single : torch.Tensor,
                f_double : torch.Tensor,
                target_pearson : float,
                log_weights : torch.Tensor,
                gaps_fraction : float,
                bias_flag : bool = False,
                progress_bar : bool = True,
                max_epochs : int = 250
                ):
    
    device = f_single.device
    dtype = f_single.dtype
    L, q = f_single.shape
    time_start = time.time() 


    fi_target_gap_distribution = f_single[:, 0].cpu()
    target_gap_distribution = torch.zeros((len(fi_target_gap_distribution), 1), dtype=torch.float32).cpu()
    target_gap_distribution = get_target_gap_distribution(frac_target=gaps_fraction, 
                                                            data_distrib=fi_target_gap_distribution, 
                                                            distrib=target_gap_distribution)
    
    print("Targetted average gap frequency :", target_gap_distribution.mean().item())

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

    while dataset['gaps_lr'] > 0.00001:
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
        chains, dataset.params = algo.gibbs_sampling(chains=chains, 
                                    params=dataset.params,
                                    gaps_target_dist=target_gap_distribution,
                                    bias_flag=bias_flag)

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