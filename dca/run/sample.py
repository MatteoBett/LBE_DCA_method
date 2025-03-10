#########################################################
#                        std Lib                        #
#########################################################
import os
from pathlib import Path
from typing import Dict
#########################################################
#                      Dependencies                     #
#########################################################
import torch
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
#########################################################
#                      Own modules                      #
#########################################################
import dca.run.algo as algo
import dca.tools.utils as utils
import dca.dataset.loader as loader
import dca.stats.metrics as metrics



def main(infile_path : str,
         sample_file: str,
         params_file: str,
         gaps_fraction : float,
         bias_flag : bool = False,
         max_sweeps : int = 500,
         beta : float = 1.0,
         nmix : int = 3,
         ngen : int = 5000):
         
    print("\n" + "".join(["*"] * 10) + f" Sampling from DCA model " + "".join(["*"] * 10) + "\n")
    dataset = loader.DatasetDCA(path_data=infile_path, params_file=params_file)
    device = dataset.device
    dtype = dataset.dtype
    tokens = dataset.alphabet
    folder = os.path.dirname(sample_file)

    print(f"Loading parameters from {dataset.params_file}...")
    params = utils.load_params(params_file=dataset.params_file, device=device, dtype=dtype)
    params["fields"] = torch.where(torch.isinf(params["fields"]), torch.tensor(-1e10, device=params["fields"].device), params["fields"])
    
    results_mix = algo.compute_mixing_time(
        data=dataset.mat,
        params=params,
        n_max_sweeps=max_sweeps,
        beta=beta,
    )
    mixing_time = results_mix["t_half"][-1]
    print(f"Measured mixing time (if converged): {mixing_time} sweeps")
    
    samples = torch.randint(low=0, high=dataset.nval, size=(ngen, dataset.nnuc), device=device)
    samples = loader.one_hot(samples, num_classes=dataset.nval, device=dataset.device, dtype=dataset.dtype)

    f_single, f_double = algo.calc_freq(mat=dataset.mat)

    results_sampling = {
        "nsweeps" : [],
        "pearson" : [],
        "slope" : [],
    }

    pbar = tqdm(
        initial=max(0, nmix * mixing_time),
        total=nmix * mixing_time,
        colour="red",
        dynamic_ncols=True,
        leave=False,
        ascii="-#",
        bar_format="{desc} {percentage:.2f}%[{bar}] Pearson: {n:.3f}/{total_fmt} [{elapsed}]"
    )  

    for i in range(nmix * mixing_time):
        samples, params = algo.gibbs_sampling(chains=samples, params=params, nsweeps=1, beta=beta)
        p_single, p_double = algo.calc_freq(mat=samples)
    
        pearson, slope = metrics.two_points_correlation(fi=f_single, pi=p_single, fij=f_double, pij=p_double)
        results_sampling["nsweeps"].append(i)
        results_sampling["pearson"].append(pearson)
        results_sampling["slope"].append(slope)
        pbar.set_description(f"Step: {i} - Gap avg freq: {p_single[:,0].mean():.3f}")
        
    pbar.close()

    p_single, p_double = algo.calc_freq(mat=samples)
    print("Gap average frequency: ", p_single[:, 0].mean().item())
    print("Computing the energy of the samples...")
    energies = metrics.compute_energy(samples, params=params).cpu().numpy()
    
    print("Saving the samples...")
    headers = [f">sequence {i+1} | DCAenergy: {energies[i]:.3f}" for i in range(ngen)]
    utils.chains_to_fasta(
        chains_path= sample_file,
        chains=samples,
        alphabet=tokens,
        headers=headers,
    )
    
    print("Writing sampling log...")
    df_mix_log = pd.DataFrame.from_dict(results_mix)    
    df_mix_log.to_csv(
        folder / Path(f"mix.log"),
        index=False
    )
    df_samp_log = pd.DataFrame.from_dict(results_sampling)    
    df_samp_log.to_csv(
        folder / Path(f"sampling.log"),
        index=False
    )
    
    print(f"Done, results saved in {str(folder)}")
