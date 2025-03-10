#########################################################
#                        std Lib                        #
#########################################################
import os, sys, argparse, shutil

#########################################################
#                      Dependencies                     #
#########################################################
import torch

#########################################################
#                      Own modules                      #
#########################################################
import dca.run.main_DCA as main_DCA
import dca.run.sample as sample
import dca.dataset.loader as loader
import dca.viz.display as display

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    family_dir = os.path.join(base_dir, r'data/input_test')
    outdir = os.path.join(base_dir,r'output')
    fig_dir = os.path.join(base_dir,r'output/figures')

    model_type = "eaDCA"
    alphabet = "-AUCG"
    target_pearson = 0.95
    target_gap_fraction = 0.2

    do_one = True
    plotting = True
    sampling = True
    run_generation = True

    bias = True 
    indel = False

    if not indel:
        out = 'raw'
        family_dir = os.path.join(family_dir, 'raw')

        
    display.get_summary(family_dir)

    for family_file, infile_path in loader.family_stream(family_dir=family_dir):
        if bias:
            family_outdir = os.path.join(outdir, "sequences", out, family_file.split('.')[0], "biased")
        else:
            family_outdir = os.path.join(outdir, "sequences", out, family_file.split('.')[0], "non_biased")
        
        os.makedirs(family_outdir, exist_ok=True)

        chain_file = os.path.join(family_outdir, "chains.fasta") #NOT converged
        params_dca = os.path.join(family_outdir, "params.json")
        sample_seq = os.path.join(family_outdir, f"generated.fasta") #sequences sampled from the converged model
        
        if run_generation:
            main_DCA.main(infile_path=infile_path,
                            chains_file=chain_file, 
                            params_file=params_dca,
                            target_pearson=target_pearson)
        
        if sampling:
            sample.main(infile_path=infile_path, 
                        sample_file=sample_seq,
                        params_file=params_dca,
                        gaps_fraction=target_gap_fraction,
                        bias_flag=bias)

        base_outdir = "/".join(family_outdir.split("/")[:-1])

        biased_seqs = os.path.join(outdir, "sequences", out, family_file.split('.')[0], "biased", "generated.fasta")
        unbiased_seqs = os.path.join(outdir, "sequences", out, family_file.split('.')[0], "non_biased", "generated.fasta")
        biased_params = os.path.join(outdir, "sequences", out, family_file.split('.')[0], "biased", "params.json")
        unbiased_params = os.path.join(outdir, "sequences", out, family_file.split('.')[0], "non_biased", "params.json")

        if plotting:
            fig_dir = os.path.join(fig_dir, family_file.split('.')[0])
            os.makedirs(fig_dir, exist_ok=True)
            display.homology_vs_gaps(chains_file_ref=unbiased_seqs, 
                                     infile_path=infile_path, 
                                     chains_file_bias=biased_seqs,
                                     indel=indel, 
                                     fig_dir=fig_dir,
                                     params_path_unbiased=unbiased_params,
                                     params_path_biased=biased_params,
                                     alphabet=alphabet)
        if do_one:
            break