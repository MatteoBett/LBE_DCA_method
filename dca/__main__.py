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
import dca.run.main_eaDCA as main_eaDCA
import dca.dataset.loader as loader
import dca.viz.display as display

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    family_dir = os.path.join(base_dir, r'data/input_test')
    outdir = os.path.join(base_dir,r'output')
    fig_dir = os.path.join(base_dir,r'output/figures')

    model_type = "eaDCA"
    do_one = True

    run_generation = True
    bias = False 

    if not bias:
        out = 'raw'
        family_dir = os.path.join(family_dir, 'raw')
        
    display.get_summary(family_dir)

    for family_file, infile_path in loader.family_stream(family_dir=family_dir):
        if bias:
            family_outdir = os.path.join(outdir, "sequences", out, family_file.split('.')[0], "biased")
        else:
            family_outdir = os.path.join(outdir, "sequences", out, family_file.split('.')[0], "non_biased")
        
        os.makedirs(family_outdir, exist_ok=True)

        chain_file = os.path.join(family_outdir, "chains.fasta")
        params_dca = os.path.join(family_outdir, "params.dat")
        
        main_eaDCA.main(infile_path=infile_path,
                        chains_file=chain_file, 
                        params_file=params_dca)
        if do_one:
            break