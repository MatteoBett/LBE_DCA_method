#########################################################
#                        std Lib                        #
#########################################################
import os, sys, argparse, shutil
from collections import Counter

#########################################################
#                      Dependencies                     #
#########################################################
from Bio import SeqIO

#########################################################
#                      Own modules                      #
#########################################################

def get_summary(all_fam_dir : str):
    template = "{0:<30} {1:<50}"
    for path, directories, files in os.walk(all_fam_dir):
        if files != []:
            for f in files:
                print(f.split('.')[0])
                path_file = os.path.join(path, f)
                seqs = Counter("".join([str(record.seq) for record in SeqIO.parse(path_file, 'fasta')]))
                seqs = {key : val/sum(seqs.values()) for key, val in seqs.items()}

                for key, freq in seqs.items():
                    print(template.format(key, freq))
