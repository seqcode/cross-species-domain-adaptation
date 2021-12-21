import os
import sys
assert len(sys.argv) > 3   # expecting root, source species, TF


GENOMES = { "mm10" : "/users/kcochran/genomes/mm10_no_alt_analysis_set_ENCODE.fasta",
            "hg38" : "/users/kcochran/genomes/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta" }

#ROOT = "/users/kcochran/projects/domain_adaptation_nosexchr/"
ROOT = sys.argv[1]
PEAKS_DIR = ROOT + "/data/"
BIGWIGS_DIR = ROOT + "/profile_model_data/"
MODEL_SAVE_DIR = ROOT + "/models/profile_models/"

SPECIES = ["mm10", "hg38"]
train_species = sys.argv[2]
assert train_species in SPECIES, train_species

TFS = ["CTCF", "CEBPA", "Hnf4a", "RXRA"]
tf = sys.argv[3]
assert tf in TFS, tf


MAX_JITTER = 200
INPUT_SEQ_LEN = 2114
OUTPUT_PROF_LEN = 1000

import gzip
from collections import defaultdict
import random
import numpy as np
from pyfaidx import Fasta
import pyBigWig
from torch.utils.data import Dataset


def get_filepaths(train_val_test, species, tf, unbound = False):
    assert train_val_test in ["train", "val", "test"], train_val_test
    if unbound:
        assert train_val_test == "train"
    assert species in SPECIES, species
    assert tf in TFS, tf
    
    filepaths = dict()
    filepaths["pos_bw"] = BIGWIGS_DIR + species + "/" + tf + "/all_reps.pos.bigWig"
    filepaths["neg_bw"] = BIGWIGS_DIR + species + "/" + tf + "/all_reps.neg.bigWig"
    filepaths["pos_control_bw"] = BIGWIGS_DIR + species + "/" + tf + "_control/all_reps.pos.bigWig"
    filepaths["neg_control_bw"] = BIGWIGS_DIR + species + "/" + tf + "_control/all_reps.neg.bigWig"
    filepaths["genome_file"] = GENOMES[species]
    
    if train_val_test == "train":
        if unbound:
            filepaths["peakfile"] = PEAKS_DIR + species + "/" + tf + "/filtered_peaks_chr3toY_gc_matched_neg.bed"
        else:
            filepaths["peakfile"] = PEAKS_DIR + species + "/" + tf + "/filtered_peaks_chr3toY.bed"
    elif train_val_test == "val":
        filepaths["peakfile"] = PEAKS_DIR + species + "/" + tf + "/filtered_peaks_chr1.bed"
    else:
        print("Loading test set.")
        filepaths["peakfile"] = PEAKS_DIR + species + "/" + tf + "/filtered_peaks_chr2.bed"

    return filepaths


from torch.utils.data import ConcatDataset
from data_transforms import *
from generators import *

def get_twosource_generator(train_val_test, species, tf):
    pos_gen = Generator(get_filepaths(train_val_test, species, tf),
                        seq_len = INPUT_SEQ_LEN,
                        profile_len = OUTPUT_PROF_LEN,
                        max_jitter = MAX_JITTER,
                        transform = Jitter(MAX_JITTER, INPUT_SEQ_LEN, OUTPUT_PROF_LEN))
    neg_gen = Generator(get_filepaths(train_val_test, species, tf, unbound = True),
                        seq_len = INPUT_SEQ_LEN,
                        profile_len = OUTPUT_PROF_LEN,
                        max_jitter = MAX_JITTER,
                        transform = Jitter(MAX_JITTER, INPUT_SEQ_LEN, OUTPUT_PROF_LEN))
    concat_gen = ConcatDataset((pos_gen, neg_gen))
    concat_gen.pos_idx_list = range(len(pos_gen))
    concat_gen.neg_idx_list = range(len(pos_gen), len(pos_gen) + len(neg_gen))
    return concat_gen


from torch.utils.data import Sampler
import torch
import numpy as np
from torch.utils.data import DataLoader

class BatchedCollatedSampler(Sampler):

    def __init__(self, pos_n, neg_n, batch_size):
        # pos_n, neg_n = # of examples in each class
        self.pos_n = pos_n
        self.neg_n = neg_n
        self.batch_size = batch_size
        self.ratio_pos_to_neg = 3
        
        self.neg_n_per_batch = batch_size // (self.ratio_pos_to_neg + 1)
        self.pos_n_per_batch = batch_size - self.neg_n_per_batch
        assert pos_n >= self.pos_n_per_batch and neg_n >= self.neg_n_per_batch 
        self.num_batches = min(pos_n // self.pos_n_per_batch, neg_n // self.neg_n_per_batch)
        

    def set_sample_order(self):
        # need batch size to be multiple of minimum # of examples
        # in each class needed to satisfy the ratio specified
        assert self.batch_size % (self.ratio_pos_to_neg + 1) == 0
        pos_perm = torch.randperm(self.pos_n, generator=None).tolist()
        # assuming the datasets are concat'd as [all pos examples] and then [all neg examples]
        neg_perm = [i + self.pos_n for i in torch.randperm(self.neg_n, generator=None).tolist()]

        batches = []
        for batch_i in range(self.num_batches):
            idxs_for_batch = []
            idxs_for_batch.extend(pos_perm[self.pos_n_per_batch * batch_i : self.pos_n_per_batch * (batch_i + 1)])
            idxs_for_batch.extend(neg_perm[self.neg_n_per_batch * batch_i : self.neg_n_per_batch * (batch_i + 1)])
            batches.append(idxs_for_batch)
            
        self.sample_order = batches
        return np.array(batches).flatten()


    def __iter__(self):
        yield from self.set_sample_order()

    def __len__(self):
        return self.num_batches


twosource_gen = get_twosource_generator("train", train_species, tf)

BATCH_SIZE = 32
train_data_loader = DataLoader(twosource_gen,
               batch_size = BATCH_SIZE,
               sampler = BatchedCollatedSampler(len(twosource_gen.pos_idx_list),
                                                len(twosource_gen.neg_idx_list),
                                                BATCH_SIZE))


import torch
from attr_prior_utils import *   # Alex's code
from torch.utils.data import DataLoader
from model_arch import *
import pandas as pd
from data_transforms import *

epochs = 30


save_dir = ROOT + "models/profile_models/" + train_species + "/" + tf + "/"
os.makedirs(save_dir, exist_ok=True)
    
    
val_species = "hg38" if train_species == "mm10" else "mm10"

print("Loading source val data.")
source_val_data_loader = DataLoader(Generator(get_filepaths("val", train_species, tf),
                                                  seq_len = INPUT_SEQ_LEN,
                                                  profile_len = OUTPUT_PROF_LEN,
                                                  max_jitter = MAX_JITTER,
                                                  transform = NoJitter(MAX_JITTER, INPUT_SEQ_LEN, OUTPUT_PROF_LEN)),
                                        batch_size = 32, shuffle = False)
print("Loading target val data.")
target_val_data_loader = DataLoader(Generator(get_filepaths("val", val_species, tf),
                                                  seq_len = INPUT_SEQ_LEN,
                                                  profile_len = OUTPUT_PROF_LEN,
                                                  max_jitter = MAX_JITTER,
                                                  transform = NoJitter(MAX_JITTER, INPUT_SEQ_LEN, OUTPUT_PROF_LEN)),
                                        batch_size = 32, shuffle = False)


learning_rate = 0.001
counts_weight = 10
num_filters = 64
num_layers = 8


# initialize the model
model = BPNetModel(n_filters=num_filters,
                  n_layers=num_layers)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# train!
print("Training (" + tf + ", " + train_species + ")...")

model.cuda()
model.fit(train_data_loader, optimizer,
          source_val_data_loader, target_val_data_loader,
          counts_weight = counts_weight,
          max_epochs = epochs)
model.cpu()

# Saving best models...
save_dir = MODEL_SAVE_DIR + train_species + "-trained/" + tf + "/"
os.makedirs(save_dir, exist_ok=True)
print("Best-model auPRC, source species:", model.best_profile_metric)
print("Best-model auPRC, target species:", model.target_profile_metric)
model.load_state_dict(model.best_state_for_profiles)
torch.save(model, save_dir + "bestprof.model")

# Saving metrics...
hist_dict = dict()
hist_dict["train_profile_loss"] = model.train_profile_losses_by_epoch
hist_dict["train_counts_loss"] = model.train_counts_losses_by_epoch
hist_dict["source_val_profile_loss"] = model.val_profile_losses_by_epoch
hist_dict["source_val_counts_loss"] = model.val_counts_losses_by_epoch
hist_dict["target_val_profile_loss"] = model.target_val_profile_losses_by_epoch
hist_dict["target_val_counts_loss"] = model.target_val_counts_losses_by_epoch

df = pd.DataFrame.from_dict(hist_dict)
df.to_csv(save_dir + "bestprof_metrics.csv", index=False)

print("Done!")
