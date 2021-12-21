import numpy as np
from pyfaidx import Fasta
import pyBigWig
import torch
from torch.utils.data import Dataset


MAX_JITTER = 200
INPUT_SEQ_LEN = 2114
OUTPUT_PROF_LEN = 1000



def expand_window(start, end, target_len):
    midpoint = (start + end) / 2
    if not midpoint.is_integer() and target_len % 2 == 0:
        midpoint = midpoint - 0.5
    if midpoint.is_integer() and target_len % 2 != 0:
        midpoint = midpoint - 0.5
    new_start = midpoint - target_len / 2
    new_end = midpoint + target_len / 2
    
    assert new_start.is_integer(), new_start
    assert new_end.is_integer(), new_end
    assert new_start >= 0
    assert new_end - new_start == target_len, (new_end, new_start, target_len)
    
    return int(new_start), int(new_end)



class Generator(Dataset):
    letter_dict = {
        'a':[1,0,0,0],'c':[0,1,0,0],'g':[0,0,1,0],'t':[0,0,0,1],
        'n':[0,0,0,0],'A':[1,0,0,0],'C':[0,1,0,0],'G':[0,0,1,0],
        'T':[0,0,0,1],'N':[0,0,0,0]}

    def __init__(self, filepaths_dict,
                 seq_len,
                 profile_len,
                 max_jitter,
                 transform = None,
                 return_labels = True, return_controls = True):
        
        for key in filepaths_dict:
            setattr(self, key, filepaths_dict[key])
        
        self.prof_len = profile_len
        self.max_jitter = max_jitter
        self.transform = transform
        self.return_labels = return_labels
        self.return_controls = return_controls
        self.seq_len = seq_len

        self.set_len()
        self.coords = self.get_coords()
        self.seqs_onehot = self.convert(self.coords)
        self.profiles, self.logcounts = self.get_profiles_and_logcounts(self.coords,
                                                                        self.pos_bw,
                                                                        self.neg_bw)
        self.control_profiles, self.control_logcounts = self.get_profiles_and_logcounts(self.coords,
                                                                                        self.pos_control_bw,
                                                                                        self.neg_control_bw)


    def __len__(self):
        return self.len
    
    
    def set_len(self):
        with open(self.peakfile) as f:
            self.len = sum([1 for _ in f])


    def get_coords(self):
        with open(self.peakfile) as posf:
            coords_tmp = [line.split()[:3] for line in posf]  # expecting bed file format
        
        coords = []
        for coord in coords_tmp:
            chrom, start, end = coord[0], int(coord[1]), int(coord[2])
            window_start, window_end = expand_window(start, end,
                                                     self.seq_len + 2 * self.max_jitter)
            coords.append((coord[0], window_start, window_end))  # no strand consideration
        return coords
            

    def get_profiles_and_logcounts(self, coords, pos_bw_file, neg_bw_file):
        profiles = []
        logcounts = []

        # pyBigWig can read from bigWig files and fetch data at a specific genomic region
        # we have two bigWig readers open, one for each DNA strand
        with pyBigWig.open(pos_bw_file) as pos_bw_reader:
            with pyBigWig.open(neg_bw_file) as neg_bw_reader:
                # for each example...
                for chrom, start, end in coords:
                    # we need to edit the start and end coords, to get a profile
                    # that is the right length to match the model's output size
                    # this is smaller than the input size (the size written in the coords files)
                    # because of the model's receptive field and deconv layer kernel width
                    prof_start, prof_end = expand_window(start, end,
                                                 self.prof_len + 2 * self.max_jitter)
                    
                    # read in profile values for the positive strand
                    pos_profile = np.array(pos_bw_reader.values(chrom, prof_start, prof_end))
                    # pyBigWig sometimes returns nan when the real data is just 0
                    pos_profile[np.isnan(pos_profile)] = 0
                    # read in profile values for the negative strand
                    neg_profile = np.array(neg_bw_reader.values(chrom, prof_start, prof_end))
                    neg_profile[np.isnan(neg_profile)] = 0
                    
                    # stick the strands together in an array of shape (2, profile_len)
                    profile = np.array([pos_profile, neg_profile])
                    
                    # derive values for the counts task by adding up profile
                    # we take the log bfor technical reasons -- counts are vaguely
                    # Poisson or negative-binomial distributed, and it is easier
                    # for the model to model them in log space because of that
                    pos_logcount = np.log(np.sum(pos_profile) + 1)
                    neg_logcount = np.log(np.sum(neg_profile) + 1)
                    
                    # stick the strands together in an array of shape (2,)
                    logcount = np.array([pos_logcount, neg_logcount])

                    profiles.append(profile)
                    logcounts.append(logcount)
                    
        profiles = np.array(profiles)
        logcounts = np.array(logcounts)
        return profiles, logcounts
                

    def convert(self, coords):
        # fetch the sequence for a given site/region in the genome, and then one-hot encode
        seqs_onehot = []
        with Fasta(self.genome_file) as converter:
            for coord in coords:
                chrom, start, stop = coord
                assert chrom in converter
                # get sequence
                seq = converter[chrom][start:stop].seq
                # convert to one-hot
                # this array will have shape (4, seq_len)
                # this is transposed relative to other code you've written, Kelly
                seq_onehot = np.array([self.letter_dict.get(x,[0,0,0,0]) for x in seq]).T
                seqs_onehot.append(seq_onehot)

        seqs_onehot = np.array(seqs_onehot)
        return seqs_onehot


    def __getitem__(self, batch_index):	
        # this function returns one batch's worth of data
        
        # get one-hot sequences for this batch
        onehot = self.seqs_onehot[batch_index]
        assert onehot.shape[0] > 0, onehot.shape
        to_return = [onehot]

        if self.return_labels:
            # get this batch's profiles and logcounts
            profiles = self.profiles[batch_index]
            logcounts = self.logcounts[batch_index]
            to_return.extend([profiles, logcounts])
            
        if self.return_controls:
            # get this batch's profiles and logcounts for the control track
            control_profiles = self.control_profiles[batch_index]
            control_logcounts = self.control_logcounts[batch_index]
            to_return.extend([control_profiles, control_logcounts])
             
        # run optional jittering on data, if applicable
        if self.transform is not None:
            to_return = self.transform(to_return)
        
        #print([thing.shape for thing in to_return])
        
        # convert numpy arrays to tensors for Pytorch to use
        to_return = [torch.tensor(x, dtype=torch.float) for x in to_return]
        return to_return



