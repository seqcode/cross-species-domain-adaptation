import random
import numpy as np


class Jitter(object):
    def __init__(self, max_jitter, seq_len, profile_len):
        self.max_jitter = max_jitter
        self.seq_len = seq_len
        self.profile_len = profile_len
        
    def jitter_seq(self, seq, rand_jitter):
        # assuming first axis is sequence axis
        assert seq.shape[-1] > self.seq_len, seq.shape
        jittered = seq[..., rand_jitter : self.seq_len + rand_jitter]
        assert jittered.shape[-1] == self.seq_len, (jittered.shape, seq.shape, rand_jitter, self.seq_len)
        return jittered
    
    def jitter_profile(self, prof, rand_jitter):
        # assuming last axis is profile axis
        assert prof.shape[-1] > self.profile_len, prof.shape
        jittered = prof[..., rand_jitter : self.profile_len + rand_jitter]
        assert jittered.shape[-1] == self.profile_len, (jittered.shape, prof.shape, rand_jitter, self.profile_len)
        return jittered
        
    def __call__(self, args):
        # going to assume we will only ever jitter labeled examples
        rand_jitter = random.randint(0, self.max_jitter * 2)
        
        assert len(args) in [1, 3, 5], len(args)
        # sequence is always the first argument
        seq_onehot = args[0]
        
        # optionally, input profiles + logcounts for the expt and control
        if len(args) == 3:
            profile, _ = args[1:]
            control_profile = None
        elif len(args) == 5:
            profile, _, control_profile, _ = args[1:]
        else:
            profile, control_profile = None, None
            
        seq_onehot_jittered = self.jitter_seq(seq_onehot, rand_jitter)
        to_return = [seq_onehot_jittered]
        
        if profile is not None:
            profile_jittered = self.jitter_profile(profile, rand_jitter)
            logcounts_jittered = np.log(np.sum(profile_jittered, axis = -1) + 1)
            to_return.extend([profile_jittered, logcounts_jittered])
            
        if control_profile is not None:
            control_profile_jittered = self.jitter_profile(control_profile, rand_jitter)
            control_logcounts_jittered = np.log(np.sum(control_profile_jittered, axis = -1) + 1)
            to_return.extend([control_profile_jittered, control_logcounts_jittered])
        
        return to_return
    
    
class NoJitter(object):
    def __init__(self, max_jitter, seq_len, profile_len):
        self.max_jitter = max_jitter
        self.seq_len = seq_len
        self.profile_len = profile_len
        
    def jitter_seq(self, seq):
        jittered = seq[..., self.max_jitter : self.seq_len + self.max_jitter]
        assert jittered.shape[-1] == self.seq_len, (jittered.shape, seq.shape, self.seq_len)
        return jittered
    
    def jitter_profile(self, prof):
        jittered = prof[..., self.max_jitter : self.profile_len + self.max_jitter]
        assert jittered.shape[-1] == self.profile_len, (jittered.shape, prof.shape, self.profile_len)
        return jittered
        
    def __call__(self, args):
        # going to assume we will only ever jitter labeled examples
        rand_jitter = random.randint(0, self.max_jitter * 2)

        assert len(args) in [1, 3, 5], len(args)
        # sequence is always the first argument
        seq_onehot = args[0]
        
        # optionally, input profiles + logcounts for the expt and control
        if len(args) == 3:
            profile = args[1]
            control_profile = None
        elif len(args) == 5:
            profile, _, control_profile, _ = args[1:]
        else:
            profile, control_profile = None, None
            
        seq_onehot_jittered = self.jitter_seq(seq_onehot)
        to_return = [seq_onehot_jittered]

        if profile is not None:
            profile_jittered = self.jitter_profile(profile)
            logcounts_jittered = np.log(np.sum(profile_jittered, axis = -1) + 1)
            to_return.extend([profile_jittered, logcounts_jittered])
 
        if control_profile is not None:
            control_profile_jittered = self.jitter_profile(control_profile)
            control_logcounts_jittered = np.log(np.sum(control_profile_jittered, axis = -1) + 1)
            to_return.extend([control_profile_jittered, control_logcounts_jittered])

        return to_return