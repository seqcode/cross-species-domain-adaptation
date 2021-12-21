import torch
from attr_prior_utils import *   # Alex Tseng's code: https://github.com/amtseng/fourier_attribution_priors
from torch.utils.data import DataLoader


def MLLLoss(logps, true_counts):
    """ ""Adapted from Alex Tseng." - Jacob Schreiber" - Kelly Cochran
    """
    # Multinomial probability = n! / (x1!...xk!) * p1^x1 * ... pk^xk
    # Log prob = log(n!) - (log(x1!) ... + log(xk!)) + x1log(p1) ... + xklog(pk)
    log_fact_sum = torch.lgamma(torch.sum(true_counts, dim=-1) + 1)
    log_prod_fact = torch.sum(torch.lgamma(true_counts + 1), dim=-1)
    log_prod_exp = torch.sum(true_counts * logps, dim=-1) 
    return -torch.mean(log_fact_sum - log_prod_fact + log_prod_exp)


def trim_profile_by_len(prof, true_prof_len, add_batch_axis = False):
    # When the model outputs a profile, it outputs too wide of an array.
    # This is because the conv layers need to have their edges shaved off,
    # just a little each layer. The edges see zero-padding that is not part
    # of the real sequence (or the real output of the previous conv layer).
    # We just want to keep the central part of the model output that corresponds
    # to where there wasn't any zero-padding.
    
    # So, use this function **every** **time** you get model output.
    if len(prof.shape) == 3:
        midpoint = prof.shape[2] / 2
        return prof[:, :, int(midpoint - true_prof_len / 2) : int(midpoint + true_prof_len / 2)]
    
    if len(prof.shape) == 2:
        midpoint = prof.shape[1] / 2
        return prof[:, int(midpoint - true_prof_len / 2) : int(midpoint + true_prof_len / 2)]
    else:
        midpoint = prof.shape[0] / 2
        if add_batch_axis:
            return prof[None, int(midpoint - true_prof_len / 2) : int(midpoint + true_prof_len / 2)]
        else:
            return prof[int(midpoint - true_prof_len / 2) : int(midpoint + true_prof_len / 2)]

        
def pad_control_profile(control_profile, target_len):
    # assuming last axis is profile len axis
    current_shape = control_profile.shape
    current_len = current_shape[-1]
    target_shape = tuple([i for i in current_shape[:-1]] + [target_len])
    padded_profile = torch.zeros(target_shape)
    offset = int((target_len - current_len) / 2)
    padded_profile[..., offset:offset + current_len] = control_profile
    #print(control_profile.shape, padded_profile.shape)
    return padded_profile



# architecture implementation borrowed from Jacob Schreiber, then modified by Kelly Cochran

class BPNetModel(torch.nn.Module):
    def __init__(self, n_filters=64,
                 n_layers=6,
                 input_seq_len=2114, output_prof_len=1000,
                 iconv_kernel_size=21,
                 deconv_kernel_size=75):
        super(BPNetModel, self).__init__()
        self.n_layers = n_layers
        self.input_seq_len = input_seq_len
        self.untrimmed_prof_len = input_seq_len - iconv_kernel_size - deconv_kernel_size + 2 
        self.output_prof_len = output_prof_len
        
        # this is the "bottom" conv layer that takes in sequence for input
        # the 4 is for the axis of length 4 in one-hot encoded sequence
        self.iconv = torch.nn.Conv1d(4, n_filters, kernel_size=iconv_kernel_size)
        
        # this list of layers is all of the other conv layers that come after the first one
        # they use dilation to capture longer-range interactions between parts of the input sequence
        self.rconvs = torch.nn.ModuleList([
        torch.nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=2**i, dilation=2**i) for i in range(1, self.n_layers+1)])
        
        # this is technically a deconvolutional layer (implementation is equivalent)
        # this outputs 2 channels, one for each strand
        # this effectively creates the profile output
        self.penultimate_conv = torch.nn.Conv1d(in_channels = n_filters, out_channels = 2, kernel_size=deconv_kernel_size)
        
        # we apply one more convolution to integrate in the control track
        self.final_conv = torch.nn.Conv1d(in_channels = 2, out_channels = 2, kernel_size=1, groups=1)
        
        # will also need these layers -- they don't have weights associated with them
        # so we can use one copy of them whenever
        self.relu = torch.nn.ReLU()
        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)
        
        # for the counts (not-profile) output:
        # we average-pool to reduce dimensions
        self.pool = torch.nn.AvgPool1d(self.output_prof_len)
        # then use a fully connected layer to convert that to 2 outputs, one per strand
        self.linear = torch.nn.Linear(in_features = n_filters, out_features = 2)
        # finally, a convolution to integrate the control track counts
        self.counts_conv = torch.nn.Conv1d(in_channels = 2, out_channels = 2, kernel_size=1, groups=1)
        
        # store performance metrics
        self.train_profile_losses_by_epoch = []
        self.train_counts_losses_by_epoch = []
        self.val_profile_losses_by_epoch = []
        self.val_counts_losses_by_epoch = []
        self.target_val_profile_losses_by_epoch = []
        self.target_val_counts_losses_by_epoch = []
        
        # for early stopping
        self.best_state_for_profiles = self.state_dict()
        self.best_profile_metric = float("inf")
        self.target_profile_metric = float("inf")
        
        
    def forward(self, inputs):
        sequence, control_profile, control_logcounts = inputs
        #print([thing.shape for thing in inputs])
        # apply first conv layer and ReLU
        X = self.relu(self.iconv(sequence))
        # apply subsequent conv layers in order, with residual connection (add)
        for i in range(self.n_layers):
            X_conv = self.relu(self.rconvs[i](X))
            X = torch.add(X, X_conv)

        # apply final layers for profile task output
        y_profile = self.penultimate_conv(X)
        y_profile = torch.add(y_profile, control_profile)   ######
        y_profile = self.final_conv(y_profile)
        y_profile = y_profile.squeeze()
        
        # apply final layers from counts task output
        X = trim_profile_by_len(X, self.output_prof_len)
        y_logcounts = self.pool(X)[:, :, 0]  # removing an unnecessary axis
        y_logcounts = self.linear(y_logcounts)
        y_logcounts = torch.add(y_logcounts, control_logcounts)    ######
        y_logcounts = y_logcounts[:, :, None]  # add extra axis
        y_logcounts = self.counts_conv(y_logcounts)
        
        return y_profile, y_logcounts

    
    def fit(self, train_data_loader, optimizer,
            source_val_data_loader, target_val_data_loader,
        max_epochs=30, counts_weight = 50, verbose = True):
        
        torch.backends.cudnn.enabled = True
        
        if verbose:
            print("Epoch\tTrain_Prof\tTrain_Counts\tVal_Prof\tVal_Counts\tTarget_Val_Prof\tTarget_Val_Counts")
        
        # begin training loop
        for epoch in range(max_epochs):
            # clear memory from GPU to avoid OOM errors
            torch.cuda.empty_cache()
            
            # training involves gradients, so set this to true
            torch.set_grad_enabled(True)
            # set model into training mode (as opposed to self.eval())
            self.train()
            
            train_profile_losses = []
            train_prior_losses = []
            train_logcounts_losses = []
            # loop over batches in training set
            for seq_onehot_batch, true_profile_batch, true_logcounts_batch, control_profile_batch, control_logcounts_batch in train_data_loader:
                # zero out the gradients on all the model's weights to start over
                optimizer.zero_grad()
                # move the input sequences onto the GPU
                seq_onehot_batch = seq_onehot_batch.cuda()
                
                control_profile_batch = pad_control_profile(control_profile_batch,
                                                            self.untrimmed_prof_len).cuda()
                control_logcounts_batch = control_logcounts_batch.cuda()
                
                # Attribution priors
                ##########
                
                seq_onehot_batch.requires_grad = True  # Reset gradient required for attr. priors
                
                # get predicted logcounts for the entire batch
                _, pred_logcounts = self((seq_onehot_batch, control_profile_batch, control_logcounts_batch))
                
                # loop over every example in batch (needed for attr. priors)
                for ex_idx in range(seq_onehot_batch.shape[0]):   # along batch axis
                    # get example
                    seq_onehot = seq_onehot_batch[ex_idx:ex_idx+1]
                    control_profile = control_profile_batch[ex_idx:ex_idx+1]
                    control_logcounts = control_logcounts_batch[ex_idx:ex_idx+1]
                    # get predicted profile, pre-trimming and softmax activation
                    pred_logits, _ = self((seq_onehot, control_profile, control_logcounts))
                    # trim profile (see trim_profile_by_len)
                    pred_logits_trimmed = trim_profile_by_len(pred_logits, self.output_prof_len)
                    # apply softmax activation so that predicted profile sums to 1
                    pred_profile = self.logsoftmax(pred_logits_trimmed)

                    # for attr. priors, we will mean-normalize the pre-activation profile
                    # and then multiply that by the post-activation profile
                    # (this weights the predicted peaks more highly than predicted regions of low signal)
                    norm_pred_logits = pred_logits_trimmed - torch.mean(pred_logits_trimmed, dim=-1, keepdim=True)
                    norm_pred_logits = norm_pred_logits * pred_profile
                    
                    # Compute the gradients of this output with respect to the input
                    input_grads, = torch.autograd.grad(norm_pred_logits, seq_onehot,
                        grad_outputs=torch.ones(norm_pred_logits.size()).cuda(),
                        retain_graph=True, create_graph=True)

                    # then multiply those gradients by the original sequence to get attributions
                    input_grads = input_grads * seq_onehot  # Gradient * input

                    # feed those attributions into the attr. prior loss function
                    att_prior_loss = fourier_att_prior_loss(input_grads, freq_limit, limit_softness,
                            att_prior_grad_smooth_sigma)

                    # calculate the gradients of that loss on all the model weights
                    # (the weights won't actually update until we cal optimizer.step() later)
                    att_prior_loss.backward(retain_graph=True) # this bool is needed for later backward calls
                    train_prior_losses.append(att_prior_loss.item())

                    ######### end attribution prior stuff

                    
                    # move the true profile to the GPU for loss calculation
                    true_profile = true_profile_batch[ex_idx].cuda()
                    # calculate the loss between predicted and true profile
                    profile_loss = MLLLoss(pred_profile, true_profile)
                    # calculate the gradients of that loss on all the model weights
                    profile_loss.backward(retain_graph=True)  # this bool is needed for second backward call
                    train_profile_losses.append(profile_loss.item())
                
                # then calculate the (mean-squared error) loss from the counts task
                true_logcounts_batch = true_logcounts_batch.cuda()
                logcounts_loss = torch.nn.MSELoss()(true_logcounts_batch.squeeze(), pred_logcounts.squeeze())
                # weight by the counts_weight before applying to model weights
                logcounts_loss = logcounts_loss * counts_weight
                logcounts_loss.backward()
                train_logcounts_losses.append(logcounts_loss.item())
                
                # finally, apply all the loss gradients we've calculated to update the model weights
                optimizer.step()
                
                
                
            ### getting validation set performance
            
            # end train() mode
            self.eval()
            
            val_profile_losses = []
            val_logcounts_losses = []
            # loop over source validation set
            for seq_onehot, true_profile, true_logcounts, control_profile, control_logcounts in source_val_data_loader:
                seq_onehot = seq_onehot.cuda()
                true_profile = true_profile.cuda()
                true_logcounts = true_logcounts.cuda()
                control_profile = pad_control_profile(control_profile, self.untrimmed_prof_len).cuda()
                control_logcounts = control_logcounts.cuda()
                
                # get predictions, trim profile and apply activation
                pred_logits, pred_logcounts = self((seq_onehot, control_profile, control_logcounts))
                pred_logits_trimmed = trim_profile_by_len(pred_logits, self.output_prof_len)
                pred_profile_trimmed = self.logsoftmax(pred_logits_trimmed)
                
                # calculate losses
                profile_loss = MLLLoss(pred_profile_trimmed, true_profile)
                val_profile_losses.append(profile_loss.item())
                
                logcounts_loss = torch.nn.MSELoss()(true_logcounts.squeeze(), pred_logcounts.squeeze())
                logcounts_loss = logcounts_loss * counts_weight
                val_logcounts_losses.append(logcounts_loss.item())
                
            target_val_profile_losses = []
            target_val_logcounts_losses = []
            # loop over target validation set
            for seq_onehot, true_profile, true_logcounts, control_profile, control_logcounts in target_val_data_loader:
                seq_onehot = seq_onehot.cuda()
                true_profile = true_profile.cuda()
                true_logcounts = true_logcounts.cuda()
                control_profile = pad_control_profile(control_profile, self.untrimmed_prof_len).cuda()
                control_logcounts = control_logcounts.cuda()
                
                # get predictions, trim profile and apply activation
                pred_logits, pred_logcounts = self((seq_onehot, control_profile, control_logcounts))
                pred_logits_trimmed = trim_profile_by_len(pred_logits, self.output_prof_len)
                pred_profile_trimmed = self.logsoftmax(pred_logits_trimmed)
                
                # calculate losses
                profile_loss = MLLLoss(pred_profile_trimmed, true_profile)
                target_val_profile_losses.append(profile_loss.item())
                
                logcounts_loss = torch.nn.MSELoss()(true_logcounts.squeeze(), pred_logcounts.squeeze())
                logcounts_loss = logcounts_loss * counts_weight
                target_val_logcounts_losses.append(logcounts_loss.item())
            
            
            # report results of validation set performance
            to_print = [np.mean(train_profile_losses),
                        np.mean(train_logcounts_losses),
                        np.mean(val_profile_losses),
                        np.mean(val_logcounts_losses),
                        np.mean(target_val_profile_losses),
                        np.mean(target_val_logcounts_losses)]
            print(epoch + 1, "\t", "\t".join(["%.3f" % x for x in to_print]))
            
            # save train/val losses
            self.train_profile_losses_by_epoch.append(np.mean(train_profile_losses))
            self.train_counts_losses_by_epoch.append(np.mean(train_logcounts_losses))
            self.val_profile_losses_by_epoch.append(np.mean(val_profile_losses))
            self.val_counts_losses_by_epoch.append(np.mean(val_logcounts_losses))
            self.target_val_profile_losses_by_epoch.append(np.mean(target_val_profile_losses))
            self.target_val_counts_losses_by_epoch.append(np.mean(target_val_logcounts_losses))
            
            # for early stopping
            if np.mean(val_profile_losses) < self.best_profile_metric:
                self.best_profile_metric = np.mean(val_profile_losses)
                self.target_profile_metric = np.mean(target_val_profile_losses)
                self.best_state_for_profiles = self.state_dict()
