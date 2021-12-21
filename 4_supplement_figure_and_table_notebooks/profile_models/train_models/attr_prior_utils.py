'''
Author of code: Alex Tseng
See https://github.com/amtseng/fourier_attribution_priors
'''


import torch
import numpy as np
import scipy.ndimage

att_prior_grad_smooth_sigma = 3
freq_limit = 150
limit_softness = 0.2


def smooth_tensor_1d(input_tensor, smooth_sigma):
    """
    Smooths an input tensor along a dimension using a Gaussian filter.
    Arguments:
        `input_tensor`: a A x B tensor to smooth along the second dimension
        `smooth_sigma`: width of the Gaussian to use for smoothing; this is the
            standard deviation of the Gaussian to use, and the Gaussian will be
            truncated after 1 sigma (i.e. the smoothing window is
            1 + (2 * sigma); sigma of 0 means no smoothing
    Returns an array the same shape as the input tensor, with the dimension of
    `B` smoothed.
    """
    # Generate the kernel
    if smooth_sigma == 0:
        sigma, truncate = 1, 0
    else:
        sigma, truncate = smooth_sigma, 1
    base = np.zeros(1 + (2 * sigma))
    base[sigma] = 1  # Center of window is 1 everywhere else is 0
    kernel = scipy.ndimage.gaussian_filter(base, sigma=sigma, truncate=truncate)
    kernel = torch.tensor(kernel).cuda()

    # Expand the input and kernel to 3D, with channels of 1
    # Also make the kernel float-type, as the input is going to be of type float
    input_tensor = torch.unsqueeze(input_tensor, dim=1)
    kernel = torch.unsqueeze(torch.unsqueeze(kernel, dim=0), dim=1).float()

    smoothed = torch.nn.functional.conv1d(input_tensor, kernel, padding=sigma)

    return torch.squeeze(smoothed, dim=1)



def fourier_att_prior_loss(input_grads, freq_limit, limit_softness,
                            att_prior_grad_smooth_sigma):
        """
        Computes an attribution prior loss for some given training examples,
        using a Fourier transform form.
        Arguments:
            `input_grads`: a B x L x 4 tensor, where B is the batch size, L is
                the length of the input; this needs to be the gradients of the
                input with respect to the output; this should be
                *gradient times input*
            `freq_limit`: the maximum integer frequency index, k, to consider for
                the loss; this corresponds to a frequency cut-off of pi * k / L;
                k should be less than L / 2
            `limit_softness`: amount to soften the limit by, using a hill
                function; None means no softness
            `att_prior_grad_smooth_sigma`: amount to smooth the gradient before
                computing the loss
        Returns a single scalar Tensor consisting of the attribution loss for
        the batch.
        """
        input_grads = input_grads.permute(0, 2, 1)
        
        abs_grads = torch.sum(torch.abs(input_grads), dim=2)

        # Smooth the gradients
        grads_smooth = smooth_tensor_1d(abs_grads, att_prior_grad_smooth_sigma)

        # Calc loss
        if grads_smooth.nelement():
            pos_fft = torch.fft.rfft(grads_smooth, dim=1)
            pos_mags = torch.abs(pos_fft)
            pos_mag_sum = torch.sum(pos_mags, dim=1, keepdim=True)
            pos_mag_sum[pos_mag_sum == 0] = 1  # Keep 0s when the sum is 0
            pos_mags = pos_mags / pos_mag_sum

            # Cut off DC
            pos_mags = pos_mags[:, 1:]

            # Construct weight vector
            weights = torch.ones_like(pos_mags).cuda()
            if limit_softness is None:
                weights[:, freq_limit:] = 0
            else:
                x = torch.arange(1, pos_mags.size(1) - freq_limit + 1).cuda().float()
                weights[:, freq_limit:] = 1 / (1 + torch.pow(x, limit_softness))

            # Multiply frequency magnitudes by weights
            pos_weighted_mags = pos_mags * weights

            # Add up along frequency axis to get score
            pos_score = torch.sum(pos_weighted_mags, dim=1)
            pos_loss = 1 - pos_score
            return torch.mean(pos_loss)
        else:
            return torch.zeros(1).cuda()
