'''
GeneRalized Autocalibrating Partial Parallel Acquisition (GRAPPA) implementation via PyTorch
'''

import torch
from torch import nn
import numpy as np
from tqdm import tqdm

def grappa(data, acs, kernel_size=[5, 5], lam=0.01, verbose=True, coils=[], gpu=False):
    # noinspection PyByteLiteral
    '''
    GeneRalized Autocalibrating Partial Parallel Acquisition (GRAPPA) implementation [1], [2].

    [1] Griswold, Mark A., et al. "Generalized autocalibrating partially parallel acquisitions (GRAPPA)."
        Magnetic Resonance in Medicine, 47.6 (2002): 1202-1210.
    [2] Uecker, Martin, et al. "ESPIRiT—an eigenvalue approach to autocalibrating parallel MRI: where SENSE meets GRAPPA."
        Magnetic resonance in medicine 71.3 (2014): 990-1001.

    General description of the estimation process:
    ----------------------------------------------
        This implementation assumes sub-sampling in the phase encode direction ONLY.

    :param data: Input acquired sub-sampled k-space data
                Assumed data structure (3D acquisition):  # Columns (Freq. encode), # Channels (Coils), # PE Lines, # Partitions (PE 2)
                Assumed data structure (2D acquisition):  # Columns (Freq. encode), # Channels (Coils), # PE Lines, # Slices # TODO: Add averages as batch size?
    :param acs: Input acquired auto-calibration signal (ACS)
                Assumed data structure (3D acquisition):  # Columns (Freq. encode), # Channels (Coils), # PE Lines, # Partitions (PE 2)
                Assumed data structure (2D acquisition):  # Columns (Freq. encode), # Channels (Coils), # PE Lines, # Slices # TODO: Add averages as batch size?
    :param R: (Phase encode) Sub-sampling factor. Currently this is assumed to be a scalar, even for a 3D scan
    :param kernel_size: GRAPPA kernel size. Currently supports only a 2D list, with odd values only
    :param lam: Tykhonov regularizatino weight (scalar)
    :return: Interpolated k-space samples
    '''

    # Determine GRAPPA kernel size
    if not isinstance(kernel_size, list):
        raise ValueError("Kernel dimensions must be a 2 items list only.")

    if len(kernel_size) != 2:
        raise ValueError("Kernel dimensions must be a 2 items list only.")

    # GRAPPA reconstruction
    # ---------------------------------------------------------------------------------
    # Permutation dimensions: # [(Columns, Coils, PE, Partitions),  (Partitions, Coils, PE, Columns)]
    #permute_dims = [(0, 1, 2, 3), (3, 1, 2, 0)]
    permute_dims = [(0, 1, 2, 3)]

    if not coils:
        ncoil = acs.shape[1] # Empty list - reconstruct for all coils
        coils = [f for f in range(ncoil)]
        kspace = torch.complex(torch.zeros(size=data.shape), torch.zeros(size=data.shape))
        if verbose: print("Working on all coils")
    else:
        ncoil = len(coils) # Reconstruct for selected coils only
        Zblock = torch.zeros(size=(data.shape[0], ncoil, data.shape[2], data.shape[3]))
        kspace = torch.complex(Zblock, Zblock)
        if verbose: print("Working on coils {}".format(coils))

    if gpu:
        kspace = kspace.cuda()

    for perdim in permute_dims:
        if verbose: print(" "); print("Orientation: {}".format(perdim)); print(" ")

        # Calculate weights
        if verbose: print("Calculating calibration matrix...")
        AtA = grappa_calibration_matrix(torch.permute(acs, perdim), kernel_size)

        # Loop over each coil
        for cc in coils:
            if verbose: print(" "); print("Coil #{}, total of {} coils to reconstruct".format(cc, ncoil))

            # Calculate and apply weights
            tmp, _ = ARC(cc, torch.permute(data, perdim), AtA, kernel_size, lam)

            # Concatenate per coil
            if cc == coils[0]:
                kspace_tmp = tmp
            else:
                kspace_tmp = torch.cat((kspace_tmp, tmp), dim=1)

        # Average repeated results
        kspace += torch.permute(kspace_tmp, perdim)

    return kspace / len(permute_dims)

def grappa_calibration_matrix(acs, kernel_size):

    # Extract patches
    unfold = torch.nn.Unfold(kernel_size=kernel_size, dilation=1, padding=0, stride=1)
    A = torch.transpose(unfold(acs), 1, 2) # Shape: Columns, L, Coils * /Pi_i kernel_size[i] (see PyTorch documentation for L) - this is a Block-Hankel matrix

    # More convenient to wor on A^H * A.
    AtA = torch.matmul(torch.conj(torch.transpose(A, 1, 2)), A) # Shape: Columns, L, L
    del A # Very large matrix

    return AtA


def est_kernel(AtA, kernel_size, ncoils, cc, idxc, idxc_flat, sampat, lam=0.01):
    '''
    Estimate kernel weights via a Tykhonov regularized least-squares solution
    :param AtA:
    :param kernel_size:
    :param ncoils:
    :param cc:
    :param idxc:
    :param sampat:
    :return: kernel - estimated GRAPPA kernel
    '''
    [sz, sx, sy] = AtA.shape

    # Don't use the center of the sampling pattern
    sampat[0, idxc[0], idxc[1], idxc[2]] = 0
    sampat0 = sampat[0, ...].detach().cpu().numpy() # Assuming sampling pattern is identical in all partitions (no 3D sub-sampling)
    idxA = np.ravel_multi_index(np.argwhere(sampat0).transpose(), dims=[ncoils, kernel_size[0], kernel_size[1]])
    lenA = len(idxA)

    Aty = AtA[:, :, idxc_flat] # A^H * y
    Aty = Aty[:, idxA] # A^H * y over the measured values only
    AtA = AtA[:, :, idxA]
    AtA = AtA[:, idxA, :]

    # Regularization parameter
    lam = (torch.norm(AtA.view(sz, -1), p='fro', dim=1) / lenA * lam).unsqueeze(1)
    if AtA.is_cuda:
        I = lam[:, :, None] * torch.eye(lenA, device=torch.device('cuda'))
    else:
        I = lam[:, :, None] * torch.eye(lenA)

    # Calculate weights
    kernel = torch.squeeze(torch.matmul(torch.linalg.inv(AtA + I), torch.unsqueeze(Aty, dim=-1)))

    if AtA.is_cuda:
        kernel_out = torch.complex(torch.zeros(sampat.shape, device=torch.device('cuda')),
                                   torch.zeros(sampat.shape, device=torch.device('cuda'))).reshape(sz, ncoils * (
                                   kernel_size[0] * kernel_size[1]))
    else:
        kernel_out = torch.complex(torch.zeros(sampat.shape), torch.zeros(sampat.shape)).reshape(sz, ncoils * (kernel_size[0] * kernel_size[1]))
    kernel_out[:, idxA] = kernel.type(torch.complex64)
    return kernel_out

def ARC(cc, data, AtA, kernel_size, lam=0.01, verbose=True):
    # Data shape
    [sz, ncoil, sx, sy] = data.shape

    # Init empty array for interpolated k-space
    # kspace = torch.complex(torch.zeros(sz, 1, sx, sy), torch.zeros(sz, 1, sx, sy))
    kspace = torch.unsqueeze(data[:, cc, :, :], dim=1).type(torch.complex128)

    # Zero-pad the data in the [sx, sy] dimensions
    if verbose: print("Zero padding k-space data...")
    pad = nn.ZeroPad2d((int(np.floor(kernel_size[0] / 2)), int(np.floor(kernel_size[0] / 2)), int(np.floor(kernel_size[0] / 2)), int(np.floor(kernel_size[1] / 2))))
    data = pad(data)

    # Get index of center of patch for coil cc
    # NOTE: Currently, we assume that sub-sampling takes place only in the first PE direction.
    dummyPatch = np.zeros((ncoil, kernel_size[0], kernel_size[1])) # This patch is "the same" for all partitions
    dummyPatch[cc, int(np.floor(kernel_size[0] / 2)), int(np.floor(kernel_size[1] / 2))] = 1

    # Current center of patch (for coil cc)
    idxc = np.array(np.where(dummyPatch == 1))
    idxc_flat = np.ravel_multi_index(idxc, dummyPatch.shape).item()

    # Loop over each patch
    if verbose: print("Estimating missing k-space samples...")

    # Computation saving triuck - there is a finite number of sampling patterns. No need to re-calculate an existing one
    kerlist = [] # To store estimated kernels
    keylist = [] # To store sampling patterns

    # Go over all patches
    for ix in range(sx):
        #print("- DEBUG: coil {}, ix = {}/{}".format(cc, ix, sx))
        for iy in range(sy):
            #print("DEBUG: coil {}, iy = {}/{}".format(cc, iy, sy))

            # Current patch
            # patch = patches[:, ii, :]
            # patch_reshape = patch.reshape(sz, ncoil, kernel_size[0], kernel_size[1])
            patch = data[:, :, ix:ix + kernel_size[0], iy:iy + kernel_size[1]]

            # Get sampling pattern. Assumption: non-sampled locations have strictly zeros in them
            sampat = (torch.abs(patch) > 0.0) * 1.0

            # TEST
            # ------------
            #print("ix: {} iy: {}".format(ix, iy))
            #print("sampat: {}".format(sampat[0, 0, :, :]))
            # ------------

            if torch.sum(sampat) == 0 or sampat[0, idxc[0], idxc[1], idxc[2]].item() == 1:  # Here we assume that all partitions have the same sampling pattern (sub-sampling in only one PE direction)
                # Get original sampled k-space data point if we are on a sampled point, or a zero patch
                #kspace[:, 0, ix, iy] = torch.squeeze(patch[:, idxc[0], idxc[1], idxc[2]])
                continue
            else:
                key = sampat[0, ...].reshape(ncoil * kernel_size[0] * kernel_size[1]).detach().cpu().numpy() # Sampling pattern is assumed to be identical to all partitions
                if keylist:
                    tmpvals = [int(np.sum(np.abs(key - f))) for f in keylist]

                    try:
                        # See if the sampling pattern has already been used
                        kernel_index = tmpvals.index(0)
                    except:
                        # Sampling pattern not in the list, so it's a new one
                        kernel_index = -1
                else:
                    # Empty list
                    kernel_index = -1

                if kernel_index == -1:
                    # Estimate missing k-space data point
                    kernel = est_kernel(AtA, kernel_size, ncoil, cc, idxc, idxc_flat, sampat, lam=lam)

                    # Add computed kernel to the list
                    keylist.append(key)
                    kerlist.append(kernel)
                else:
                    # Use the already computed kernel
                    kernel = kerlist[kernel_index]

                # Estimate missing k-space sample
                kspace[:, 0, ix, iy] = torch.sum(torch.mul(kernel, patch.reshape(sz, ncoil * kernel_size[0] * kernel_size[1])), dim=1)

    # Output coil number for parallel processing recombination (no use in single CPU)
    return kspace, cc


