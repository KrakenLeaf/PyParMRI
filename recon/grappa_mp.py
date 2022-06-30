

'''
GeneRalized Autocalibrating Partial Parallel Acquisition (GRAPPA) implementation via PyTorch
                        - Multi CPU implementation -
'''

import torch
from torch import nn
import numpy as np
from tqdm import tqdm

# Multi-Processing related imports
from queue import PriorityQueue
import multiprocessing as mp
from multiprocessing import Process
from multiprocessing.managers import SyncManager
from multiprocessing import shared_memory

from recon.grappa import ARC, grappa_calibration_matrix

# Multi-Processing functions
# ----------------------------------------------------------------------------------------------------------------------
class MyManager(SyncManager):
    pass
MyManager.register("PriorityQueue", PriorityQueue)  # Register a shared PriorityQueue

# Synchronized queue manager
def Manager():
    m = MyManager()
    m.start()
    return m
m = Manager()
# ----------------------------------------------------------------------------------------------------------------------

def grappa_mp(data, acs, kernel_size=[5, 5], lam=0.01, verbose=True):
    # noinspection PyByteLiteral
    '''
                                -  Multi-processing implementation  -

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
    if verbose: print("GRAPPA multi-CPU implementation"); print(" ")

    # Allocate shared memory to all processes for the data
    #shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
    #data = np.ndarray(data, buffer=shm)

    # Determine GRAPPA kernel size
    if not isinstance(kernel_size, list):
        raise ValueError("Kernel dimensions must be a 2 items list only.")

    if len(kernel_size) != 2:
        raise ValueError("Kernel dimensions must be a 2 items list only.")

    # GRAPPA reconstruction
    # ---------------------------------------------------------------------------------
    # Permutation dimensions: # [ (Partitions, Coils, PE, Columns), (Columns, Coils, PE, Partitions) ]
    permute_dims = [(3, 1, 2, 0), (0, 1, 2, 3)]
    kspace = torch.complex(torch.zeros(size=data.shape), torch.zeros(size=data.shape))
    ncoil = acs.shape[1]
    for perdim in permute_dims:
        if verbose: print(" "); print("Orientation: {}".format(perdim)); print(" ")

        # Calculate weights
        if verbose: print("Calculating calibration matrix...")
        AtA = grappa_calibration_matrix(torch.permute(acs, perdim), kernel_size)

        # Loop over each coil
        # --------------------------------------------------------------------------------------------------------------
        # Multiprocessing init (for each permutation)
        n_threads = ncoil  # mp.cpu_count() - Each scan gets a new process/thread
        processes = [None] * n_threads
        queue = m.PriorityQueue()

        for cc in range(ncoil):
            if verbose: print(" "); print("Coil {}/{}".format(cc + 1, ncoil))

            # Calculate and apply weights in parallel
            p = mp.Process(target=ARC, args=(cc, torch.permute(data, perdim), AtA, kernel_size, lam))
            processes[cc] = p
            p.start()

        # Wait for the processes to finish
        for p in processes:
            p.join()

        # Concatenate processed data
        kspace_tmp = torch.complex(torch.zeros(size=data.shape), torch.zeros(size=data.shape))
        for tt in range(len(ncoil)):
            if verbose: print("Collecting interpolated k-space data from coil {}".format(tt))
            tmp, cct = queue.get()
            kspace_tmp[:, cct, :, :] = tmp
        # --------------------------------------------------------------------------------------------------------------

        # Average repeated results
        kspace += torch.permute(kspace_tmp, perdim)

        # Close shared memory pool
        #shm.close()
        #shm.unlink()
    return kspace / len(permute_dims)








