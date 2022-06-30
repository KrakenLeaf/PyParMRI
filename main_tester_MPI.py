'''
A test file for development
'''

import os
import numpy as np
from data_io import read_twix
import torch
import nibabel as nib
import time
import nibabel as nib
from mpi4py import MPI
from recon.grappa import grappa
from utils.image_recon import image_sos

def get_coils_per_rank(rank, nproc=4, ncoils=32):
    coilinds = [f for f in range(ncoils)]
    n = int(ncoils / nproc) # Numbers should divide without reminder
    list_of_groups = [coilinds[i:i+n] for i in range(0, len(coilinds), n)]
    return list_of_groups[rank]

# Run
# ------------------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="GRAPPA reconstruction")

    parser.add_argument('--twix_file',
                        type=str,
                        help='Full path to TWIX .dat file',
                        default='path_to_some_default_twix_file.dat')

    parser.add_argument('--kernel_size',
                        type=int,
                        nargs="+",
                        help='List of kernel size (t2o elements)',
                        default=[5, 5])

    parser.add_argument('--nproc',
                        type=float,
                        help='Required number of workers - should be the same when running mpiexec -n N python main_tester_MPI.py --nproc N',
                        default=4)

    parser.add_argument('--lam',
                        type=float,
                        help='Tykhonov regularization parameter',
                        default=0.01)

    parser.add_argument('--ncoils',
                        type=float,
                        help='NUmber of acquisition coils',
                        default=32)

    parser.add_argument('--savefolder',
                        type=str,
                        help='Folder name for reconstructed k-space and image',
                        default='/Projects/PI/Recon1')

    args = parser.parse_args()

    # MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    coils = get_coils_per_rank(rank, nproc=args.nproc, ncoils=args.ncoils)
    print("Rank {}, coils {}".format(rank, coils))

    # Create save directory
    if not os.path.exists(args.savefolder) and rank == 0:
        os.mkdir(args.savefolder)
        print("Created save folder at {}".format(args.savefolder))

    # Create save names
    image_savename = os.path.join(args.savefolder, "image_grappa_recon_rank{}.nii.gz".format(rank))
    kspace_real_savename = os.path.join(args.savefolder, "kspace_grappa_recon_real_rank{}.nii.gz".format(rank))
    kspace_imag_savename = os.path.join(args.savefolder, "kspace_grappa_recon_imag_rank{}.nii.gz".format(rank))

    # Load T1 data
    tstart = time.time()
    t1_data, t1_acs, t1_hdr = read_twix(args.twix_file, modality='T1', gpu=False)

    # GRAPPA recon
    kspace = grappa(t1_data, t1_acs, kernel_size=args.kernel_size, lam=args.lam, coils=coils)

    # Recon image
    image_sos(kspace, filesavenae=image_savename)

    # Save k-space data
    kspace = torch.permute(kspace, (0, 2, 3, 1))  # Columns, PE, Partitions, Coil
    nib.Nifti1Image(torch.real(kspace).detach().cpu().numpy(), affine=None, header=None).to_filename(kspace_real_savename)
    nib.Nifti1Image(torch.imag(kspace).detach().cpu().numpy(), affine=None, header=None).to_filename(kspace_imag_savename)

    print("Total processing time = {:.2f} minutes".format((time.time() - tstart) / 60.0))
    print('Done')

