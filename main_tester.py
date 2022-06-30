'''
A test file for development
'''

import os
import numpy as np
from data_io import read_twix
import torch
import time
import nibabel as nib

torch.set_printoptions(precision=2, threshold=100, edgeitems=20)
np.set_printoptions(precision=2, threshold=100, edgeitems=10)
np.set_printoptions(threshold=np.inf)

# Run
# ------------------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="GRAPPA reconstruction")

    parser.add_argument('--twix_file',
                        type=str,
                        help='Full path to TWIX .dat file',
                        default='path_to_some_default_twix_file.dat')

    parser.add_argument('--coils',
                        type=int,
                        nargs="+",
                        help='List of coil numbers to explicitly reconstruct (empty list means reconstruct all coils)',
                        default=[])

    parser.add_argument('--kernel_size',
                        type=int,
                        nargs="+",
                        help='List of kernel size (t2o elements)',
                        default=[5, 5])

    parser.add_argument('--lam',
                        type=float,
                        help='Tykhonov regularization parameter',
                        default=0.01)

    parser.add_argument('--savefolder',
                        type=str,
                        help='Folder name for reconstructed k-space and image',
                        default='/Projects/PI/Recon1')

    parser.add_argument('--do_mp',
                        type=bool,
                        help='[True] - do multi-CPU processing. Overrides --gpu',
                        default=False)

    parser.add_argument('--gpu',
                        type=bool,
                        help='[True] - do GPU processing',
                        default=False)

    args = parser.parse_args()

    # Imports
    gpu = args.gpu # [True] - process on the GPU
    if args.do_mp:
        # Multi-CPU reconstruction
        from recon.grappa_mp import grappa_mp as grappa
        gpu = False # If do_mp, override GPU processing
    else:
        # Single CPU reconstruction
        from recon.grappa import grappa
    from utils.image_recon import image_sos

    # Create save directory
    if not os.path.exists(args.savefolder):
        os.mkdir(args.savefolder)
        print("Created save folder at {}".format(args.savefolder))

    # Create save names
    if not args.coils:
        coilsname = " "
        image_savename = os.path.join(args.savefolder, "image_grappa_recon.nii.gz")
        kspace_real_savename = os.path.join(args.savefolder, "kspace_grappa_recon_real.nii.gz")
        kspace_imag_savename = os.path.join(args.savefolder, "kspace_grappa_recon_imag.nii.gz")
    else:
        coilsname = "".join([str(f) for f in args.coils])
        image_savename = os.path.join(args.savefolder, "image_grappa_recon_coils_{}.nii.gz".format(coilsname))
        kspace_real_savename = os.path.join(args.savefolder, "kspace_grappa_recon_real_coils_{}.nii.gz".format(coilsname))
        kspace_imag_savename = os.path.join(args.savefolder, "kspace_grappa_recon_imag_coils_{}.nii.gz".format(coilsname))

    # Load T1 data
    tstart = time.time()
    t1_data, t1_acs, t1_hdr = read_twix(args.twix_file, modality='T1', gpu=gpu)

    # GRAPPA recon
    kspace = grappa(t1_data, t1_acs, kernel_size=args.kernel_size, lam=args.lam, coils=args.coils, gpu=gpu)

    # Recon image
    image_sos(kspace.detach().cpu(), hdr=t1_hdr, filesavename=image_savename)

    # Save k-space data
    kspace = torch.permute(kspace, (0, 2, 3, 1))  # Columns, PE, Partitions, Coil
    nib.Nifti1Image(torch.real(kspace).detach().cpu().numpy(), affine=None, header=None).to_filename(kspace_real_savename)
    nib.Nifti1Image(torch.imag(kspace).detach().cpu().numpy(), affine=None, header=None).to_filename(kspace_imag_savename)

    print("Total processing time = {:.2f} minutes".format((time.time() - tstart) / 60.0))
    print('Done')

