'''
Image reconstruction utility functions
'''

import os
import torch
import nibabel as nib
import numpy as np

def image_sos(kspace, hdr=None, filesavename=[]):
    '''
    Reconstruct a Sum Of Squares (SOS) image from k-space data
    :param kspace: k-space data tensor. Currently, supported dimensions are of a 4D tensor:
                # columns (readout), # channels, # Phase Encode (PE) lines 1, # Phase Encode (PE) lines 2 (Partitions)
    :param filesavename: Optional path to save the reconstructed image as a NIfTI file (default is empty list)
    :return: Reconstructed SOS image
    '''
    from torch.fft import fft
    from torch.fft import fftshift
    from torch.fft import ifftshift
    
    # Transform to image domain
    dims = [0, 2, 3]
    image = kspace
    for dim in dims:
        image = ifftshift(fft(fftshift(image, dim=dim), dim=dim), dim=dim)

    # Combine sum of squares
    image = torch.squeeze(torch.sum(torch.square(torch.abs(image)), dim=1))

    if isinstance(filesavename, str):
        # TODO: Save with correct header
        nib.Nifti1Image(image.detach().cpu().numpy(), affine=None, header=hdr).to_filename(filesavename)

    # Dimension are of a 3D Cartesian image
    return image

# Fourier based aggregation image
def image_fba(kspace, p=1, hdr=None, filesavename=[]):
    '''
    Fourier based aggregation image reconstruction. Aggregation is performed on the coil (channel) dimension,
    assumed to be 1
    :param kspace: k-space data tensor. Currently, supported dimensions are of a 4D tensor:
                # columns (readout), # channels, # Phase Encode (PE) lines 1, # Phase Encode (PE) lines 2 (Partitions)
    :param p: Weights power
    :param filesavename: Optional path to save the reconstructed image as a NIfTI file (default is empty list)
    :return: Reconstructed FBA image
    '''
    from torch.fft import fft
    from torch.fft import fftshift
    from torch.fft import ifftshift

    # Permute such that coils dimension is first
    print("kspace dims {}".format(kspace.shape))
    kspace = torch.permute(kspace, dims=(1, 0, 2, 3))
    print("kspace dims {}".format(kspace.shape))

    kspace_p = torch.pow(torch.abs(kspace), exponent=p)
    fsum = torch.sum(kspace_p, dim=0)

    # Fourier weights
    w = torch.div(kspace_p, fsum)
    print("W dims {}".format(w.shape))

    # Aggregation
    image = torch.sum(torch.mul(w, kspace), dim=0)
    print("image dims {}".format(image.shape))

    # Reconstruct image
    # Transform to image domain
    dims = [0, 1, 2]
    for dim in dims:
        image = ifftshift(fft(fftshift(image, dim=dim), dim=dim), dim=dim)
    image = torch.abs(image) # Remove residual imaginary parts

    if isinstance(filesavename, str):
        # TODO: Save with correct header
        nib.Nifti1Image(image.detach().cpu().numpy(), affine=None, header=hdr).to_filename(filesavename)

    # Dimension are of a 3D Cartesian image
    return image

# Recon image from chunks in folder
# ------------------------------------------------------------------------------------
if __name__ == '__main__':
    print(' done ')

