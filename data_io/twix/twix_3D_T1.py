'''
Read T1 TWIX data from a 3D scan using pymapVBVD. These are preset functions with certain flags turned ON or OFF.
'''
import nibabel
import numpy as np
import os
import mapvbvd
#import spec2nii
#from spec2nii.twixfunctions import *
from data_io.twix.twixfunctions import *
import torch

def twix_3D_T1(filename, flagRemoveOS=True, flagIgnoreSeg=True, flagDoAverage=True, gpu=False, verbose=True):
    '''
    Preset function to read 3D T1 acquisition
    Fully expanded data has a typical order of:

                0. Columns
                1. Channels/Coils
                2. Lines
                3. Partitions
                4. Slices
                5. Averages
                6. (Cardiac-) Phases
                7. Contrasts/Echoes
                8. Measurements
                9. Sets
                10. Segments
                11. Ida
                12. Idb
                13. Idc
                14. Idd
                15. Ide
    
    After squeezing the data, for a 3D T1 (non-cardiac) scan (no slices, and usually NEX=1) it would mostly look like:
    
                0. Columns
                1. Channels/Coils
                2. Lines
                3. Partitions
    
    :param filename: Full path to twix (.dat) file to be read.
           flagRemoveOS: Remove readout direction over-sampling (typically by a factor of 2)
           flagIgnoreSeg: Combine all shots together for a multi-shot acquisition 
           flagDoAverage: Average multiple NEX (number of excitations)
           gpu: if [True] - put data on GPU, else CPU
           verbose: If [True], print text messages
    :return: data: (sub) sampled k-space data [PyTorch tensor]
             acs: (optional) k-space calibration data [PyTorch tensor]
             hdr: header dictionary
    '''
    if verbose: print("3D T1 file: {}".format(filename)); print(" ")

    # Initial read
    twixObj = mapvbvd.mapVBVD(filename)
    if verbose: print("{}".format(twixObj))

    # TODO: Place correct nifti header
    hdr_nifti = None

    # Image: Remove over-sampling, combine shots and average multiple NEX
    twixObj.image.flagRemoveOS = flagRemoveOS
    twixObj.image.flagIgnoreSeg = flagIgnoreSeg
    twixObj.image.flagDoAverage = flagDoAverage

    # Retrieve data and ACS (i exists)
    if verbose: print("Beginning image data read:")
    data = torch.tensor(np.squeeze(twixObj.image['']))
    hdr = twixObj.hdr
    
    if gpu:
        data = data.cuda()
    
    # ACS: Remove over-sampling, combine shots and average multiple NEX
    try:
        twixObj.refscan.flagRemoveOS = flagRemoveOS
        twixObj.refscan.flagIgnoreSeg = flagIgnoreSeg
        twixObj.refscan.flagDoAverage = flagDoAverage

        # Retrieve ACS
        if verbose: print("Beginning ACS read:")
        acs = torch.tensor(np.squeeze(twixObj.refscan['']))

        if gpu:
            acs = acs.cuda()
    except:
        if verbose: print("No ACS data was found.")
        acs = []

    # After squeezing, dimensions should be: 
    #           # columns (readout), # channels, # Phase Encode (PE) lines 1, # Phase Encode (PE) lines 2 (Partitions)
    return data, acs, hdr_nifti








