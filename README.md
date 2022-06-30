# PyParMRI
Collection of magnetic resonance imaging (MRI) reconstruction tools written in Python with PyTorch.

Currently implemented methods:
------------------------------
1. GeneRalized Autocalibrating Partial Parallel Acquisition (GRAPPA) [1]. Implementation follows similar lines to the description provided in [2].

        [1] Griswold, Mark A., et al. "Generalized autocalibrating partially parallel acquisitions (GRAPPA)."
        Magnetic Resonance in Medicine, 47.6 (2002): 1202-1210.
        
        [2] Uecker, Martin, et al. "ESPIRiT—an eigenvalue approach to autocalibrating parallel MRI: where SENSE meets GRAPPA."
        Magnetic resonance in medicine 71.3 (2014): 990-1001.
        
2. Sum-Of-Squares (SOS) image - standard image combination techniques for parallel images (PI) MRI.


Notes about the implementations:
--------------------------------
1. Implementation was written with PyTorch - should be differentiable, but this has not been tested.
2. GRAPPA and SOS image reconstructions are GPU supported - this can be very useful when reconstructing from large k-space raw data files.
3. Two use code examples are provided:
  a. main_tester.py - Can be run on GPU, or through multi-CPU processing (this feature is not fully tested).
  b. main_tester_MPI.py - Can be run with MPI, but this is not fully tested.
4. Using the Python implementation of mapVBVD, data_io folder provides some simple scripts to read Siemens TWIX data files. However, the rsulting images do not contain the proper header.
5. GRAPPA recosntruction code supports multi-coil (multi-channel) 3D acquisitions (2D multi-slice acquisitions should also be supported, but this has not been tested yet). Data format should follow (4D tensor):
           # Columns (Freq. encode), # Channels (Coils), # PE Lines, # Partitions. 
   This should be automatically fullfiled if you read from a TWIX .dat file.
   GRAPPA kernels are 2D, but processing is batch performed for all frequency encode columns. 

Requirements:
-------------
1. PyTorch 1.11.0
2. Numpy 1.22.1
3. pyMapVBVD 0.4.8 - https://pypi.org/project/pyMapVBVD/
4. mpi4py 3.1.3 - https://pypi.org/project/mpi4py/ (if you want to test multi-CPU processing)



