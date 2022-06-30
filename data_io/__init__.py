'''
Read Siemens TWIX data according to predefined scan setups.
Currently supported setups are:

        1. T1 - 3D acquisition
                Single average
                Combined shots
                No oversampling in readout direction



'''
def read_twix(name, modality='t1', gpu=False):
    # Get relevant reader function
    if modality.lower() == 't1':
        from data_io.twix.twix_3D_T1 import twix_3D_T1

    # Execute and return values
    return {
            't1': twix_3D_T1(name, gpu=gpu),
           }[modality.lower()]