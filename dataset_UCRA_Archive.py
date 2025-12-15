"""
Functions and class to read and use the data from the UCR Time Series Classification Archive.
See here for more information : https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import numpy as np
import pandas as pd

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def read_tsv_file(path : str) :
    
    # Read file and convert to numpy
    tmp_df = pd.read_csv(filepath_or_buffer = path_tsv_file, delimiter = '\t', quotechar = '"')
    tmp_data = tmp_df.to_numpy()
    
    # Get data and labels
    data = tmp_data[:, 1:]
    labels = tmp_data[:, 0]

    return data, labels



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


if __name__ == "__main__":
    print("TEST")

    path_tsv_file = "./data/UCRArchive_2018/NonInvasiveFetalECGThorax1/NonInvasiveFetalECGThorax1_TRAIN.tsv"
    path_tsv_file = "./data/UCRArchive_2018/MixedShapesRegularTrain/MixedShapesRegularTrain_TRAIN.tsv"
    
    data, labels = read_tsv_file(path_tsv_file)
