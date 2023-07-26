import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
from   utilities import *
import warnings
import pickle
import argparse
from   os.path import abspath
from classifier_utilities import *
from sklearn.externals import joblib

################################################################################
################################################################################
################################################################################








################################################################################
################################################################################
################################################################################

if __name__ == "__main__":

	warnings.filterwarnings('ignore')
	print("load classifier \n")

	parser = argparse.ArgumentParser()

	parser.add_argument('--filename', type=str, help="path to generated training data m-file")
    parser.add_argument('--weights_name', type=str, help="path to save model weights as hdf5-file")
    parser.add_argument('--meanstd_name', type=str, help="path to save normalization factors as m-file")

    # parse the input arguments
    args = parser.parse_args()


path_to_data = "C:/Users/Mohamed/Desktop/Feature Testing/raw-data-tests/first/"
name_test    = "image-features.csv"
name_train   = "tag-features.csv"
name_tag     = "tag-features.csv"

	exit()
