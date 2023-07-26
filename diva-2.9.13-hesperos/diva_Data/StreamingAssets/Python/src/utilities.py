# test
##
import pandas as pd
import numpy  as np
##
##
from   sklearn.model_selection import train_test_split
from   sklearn.ensemble        import RandomForestClassifier
from   sklearn.datasets        import make_classification
from   sklearn.metrics         import confusion_matrix
from   sklearn                 import preprocessing
##
##
import matplotlib.pyplot as plt
from numpy import loadtxt
##
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
##
############################################
############################################
############################################

def split_state_features(data):

	state    = data[[0]].copy()
	features = data.drop([0], axis = 1)
	return state, features
############################################
############################################
############################################
def split_state_features_with_header(data):

	liste_loc       = data.columns.tolist()
#	print(liste_loc[0])

	# if liste_loc[0] == "0":
	# 	state    = data.iloc[:,0].copy()
	# 	features = data.drop(data.columns[0],axis=1)
	# else:
	# 	state    = data["b'LABEL'"].copy()
	# 	features = data.drop("b'LABEL'", axis = 1)

	if liste_loc[0] == "b'LABEL'":
		state    = data["b'LABEL'"].copy()
		# remove labels column
		features = data.drop("b'LABEL'", axis = 1)
	else:
		state    = data.iloc[:,0].copy()
		# remove labels column
		features = data.drop(data.columns[0],axis = 1)

	return state, features

############################################
############################################
############################################
def get_features(data):

	# print(data)

	# remove header (first) row
	features = data.drop(data.index[0])

	# print("")

	# print(features)

	# if liste_loc[0] == "b'LABEL'":
	# 	state    = data["b'LABEL'"].copy()
	# 	# remove labels column
	# 	features = data.drop("b'LABEL'", axis = 1)
	# else:
	# 	state    = data.iloc[:,0].copy()
	# 	# remove labels column
	# 	features = data.drop(data.columns[0],axis = 1)

	return features


############################################
############################################
############################################
def dataframe_to_array(state, features):
	state2     = state.values
	features2  = features.values

	return state2, features2

############################################
############################################
############################################
def split_data_train_test(state, features, fraction = 0.25):

	features_train, features_test, feature_train, feature_test = train_test_split(features, state, test_size=fraction)
	return features_train, features_test, state_train, state_test

############################################
############################################
############################################
def extract_data(name_file):

	data            = pd.read_csv(name_file, header=None)
	state, features = split_state_features(data)
	state, features = dataframe_to_array(state, features)

	return state, features

############################################
############################################
############################################

def prepare_data_for_random_forest(name_file):


	fraction        = 0.25
	data            = pd.read_csv(name_file, header=None)
	state, features = split_state_features(data)
	state, features = dataframe_to_array(state, features)
	features_train, features_test, state_train, state_test = split_data_train_test(state, features, fraction)

	state           = np.ravel(state)
	state_train     = np.ravel(state_train)
	state_test      = np.ravel(state_test)

	return features_train, features_test, state_train, state_test, state, features

############################################
############################################
############################################
def prepare_data_for_random_forest_equalize_state(name_file):

	data            = pd.read_csv(name_file, header=None)
	data_0          = data[ data[0]==0]
	data_1          = data[ data[0]==1]

	nb_size_data    = data.shape
	nb_size_data_0  = data_0.shape
	nb_size_data_1  = data_1.shape

	nb_to_gen       = int(nb_size_data[0])
	nb_to_gen_2     = int(np.floor(nb_to_gen/2))
	nb_to_gen_0     = int(nb_size_data_0[0])
	nb_to_gen_1     = int(nb_size_data_1[0])

	II_0            = np.random.randint(nb_to_gen_0, size =nb_to_gen_2)
	II_1            = np.random.randint(nb_to_gen_1, size =nb_to_gen_2)

	data_0          = data_0.iloc[II_0]
	data_1          = data_1.iloc[II_1]

	data            = pd.concat([data_0,data_1])

	state, features = split_state_features(data)
	state, features = dataframe_to_array(state, features)
	#features_train, features_test, feature_train, feature_test = split_data_train_test(state, features, fraction)

	state           = np.ravel(state)

	return  state, features

############################################
############################################
############################################

def generate_confusion_matrix(state_true,state_predict):

	confusion  = 	confusion_matrix(state_true, state_predict)
	return confusion


############################################
############################################
############################################
def generate_confusion_matrix_and_plot(state_true,state_predict):

	confusion  = 	confusion_matrix(state_true, state_predict)
	return confusion
	plt.pcolor(confusion)
	plt.colorbar()
	plt.show()

############################################
############################################
############################################
def prepare_data_equalize_state(full_path_to_tagged_data):

	# if isinstance(full_path_to_tagged_data,str):
	# 	data            = pd.read_csv(full_path_to_tagged_data)
	# else:
	data 			= full_path_to_tagged_data

#	liste_loc       = list(data.columns.values)

	# contains feature name for each column
	liste_loc       = data.columns.tolist()

	# header specified
	if (liste_loc[0] == "b'LABEL'"):
		data_0          = data[ data["b'LABEL'"]==0]
		data_1          = data[ data["b'LABEL'"]==1]
	# if no header
	else:
		data_0          = data[ data.iloc[:,0]==0]
		data_1          = data[ data.iloc[:,0]==1]

	# if liste_loc[0] == "0":
	# 	print("here")
	# 	data_0          = data[ data.iloc[:,0]==0]
	# 	data_1          = data[ data.iloc[:,0]==1]
	# else:
	# 	print("not here")
	# 	data_0          = data[ data["b'LABEL'"]==0]
	# 	data_1          = data[ data["b'LABEL'"]==1]

	nb_size_data    = data.shape
	nb_size_data_0  = data_0.shape
	nb_size_data_1  = data_1.shape

	nb_to_gen       = int(nb_size_data[0])
	nb_to_gen_2     = int(np.floor(nb_to_gen/2))
	nb_to_gen_0     = int(nb_size_data_0[0])
	nb_to_gen_1     = int(nb_size_data_1[0])

	II_0            = np.random.randint(nb_to_gen_0, size =nb_to_gen_2)
	II_1            = np.random.randint(nb_to_gen_1, size =nb_to_gen_2)

	data_0          = data_0.iloc[II_0]
	data_1          = data_1.iloc[II_1]
	data            = pd.concat([data_0,data_1])

	state, features = split_state_features_with_header(data)
	state, features = dataframe_to_array(state, features)

	# print(state)
	# print(type(state[0]))
	#
	# print(features)
	# print(type(features[0,0]))

	state           = np.ravel(state)

	return  state, features

############################################
############################################
############################################
def prepare_data_raw_state(full_path_to_tagged_data):

	data            = pd.read_csv(full_path_to_tagged_data)
	state, features = split_state_features_with_header(data)
	state, features = dataframe_to_array(state, features)
	state           = np.ravel(state)

	return  state, features
###########################################
###########################################
###########################################
def prepare_data_raw_state_whitening(full_path_to_tagged_data):

	data                   = pd.read_csv(full_path_to_tagged_data)
	state, features        = split_state_features_with_header(data)
	state, features        = dataframe_to_array(state, features)
	state                  = np.ravel(state)
	scaler, features_white = features_whitening(features)


	return  state, features, scaler

###########################################
###########################################
###########################################
def prepare_data_raw_state_Dmatrix(full_path_to_tagged_data):

	data            = pd.read_csv(full_path_to_tagged_data)
	state, features = split_state_features_with_header(data)
	state, features = dataframe_to_array(state, features)
	state           = np.ravel(state)
	dtrain          = xgb.DMatrix(features, label=state)

	return  state, features, dtrain

############################################
############################################
############################################
def features_whitening(features):


	scaler         = preprocessing.StandardScaler(with_mean=True, with_std=True).fit(features)
	features_white = scaler.transform(features)

	return scaler, features_white
############################################
############################################
############################################
