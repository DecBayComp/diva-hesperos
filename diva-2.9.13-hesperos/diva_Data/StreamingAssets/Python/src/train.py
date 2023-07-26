import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
try:
	from   .classifier_utilities import *
except ImportError:
	from   classifier_utilities import *
try:
	from   .utilities import *
except ImportError:
	from   utilities import *

from   sklearn.model_selection import train_test_split
import warnings
import pickle
import argparse
from   os.path import abspath
import optuna


#from sklearn.externals import joblib

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
class combining_classifiers:

	# search for existing learner ; create new one if doesn't exist
	def __init__(self, full_path_or_features, classifier_path, classification_strength):
		#if isinstance(full_path_or_features, str):
		self.full_path = full_path_or_features
		self.classifier_path = classifier_path
		self.classification_strength = classification_strength
		# else:
		# 	self.state = None
		# 	self.features = full_path_or_features
		## look for existing set in learning
		try:
			liste_learner      = pickle.load(open(self.classifier_path, 'rb'))#pickle.load(open("liste_classifier.pckl", 'rb'))
			self.liste_learner = liste_learner
		except FileNotFoundError:
			self.liste_learner = []

	def __prepare_data_for_training__(self):
		# separate states (i.e. label tags -- 0 or 1) from features
		state, features                  = prepare_data_equalize_state(self.full_path)
		train_x, test_x, train_y, test_y = train_test_split(features, state, test_size=0.20)
		self.state                       = state
		self.features                    = features
		self.train_x                     = train_x
		self.train_y                     = train_y
		self.test_x                      = test_x
		self.test_y                      = test_y
		if not self.liste_learner:
			pass
		else:
			self.__recreate_all_features_from_previous_learners__()

		# print(self.train_x)
		# print(type(self.train_x))
		# print(type(self.train_x[0,0]))

	def __prepare_data_for_inference__(self):
		if not self.liste_learner:
			raise FileNotFoundError("Classifier file not found")

		self.features = self.full_path

		# print(self.features)

		# get features from self.full_path
		# self.features = get_features(self.full_path)

	def __apply_one_classifier_test__(self):
		 classifier_, score  = create_the_random_forest_classifier(self.state, self.features)
		 self.classifier_    = classifier_

	def __apply_multiple_classifier_to_features__(self):

		classifier_1, score      = create_the_random_forest_classifier(self.state, self.features)
		# classifier_2, score      = create_adaboost_classifier_tree(self.state, self.features)
		# classifier_3, score      = create_adaboost_classifier_SGD(self.state, self.features)
		# classifier_4, score      = create_Nearest_Neighbors_classifier(self.state, self.features)

		# self.set_classifier      = {'c1': classifier_1, 'c2':classifier_2, 'c3':classifier_3, 'c4':classifier_4}
		self.set_classifier      = {'c1': classifier_1, 'c2':classifier_1}
		self.features            = create_new_features_from_weak_classifiers(self.set_classifier, self.features)

	def __apply_main_classifier__(self):

		classifier_main, score         = create_xgboost_classifier(self.state, self.features)
		state_out, log_proba           = perform_inference_xgboost(classifier_main, self.features)
		# print(log_proba)
		self.set_classifier['main']    = classifier_main


		if not self.liste_learner:
			self.liste_learner             = [self.set_classifier]
		else:
			self.liste_learner.append(self.set_classifier)

	def __predict_log_proba__(self):
		self.__recreate_all_features_from_previous_learners__()
		# for i in range(len(self.log_proba)):
		# 	print(self.log_proba[i],str(i))
		return self.log_proba
		# return self.features[:,-1]

	def __predict_proba__(self):
		self.__recreate_all_features_from_previous_learners__()
		return np.exp(self.log_proba)
		# return np.exp(self.features[:,-1])

	def __recreate_all_features_from_previous_learners__(self):
		liste_learner = self.liste_learner
		features      = self.features
		for dict_classif_loc in liste_learner:
			# log_proba,_,features =  perform_inference_all_classifiers(dict_classif_loc, features)
			log_proba,_,features =  perform_inference_all_classifiers(dict_classif_loc, features)
		self.features  = features
		self.log_proba = log_proba

	def __generate_features__(self):
		#prob_loc_out      = self.classifier_.predict_proba(self.features)
		junk, log_proba_out  = perform_inference_RF_sklearn(self.classifier_ , self.features)

		#self.log_proba_out   = log_proba_out[:,1]

	def __fuse_new_features__(self, features_loc):
		self.features = np.concatenate((self.features, features_loc), axis=1)
		#self.features = np.column_stack(self.features, self.log_proba_out  )

	def __apply_full_analysis__(self):
		1



################################################################################
################################################################################
################################################################################

#def perform_training_one(full_path ):


#	state, features        = prepare_data_equalize_state(full_path)
#	classifier_, score     = create_the_random_forest_classifier(state, features)

#	return classifier_, score

################################################################################
################################################################################
################################################################################

# if __name__ == "__main__":
#
# 	warnings.filterwarnings('ignore')
#
#     # parse the input arguments
# 	parser = argparse.ArgumentParser()
# 	parser.add_argument('--full_path', type=str, help="full path to the training file")
# 	args = parser.parse_args()
#
# 	#
# 	glob_class = combining_classifiers(abspath(args.full_path))
# 	glob_class.__prepare_data__()
# 	glob_class.__apply_multiple_classifier_to_features__()
# 	glob_class.__apply_main_classifier__()
# 	#
#
# 	print(np.size(glob_class.liste_learner))
# 	print((glob_class.features.shape))
#
# 	filename = "liste_classifier.pckl"
# 	pickle.dump(glob_class.liste_learner, open(filename, 'wb'))
#
#
# 	exit()
