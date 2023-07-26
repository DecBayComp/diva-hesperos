#TabNine::version
#TabNine::sem

#### Here we store all the utilisies linked to te classifier
##TabNine::sem

## stuff
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
## process
from sklearn.model_selection import train_test_split
from sklearn.datasets        import make_classification
from sklearn.metrics         import confusion_matrix, accuracy_score
## ensemble methods
from sklearn.ensemble        import RandomForestClassifier
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble 	     import AdaBoostClassifier
## gaussian process
from sklearn.gaussian_process        import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
##
import xgboost as xgb
from   xgboost         import XGBClassifier
from   xgboost.sklearn import XGBRegressor
##
from sklearn.neighbors import KNeighborsClassifier
##
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
##
##
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

############################################
############################################
############################################
def perform_inference_general(set_classifier, features, nature):
	#print(nature)
	#print(type(nature))
	classifier    = set_classifier
	try: 
		log_proba = classifier.predict_log_proba(features)
		state_out = classifier.predict(features)
		log_proba[np.where(np.isinf(log_proba))] = -100
	except AttributeError:
		state_out = classifier.predict(features)
		log_proba = classifier.predict_proba(features)
		log_proba = np.log(log_proba)
		log_proba[np.where(np.isinf(log_proba))] = -100		

	#if nature[0] == "c":
#		state_out = classifier.predict(features)
#		log_proba = classifier.predict_log_proba(features)
#		log_proba[np.where(np.isinf(log_proba))] = -100
#	else:
#		state_out = classifier.predict(features)
#		log_proba = classifier.predict_proba(features)
#		log_proba = np.log(log_proba)
#		log_proba[np.where(np.isinf(log_proba))] = -100

	return state_out, log_proba 

############################################
############################################
############################################

def perform_inference_RF_sklearn(classifier, features):


	state_out = classifier.predict(features)
	log_proba = classifier.predict_log_proba(features)
	log_proba[np.where(np.isinf(log_proba))] = -100


	return state_out, log_proba 

################################################################################
################################################################################
################################################################################
def perform_inference_ada_tree(classifier, features):


	state_out = classifier.predict(features)
	log_proba = classifier.predict_log_proba(features)
	log_proba[np.where(np.isinf(log_proba))] = -100


	return state_out, log_proba 
################################################################################
################################################################################
################################################################################
def perform_inference_ada_SGD(classifier, features):


	state_out = classifier.predict(features)
	log_proba = classifier.predict_log_proba(features)
	log_proba[np.where(np.isinf(log_proba))] = -100


	return state_out, log_proba 

################################################################################
################################################################################
################################################################################
def perform_inference_xgboost(classifier, features):

	state_out = classifier.predict(features)
	#log_proba = classifier.predict_log_proba(features)
	log_proba = classifier.predict_proba(features)
	log_proba = np.log(log_proba)
	log_proba[np.where(np.isinf(log_proba))] = -100

	return state_out, log_proba


################################################################################
################################################################################
################################################################################
def perform_inference_all_classifiers(set_classifier, features):
	
	#print(features.shape)
	try:
		del log_proba_out
	except NameError:
		1

	for k in set_classifier.keys():
		if k[0] == "c":
			state_out, log_proba = perform_inference_general(set_classifier[k], features, k)
			try:
				log_proba_out = np.column_stack((log_proba_out ,log_proba[:,1]))
			except NameError:
				# Do something
				log_proba_out = log_proba[:,1]

	features 			 = np.column_stack((features,log_proba_out))
	#print(features.shape)
	k        			 = "main"
	state_out, log_proba = perform_inference_general(set_classifier[k], features, k)
	features             = np.column_stack((features,log_proba_out[:,1]))
	#print(features.shape)
	
	

	return  state_out,features

################################################################################
################################################################################
################################################################################
def create_new_features_from_weak_classifiers(set_classifier, features):

	#print(features.shape)
	try:
		del log_proba_out
	except NameError:
		1

	for k in set_classifier.keys():
		if k[0] == "c":
			state_out, log_proba = perform_inference_general(set_classifier[k], features, k)
			try:
				log_proba_out = np.column_stack((log_proba_out ,log_proba[:,1]))
			except NameError:
				# Do something
				log_proba_out = log_proba[:,1]

	features 			 = np.column_stack((features,log_proba_out))
	print(features.shape)
		

	return features
	
################################################################################
################################################################################
################################################################################


def create_the_random_forest_classifier(state, features):
	# we will be very contraining in our parameters to promote robustness 
	# parameters may be adapted to 

	nb_tree      = 50
	max_depth    = 50
	
	criterion    = 'gini'
	max_features = "sqrt"

	oob_score    = True
	warm_start   = False
	bootstrap    = True
	class_weight = None


	classifier = RandomForestClassifier(n_estimators=nb_tree, max_depth=max_depth ,  criterion  = criterion,
		max_features = max_features , oob_score = oob_score, warm_start = warm_start , bootstrap =bootstrap ,class_weight=class_weight  )

	classifier.fit(features, state)
	score   = classifier.score(features, state)

	return classifier, score 

################################################################################
################################################################################
################################################################################
def create_the_gaussian_process_classifier(state, features):
	"""
	gaussian process classifier with sklearn implementation 
	not supposed to learn too much
	----------------------
	"""


	kernel               = 1.0 * RBF(1.0)
	n_restarts_optimizer = 10


	gpc = GaussianProcessClassifier(kernel=kernel,random_state=0).fit(features, state)






	return gpc

################################################################################
################################################################################
################################################################################
def create_Nearest_Neighbors_classifier(state, features):
	"""
	nearest neighbor classifier
	very simple but usefull for direct application
	---------------------------------
	"""
	n_neighbors = 3
	weigths     = 'uniform'
	algorithm   = 'kd_tree'
	leaf_size   = 25
	metric      = 'minkowski'
	p           = 1

	classifier = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weigths, algorithm=algorithm, leaf_size=leaf_size , p=p, metric=metric)
	classifier.fit(features, state) 

	score           = classifier.score(features, state)

	return classifier, score 


################################################################################
################################################################################
################################################################################
def create_adaboost_classifier_tree(state,features):
	"""
	adaboost classifier with tree based 
	very simple but usefull for direct application
	---------------------------------
	"""

	n_estimators    = 25
	learning_rate   = 0.1 # slower than usual
	nb_trees        = 10; 
	weak_classifier = DecisionTreeClassifier(max_depth=nb_trees, min_samples_split=2)
	algorithm       ='SAMME.R'

	classifier      = AdaBoostClassifier( base_estimator=weak_classifier, n_estimators= n_estimators ,
                         learning_rate= learning_rate , algorithm = algorithm)

	classifier.fit(features, state)
	score           = classifier.score(features, state)

	return classifier, score 
################################################################################
################################################################################
################################################################################
def create_adaboost_classifier_SVM(state, features):
	"""
	adaboost classifier with tree based 
	very simple but usefull for direct application
	---------------------------------
	"""
	n_estimators    = 25  #
	learning_rate   = 0.1 # slower than usual
	nb_tress        = 10; # 
	weak_classifier = SVC(probability = True, kernel = 'linear',gamma='auto')
	algorithm       ='SAMME.R'


	classifier   = AdaBoostClassifier( base_estimator=weak_classifier, n_estimators= n_estimators ,
                         learning_rate= learning_rate , algorithm = algorithm)

	classifier.fit(features, state)
	score           = classifier.score(features, state)

	return classifier, score 


################################################################################
################################################################################
################################################################################
def create_adaboost_classifier_SGD(state, features):
	"""
	adaboost classifier with tree based 
	very simple but usefull for direct application
	---------------------------------
	"""
	n_estimators    = 25  #
	learning_rate   = 0.1 # slower than usual
	l1_ratio        = 0.1 
	weak_classifier = SGDClassifier(loss = 'log', penalty = 'elasticnet',max_iter=1000,
	 tol=0.001 , shuffle = True, learning_rate= 'optimal', eta0=0.0)
	algorithm       ='SAMME.R'


	classifier   = AdaBoostClassifier( base_estimator=weak_classifier, n_estimators= n_estimators ,
                         learning_rate= learning_rate , algorithm = algorithm)

	classifier.fit(features, state)
	score           = classifier.score(features, state)

	return classifier, score 
################################################################################
################################################################################
################################################################################
def create_xgboost_classifier(state, features ):
	"""
	xgboost classifier 
	very simple but usefull for direct application
	---
	"""

	

	booster = ''




	classifier = XGBClassifier(booster='dart', max_depth=5,
								learning_rate=0.1, objective='binary:logistic',
								sampling_type='uniform', normalize_type='tree'
								)


	classifier.fit(features, state)
	score = 0


	return classifier, score


	#param = {'booster': 'dart',
    #     'max_depth': 5, 'learning_rate': 0.1,
    #     'objective': 'binary:logistic',
    #     'sample_type': 'uniform',
    #     'normalize_type': 'tree',
    #     'rate_drop': 0.1,
    #     'skip_drop': 0.5}




	#num_round  = 100
	#classifier = xgb.train(param, dtrain, num_round)


	#score      = classifier.score(features, state)
	#score = 0


	#return classifier, score 

#time

	#model = xgb.XGBClassifier(max_depth=12,
    #                    subsample=0.33,
    #                    objective='binary:logistic',
    #                    n_estimators=300,
    #                    learning_rate = 0.01)
	#eval_set = [(train_X, train_Y), (test_X, test_Y)]
	#model.fit(train_X, train_Y.values.ravel(), early_stopping_rounds=15,
 	#eval_metric=["error", "logloss"], eval_set=eval_set, verbose=True)





################################################################################
################################################################################
################################################################################




################################################################################
################################################################################
################################################################################










################################################################################
################################################################################
################################################################################

