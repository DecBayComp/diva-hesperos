import numpy as np
import xgboost as xgb
from sklearn import datasets
from sklearn.ensemble        import RandomForestClassifier

#from sklearn.cross_validation import train_test_split

iris = datasets.load_digits(n_class=2)

X = iris.data
y = iris.target

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
n_train = int(X.shape[0] * .8)
X_train = X[:n_train]
y_train = y[:n_train]
X_test = X[n_train:]

print(y_train)



nb_tree      = 50
max_depth    = 50

criterion    = 'gini'
max_features = "sqrt"

oob_score    = True
warm_start   = False
bootstrap    = True
class_weight = None


classifier1 = RandomForestClassifier(n_estimators=nb_tree, max_depth=max_depth ,  criterion  = criterion,
	max_features = max_features , oob_score = oob_score, warm_start = warm_start , bootstrap =bootstrap ,class_weight=class_weight  )

classifier2 = RandomForestClassifier(n_estimators=nb_tree, max_depth=max_depth ,  criterion  = criterion,
	max_features = max_features , oob_score = oob_score, warm_start = warm_start , bootstrap =bootstrap ,class_weight=class_weight  )

classifier1.fit(X_train, y_train)
classifier2.fit(X_train, y_train)

log_proba = classifier1.predict_log_proba(X_train)[:,1]

extra = np.stack((log_proba, log_proba), axis=1)
print(extra.shape)
features = np.hstack((X_train,extra))
print(features.shape)

classifier_main = xgb.XGBClassifier(booster='dart', max_depth=5,
							learning_rate=0.1, objective='binary:logistic',
							sampling_type='uniform', normalize_type='tree'
							)

classifier_main.fit(features, y_train)


log_proba = classifier1.predict_log_proba(X_test)[:,1]

features = np.hstack((X_test,np.stack((log_proba, log_proba), axis=1)))

print(classifier_main.predict_proba(features))
