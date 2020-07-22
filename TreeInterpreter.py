# %%
# Packages, libraries, data, etc.
import numpy as np
from treeinterpreter import treeinterpreter as ti
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Models
rf_clf = RandomForestClassifier(n_estimators=1666, max_features="auto", min_samples_split=2, min_samples_leaf=2,
                                max_depth=20, bootstrap=True, n_jobs=1)

# load and organize Wisconsin Breast Cancer Data 
data = load_breast_cancer()
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']

# Random split data
X_tng, X_val, y_tng, y_val = train_test_split(features, labels, test_size=0.33, random_state=42)

print(X_tng.shape) # (381, 30)
print(X_val.shape) # (188, 30)

# Run the Random Forest Classifier
rf_clf.fit(X_tng, y_tng)
y_hat = rf_clf.predict(X_val)
print("Random Forest Classifier Binary AUC is ", roc_auc_score(y_val, y_hat))
# Random Forest Classifier Binary AUC is  0.9577525595164673

# %%
# Tree interpreter usage
prediction, bias, contributions = ti.predict(rf_clf, X_val)
assert(np.allclose(prediction, bias + np.sum(contributions, axis=1)))

# Scant examples of applying this information.