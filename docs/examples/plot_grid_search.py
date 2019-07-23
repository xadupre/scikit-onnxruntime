# Licensed under the MIT License.

"""
.. _l-example-grid-search:

Grid search ONNX models
=======================

This example uses *OnnxTransformer* to freeze a model. We first fit a few
preprocessing models, convert them into an *ONNX* model, and then use them as
the parameter to the ``OnnxTransformer``. As a result, they are not fit again
during the ``pipeline.fit``, and are only used as a frozen model.
The pipeline is then put into a ``GridSearchCV``, and the frozen models are
the hyperparameters.

.. contents::
    :local:

Fit all preprocessings and serialize with ONNX
++++++++++++++++++++++++++++++++++++++++++++++
"""

from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# slk2onnx is used to convert a sklearn model into ONNX format
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn

# skonnxrt provides OnnxTransformer which serves the model converted to ONNX
from skonnxrt.sklapi import OnnxTransformer

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

dec_models = [
    PCA(n_components=1),
    PCA(n_components=2),
    StandardScaler(),
]

onx_bytes = []

for model in dec_models:
    model.fit(X_train)
    onx = convert_sklearn(model,
                          initial_types=[('X',
                                          FloatTensorType((1, X.shape[1])))])
    onx_bytes.append(onx.SerializeToString())

##############################
# Pipeline with OnnxTransformer
# +++++++++++++++++++++++++++++++

pipe = make_pipeline(OnnxTransformer(onx_bytes[0]),
                     LogisticRegression(multi_class='ovr'))

################################
# Grid Search
# +++++++++++
#
# The serialized models are now used as a parameter
# in the grid search.

param_grid = [{'onnxtransformer__onnx_bytes': onx_bytes,
               'logisticregression__penalty': ['l2', 'l1'],
               'logisticregression__solver': ['liblinear', 'saga']
               }]

clf = GridSearchCV(pipe, param_grid, cv=5)
clf.fit(X_train, y_train)

y_true, y_pred = y_test, clf.predict(X_test)
cl = classification_report(y_true, y_pred)
print(cl)

#####################################
# Best preprocessing?
# +++++++++++++++++++
#
# We get the best parameters returned by the grid search
# and we search for it in the list of serialized
# preprocessing models.
# And the winner is...

bp = clf.best_params_
best_step = onx_bytes.index(bp["onnxtransformer__onnx_bytes"])
print(dec_models[best_step])
