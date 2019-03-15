# Licensed under the MIT License.

"""
.. _l-example-intermediate-outputs:

Investigate intermediate outupts
================================


.. contents::
    :local:

Train a model
+++++++++++++

A very basic example using
`TfidfVectorizer <https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html>`_
on a dummy example.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


corpus = np.array([
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        ' ',
        ]).reshape((4, 1))
vect = TfidfVectorizer(ngram_range=(1, 2), norm=None)
vect.fit(corpus.ravel())
pred = vect.transform(corpus.ravel())

###########################
# Convert a model into ONNX
# +++++++++++++++++++++++++

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType

model_onnx = convert_sklearn(vect, 'TfidfVectorizer',
                             [('input', StringTensorType([1, 1]))])

with open("TfidfVectorizer.onnx", "wb") as f:
    f.write(model_onnx.SerializeToString())
    
###########################
# Visualize
# +++++++++

from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer
pydot_graph = GetPydotGraph(model_onnx.graph, name=model_onnx.graph.name, rankdir="TB",
                            node_producer=GetOpNodeProducer("docstring", color="yellow",
                                                            fillcolor="yellow", style="filled"))
pydot_graph.write_dot("tfidfvectorizer.dot")

import os
os.system('dot -O -Gdpi=300 -Tpng tfidfvectorizer.dot')

import matplotlib.pyplot as plt
image = plt.imread("tfidfvectorizer.dot.png")
fig, ax = plt.subplots(figsize=(40, 20))
ax.imshow(image)
ax.axis('off')


###########################
# Visualize intermediate outputs
# ++++++++++++++++++++++++++++++

from skonnxrt.sklapi import OnnxTransformer

with open("TfidfVectorizer.onnx", "rb") as f:
    content = f.read()

input = corpus[2]
print("with input:", [input])
for step in OnnxTransformer.enumerate_create(content):
    print("-> node '{}'".format(step[0]))
    step[1].fit()
    print(step[1].transform(input))

#################################
# **Versions used for this example**

import numpy
import sklearn
print("numpy:", numpy.__version__)
print("scikit-learn:", sklearn.__version__)
import onnx
import onnxruntime
import skl2onnx
import skonnxrt
print("onnx: ", onnx.__version__)
print("onnxruntime: ", onnxruntime.__version__)
print("scikit-onnxruntime: ", skonnxrt.__version__)
print("skl2onnx: ", skl2onnx.__version__)
