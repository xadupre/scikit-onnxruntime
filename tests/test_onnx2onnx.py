# Licensed under the MIT License.

# -*- coding: UTF-8 -*-
import unittest
from numpy.testing import assert_almost_equal
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn
from skonnxrt.sklapi import OnnxTransformer


class TestInferenceSessionOnnx2Onnx(unittest.TestCase):

    def test_pipeline(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        pca = PCA(n_components=2)
        pca.fit(X)

        onx = convert_sklearn(pca, initial_types=[
                              ('input', FloatTensorType((1, X.shape[1])))])
        onx_bytes = onx.SerializeToString()
        tr = OnnxTransformer(onx_bytes)

        pipe = make_pipeline(tr, LogisticRegression())
        pipe.fit(X, y)
        pred = pipe.predict(X)
        self.assertEqual(pred.shape, (150, ))
        skl_pred = pca.transform(X)
        skl_onx = pipe.steps[0][1].transform(X)
        assert_almost_equal(skl_pred, skl_onx, decimal=5)


if __name__ == '__main__':
    unittest.main()
