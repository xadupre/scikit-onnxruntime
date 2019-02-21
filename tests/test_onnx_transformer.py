# Licensed under the MIT License.

# -*- coding: UTF-8 -*-
import unittest
import os
import sys
import numpy as np
import pandas
import onnxruntime
from onnxruntime.capi._pybind_state import onnxruntime_ostream_redirect
from onnxruntime.datasets import get_example
from skonnxrt.sklapi import OnnxTransformer


class TestInferenceSessionSklearn(unittest.TestCase):
    
    def get_name(self, name):
        return get_example(name)

    def test_transform_numpy(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        name = self.get_name("mul_1.pb")
        with open(name, "rb") as f:
            content = f.read()
            
        tr = OnnxTransformer(content)
        tr.fit()
        res = tr.transform(x)
        exp = np.array([[ 1.,  4.], [ 9., 16.], [25., 36.]], dtype=np.float32)
        self.assertEqual(list(res.ravel()), list(exp.ravel()))

    def test_transform_list(self):
        x = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        name = self.get_name("mul_1.pb")
        with open(name, "rb") as f:
            content = f.read()
            
        tr = OnnxTransformer(content)
        tr.fit()
        res = tr.transform(x)
        exp = np.array([[ 1.,  4.], [ 9., 16.], [25., 36.]], dtype=np.float32)
        self.assertEqual(list(res.ravel()), list(exp.ravel()))

    def test_transform_dict(self):
        x = {'X': np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])}
        name = self.get_name("mul_1.pb")
        with open(name, "rb") as f:
            content = f.read()
            
        tr = OnnxTransformer(content)
        tr.fit()
        res = tr.transform(x)
        exp = np.array([[ 1.,  4.], [ 9., 16.], [25., 36.]], dtype=np.float32)
        self.assertEqual(list(res.ravel()), list(exp.ravel()))

    def test_transform_dataframe(self):
        x = pandas.DataFrame(data=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        x.columns = "X1 X2".split()
        name = self.get_name("mul_1.pb")
        with open(name, "rb") as f:
            content = f.read()
            
        tr = OnnxTransformer(content)
        tr.fit()
        try:
            tr.transform(x)
        except RuntimeError:
            pass

        
if __name__ == '__main__':
    unittest.main()
