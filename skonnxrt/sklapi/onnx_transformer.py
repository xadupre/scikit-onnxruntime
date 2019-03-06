# coding: utf-8
"""
Wraps runtime into a :epkg:`scikit-learn` transformer.
"""
# author: Xavier Dupré
# license: MIT

import numpy
import pandas
from sklearn.base import BaseEstimator, TransformerMixin
from onnxruntime import InferenceSession


class OnnxTransformer(BaseEstimator, TransformerMixin):
    """
    Calls :epkg:`onnxruntime` inference following :epkg:`scikit-learn` API
    so that it can be included in a :epkg:`scikit-learn` pipeline.

    Parameters
    ----------

    onnx_bytes : bytes
    output_name: string
        requested output name or None to request all and
        have method *transform* to store all of them in a dataframe
    enforce_float32 : boolean
        :epkg:`onnxruntime` only supports *float32*,
        :epkg:`scikit-learn` usually uses double floats, this parameter
        ensures that every array of double floats is converted into
        single floats
    """

    def __init__(self, onnx_bytes, output_name=None, enforce_float32=True):
        BaseEstimator.__init__(self)
        TransformerMixin.__init__(self)
        self.onnx_bytes = onnx_bytes
        self.output_name = output_name
        self.enforce_float32 = enforce_float32
        if not isinstance(onnx_bytes, bytes):
            raise TypeError("onnx_bytes must be bytes to be pickled.")

    def fit(self, X=None, y=None, **fit_params):
        """
        Loads the :epkg:`ONNX` model.

        Parameters
        ----------
        X : unused
        y : unused

        Returns
        -------
        self
        """
        self.onnxrt_ = InferenceSession(self.onnx_bytes)
        self.inputs_ = [_.name for _ in self.onnxrt_.get_inputs()]
        return self

    def _check_arrays(self, inputs):
        """
        Ensures that double floats are converted into single floats
        if *enforce_float32* is True or raises an exception.
        """
        for k in inputs:
            v = inputs[k]
            if isinstance(v, numpy.ndarray):
                if v.dtype == numpy.float64:
                    if self.enforce_float32:
                        inputs[k] = v.astype(numpy.float32)
                    else:
                        raise TypeError(
                            "onnxunruntime only supports floats. Input '{0}' should be converted.".format(k))

    def transform(self, X, y=None, **inputs):
        """
        Runs the predictions. If *X* is a dataframe,
        the function assumes every columns is a separate input,
        otherwise, *X* is considered as a first input and *inputs*
        can be used to specify extra inputs.

        Parameters
        ----------
        X : iterable, data to process (or first input if several expected)
        y : unused
        inputs: :epkg:`ONNX` graph support multiple inputs,
            each column of a dataframe is converted into as many inputs if
            *X* is a dataframe, otherwise, *X* is considered as the first input
            and *inputs* can be used to specify the other ones

        Returns
        -------
        :epkg:`DataFrame`
        """
        if not hasattr(self, "onnxrt_"):
            raise AttributeError(
                "Transform OnnxTransformer must be fit first.")
        rt_inputs = {}
        if isinstance(X, pandas.DataFrame):
            for c in X.columns:
                rt_inputs[c] = X[c]
        elif isinstance(X, numpy.ndarray):
            rt_inputs[self.inputs_[0]] = X
        elif isinstance(X, dict) and len(inputs) == 0:
            for k, v in X.items():
                rt_inputs[k] = v
        elif isinstance(X, list):
            if len(self.inputs_) == 1:
                rt_inputs[self.inputs_[0]] = numpy.array(X)
            else:
                for i in range(len(self.inputs_)):
                    rt_inputs[self.inputs_[i]] = [row[i] for row in X]

        for k, v in inputs.items():
            rt_inputs[k] = v

        names = [self.output_name] if self.output_name else None
        self._check_arrays(rt_inputs)
        outputs = self.onnxrt_.run(names, rt_inputs)

        if self.output_name or len(outputs) == 1:
            if isinstance(outputs[0], list):
                return pandas.DataFrame(outputs[0])
            else:
                return outputs[0]
        else:
            names = self.output_name if self.output_name else [
                o.name for o in self.onnxrt_.get_outputs()]
            return pandas.DataFrame({k: v for k, v in zip(names, outputs)})

    def fit_transform(self, X, y=None, **inputs):
        """
        Loads the *ONNX* model and runs the predictions.

        Parameters
        ----------
        X : iterable, data to process (or first input if several expected)
        y : unused
        inputs: :epkg:`ONNX` graph support multiple inputs,
            each column of a dataframe is converted into as many inputs if
            *X* is a dataframe, otherwise, *X* is considered as the first input
            and *inputs* can be used to specify the other ones

        Returns
        -------
        :epkg:`DataFrame`
        """
        return self.fit(X, y=y, **inputs).transform(X, y)
