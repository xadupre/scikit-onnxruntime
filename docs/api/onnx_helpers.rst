

Onnx helpers
============

The following functions helps manipulating the graph.
It extends :epkg:`ONNX` documentation such as the following page
`Python API Overview <https://github.com/onnx/onnx/blob/master/docs/PythonAPIOverview.md>`_.

IO
++

.. autofunction:: skonnxrt.helpers.onnx_helper.load_onnx_model

.. autofunction:: skonnxrt.helpers.onnx_helper.save_onnx_model

Structure
+++++++++

.. autofunction:: skonnxrt.helpers.onnx_helper.enumerate_model_node_outputs

.. autofunction:: skonnxrt.helpers.onnx_helper.select_model_inputs_outputs
