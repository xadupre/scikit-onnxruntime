
scikit-onnxruntime: use onnxruntime in scikit-learn piplines
============================================================

.. list-table:
    :header-rows: 1
    :widths: 5 5
    * - Linux
      - Windows
    * - .. image:: https://dev.azure.com/onnxmltools/sklearn-onnx/_apis/build/status/sklearn-onnx-linux-conda-ci?branchName=master
            :target: https://dev.azure.com/onnxmltools/sklearn-onnx/_build/latest?definitionId=5?branchName=master
      - .. image:: https://dev.azure.com/onnxmltools/sklearn-onnx/_apis/build/status/sklearn-onnx-win32-conda-ci?branchName=master
            :target: https://dev.azure.com/onnxmltools/sklearn-onnx/_build/latest?definitionId=5?branchName=master
	    
	    
*scikit-onnxruntime* enables you to use `ONNX <https://onnx.ai>`_ file
in `sklearn-learn <https://scikit-learn.org/stable/>`_ pipelines.

.. toctree::
    :maxdepth: 1
    
    tutorial
    api_summary
    auto_examples/index

**Issues, questions**

You should look for `existing issues <https://github.com/xadupre/scikit-onnxruntime/issues?utf8=%E2%9C%93&q=is%3Aissue>`_
or submit a new one. Sources are available on
`xadupre/scikit-onnxruntime <https://github.com/xadupre/scikit-onnxruntime>`_.

::

    # Train a model.
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clr = RandomForestClassifier()
    clr.fit(X_train, y_train)

    # Convert into ONNX format with onnxmltools
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    initial_type = [('float_input', FloatTensorType([1, 4]))]
    onx = convert_sklearn(clr, initial_types=initial_type)
    with open("rf_iris.onnx", "wb") as f:
    f.write(onx.SerializeToString())

    # Compute the prediction with ONNX Runtime
    import onnxruntime as rt
    import numpy
    sess = rt.InferenceSession("rf_iris.onnx")
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    pred_onx = sess.run([label_name], {input_name: X_test.astype(numpy.float32)})[0]


**License**

It is licensed with `MIT License <../LICENSE>`_.


