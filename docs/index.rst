
scikit-onnxruntime: use onnxruntime in scikit-learn piplines
============================================================

.. list-table:
    :header-rows: 1
    :widths: 5
    * - Linux
    * - .. image:: https://dev.azure.com/xadupre/scikit-onnxruntime/_apis/build/status/xadupre.scikit.onnxruntime
            :target: https://dev.azure.com/xadupre/scikit-onnxruntime/
	    
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
The following example is taken from :ref:`l-example-transfer-learning`.

::

    with open("rf_iris.onnx", "rb") as f:
        content = f.read()

    ot = OnnxTransformer(content, output_name="output_probability")
    ot.fit(X_train, y_train)

    print(ot.transform(X_test[:5]))

**License**

It is licensed with `MIT License <../LICENSE>`_.


