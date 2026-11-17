"""MLflow pyfunc wrapper so registered bundles are self-describing and portable.

The registry entry contains the bundle folder (voice model + transcript model +
fusion parameters), the ``src`` package (``code_paths``) and pinned requirements.
Any MLflow consumer can do::

    m = mlflow.pyfunc.load_model("models:/covert-call-detector@champion")
    m.predict(pandas.DataFrame({"transcript": [...], "stress": [...]}))

and get the fused threat decision for each row. The FastAPI service downloads the
same bundle and runs the full audio path (windows, voice model, speech-to-text).
"""

from __future__ import annotations

import mlflow.pyfunc


class CallPyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from src.bundle import load_bundle

        self.model = load_bundle(context.artifacts["bundle"])

    def predict(self, context, model_input, params=None):
        import numpy as np

        rows = model_input.to_dict("records")
        return [
            self.model.decide(np.array([float(r.get("stress", 0.0))]), r.get("transcript", ""))
            for r in rows
        ]
