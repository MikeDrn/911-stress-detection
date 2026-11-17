"""MLflow pyfunc wrapper so registered models are self-describing and portable.

The registry entry contains the raw checkpoint (``artifacts/model.pt``), the
``src`` package (``code_paths``) and pinned requirements, so any MLflow
consumer can ``mlflow.pyfunc.load_model("models:/voice-stress-detector@champion")``
and call ``predict`` on a batch of Log-Mel / MFCC tensors.

The FastAPI service does *not* go through pyfunc: it downloads the same
checkpoint and builds a ``TorchStressPredictor`` directly, which keeps the API
image free of pyfunc's environment machinery.
"""

from __future__ import annotations

import mlflow.pyfunc
import numpy as np


class StressPyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from src.model import TorchStressPredictor

        self.predictor = TorchStressPredictor.from_checkpoint(context.artifacts["checkpoint"])

    def predict(self, context, model_input, params=None):
        x = np.asarray(model_input, dtype=np.float32)
        return self.predictor.predict_proba({self.predictor.input_key: x})
