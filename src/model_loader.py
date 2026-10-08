"""Load the recommender without version-specific serialized Python bytecode."""

import json
import zipfile

import tensorflow as tf


def load_recommender(path):
    with zipfile.ZipFile(path) as archive:
        config = json.loads(archive.read("config.json"))

    # Inference needs neither the saved optimizer nor its training configuration.
    config.pop("compile_config", None)
    for layer in config["config"]["layers"]:
        settings = layer["config"]
        if settings.get("quantization_config", "missing") is None:
            settings.pop("quantization_config")
        initializer = settings.get("kernel_initializer", {})
        if initializer.get("class_name") == "GlorotUniform":
            for axis in ("input_axes", "output_axes"):
                if initializer["config"].get(axis, "missing") is None:
                    initializer["config"].pop(axis)
        if layer["class_name"] == "Lambda" and settings["name"] == "rating_scale":
            # Matches src/neural_network.py: value * 9.0 + 1.0.
            # Rescaling is portable across Python versions and has no weights.
            layer["class_name"] = "Rescaling"
            settings.pop("function")
            settings.pop("arguments", None)
            settings.update(scale=9.0, offset=1.0)

    model = tf.keras.models.model_from_json(json.dumps(config))
    model.load_weights(path)
    return model
