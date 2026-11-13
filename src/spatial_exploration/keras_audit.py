"""Keras/TensorFlow training controls for the public DeepKriging 2D script.

Historical fold reuse is deliberately reproduced only as a diagnostic. Its
scores are not valid held-out performance. TensorFlow remains optional.
"""

from dataclasses import dataclass
import hashlib
import os
import time

import numpy as np

from .baseline import make_grid
from .spatial_features import wendland


@dataclass(frozen=True)
class AuditConfig:
    folds: int = 10
    count: int = 810
    seeds: tuple = (0, 1, 2)
    knot_grids: tuple = (10, 19, 37)
    width: int = 100
    batch_size: int = 64
    stage_epochs: int = 400
    chunk_epochs: int = 50
    order_seed: int = 8700000

    def validate(self):
        if min(self.folds, self.count, self.batch_size) < 2:
            raise ValueError("At least two folds, acquisitions, and batch size")
        if min(self.width, self.stage_epochs, self.chunk_epochs) < 1:
            raise ValueError("Positive network and training settings required")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Use distinct initialization seeds")
        if not self.knot_grids or min(self.knot_grids) < 2:
            raise ValueError("Invalid basis resolutions")


CONDITIONS = ("reuse_weights", "fresh_400", "fresh_matched")
_TF = None


def tensorflow():
    global _TF
    if _TF is None:
        os.environ.setdefault("KERAS_BACKEND", "tensorflow")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
        import tensorflow as tf
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.experimental.enable_op_determinism()
        _TF = tf
    return _TF


def source_features(points, grids):
    """Public-script raw dictionary, independent of the prior Torch module."""
    blocks = []
    for size in grids:
        knots = make_grid(size)
        distances = np.linalg.norm(points[:, None] - knots[None], axis=-1) / (2.5 / size)
        blocks.append(wendland(distances))
    return np.column_stack(blocks).astype(np.float32)


def stage_orders(config, seed, stage, size):
    rng = np.random.default_rng(np.random.SeedSequence([config.order_seed, seed, stage]))
    return np.stack([rng.permutation(size) for _ in range(config.stage_epochs)]).astype(np.int32)


def weight_hash(weights):
    digest = hashlib.sha256()
    for weight in weights:
        value = np.asarray(weight)
        digest.update(str(value.shape).encode()); digest.update(value.tobytes())
    return digest.hexdigest()


class KerasTrainer:
    """Source architecture/Adam, with an equivalent compiled batch loop.

    A graph loops over the same shuffled minibatches to avoid model.fit's
    per-epoch input-pipeline overhead. Tests compare against train_on_batch
    with identical batches, weights, and the actual Keras Adam implementation.
    Only currently acquired labels enter this API.
    """
    def __init__(self, inputs, config, seed, initial_weights=None):
        self.tf = tf = tensorflow()
        self.config = config
        tf.keras.utils.set_random_seed(int(seed))
        self.model = tf.keras.Sequential([
            tf.keras.Input(shape=(inputs,)),
            tf.keras.layers.Dense(config.width, activation="relu", kernel_initializer="he_uniform"),
            tf.keras.layers.Dense(config.width, activation="relu"),
            tf.keras.layers.Dense(config.width, activation="relu"),
            tf.keras.layers.Dense(1),
        ])
        if initial_weights is not None:
            self.model.set_weights(initial_weights)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=.001, beta_1=.9, beta_2=.999, epsilon=1e-7)
        self.optimizer.build(self.model.trainable_variables)
        self.initial_hash = weight_hash(self.model.get_weights())
        self._step = tf.function(self._batch, reduce_retracing=True)
        self._epochs = tf.function(self._run_epochs, reduce_retracing=True)

    def _batch(self, x, y):
        tf = self.tf
        with tf.GradientTape() as tape:
            prediction = tf.squeeze(self.model(x, training=True), axis=-1)
            loss = tf.reduce_mean(tf.square(prediction - y))
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def _run_epochs(self, x, y, orders):
        tf = self.tf
        for epoch in tf.range(tf.shape(orders)[0]):
            order = orders[epoch]
            for start in tf.range(0, tf.shape(x)[0], self.config.batch_size):
                ids = order[start:start + self.config.batch_size]
                self._batch(tf.gather(x, ids), tf.gather(y, ids))
        return tf.reduce_mean(tf.square(tf.squeeze(self.model(x, training=False), -1) - y))

    def reset_optimizer(self):
        # Source recompiles with a fresh Adam at each fold. Keep the learning
        # rate; clear iteration and moment slots, including in matched stages.
        for variable in self.optimizer.variables:
            if variable is not self.optimizer.learning_rate:
                variable.assign(self.tf.zeros_like(variable))

    def fit_stage(self, features, acquired, orders):
        x, y = np.asarray(features, dtype=np.float32), np.asarray(acquired, dtype=np.float32)
        if x.ndim != 2 or y.shape != (len(x),) or len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Supply only finite acquired features and labels")
        if (orders.shape != (self.config.stage_epochs, len(x))
                or not np.all(np.sort(orders, axis=1) == np.arange(len(x)))):
            raise ValueError("Each training epoch must permute acquired locations exactly once")
        self.reset_optimizer()
        start_hash = weight_hash(self.model.get_weights())
        began = time.perf_counter()
        histories = []
        for start in range(0, len(orders), self.config.chunk_epochs):
            end = min(start + self.config.chunk_epochs, len(orders))
            loss = float(self._epochs(self.tf.constant(x), self.tf.constant(y), self.tf.constant(orders[start:end])))
            if not np.isfinite(loss):
                raise RuntimeError("Nonfinite Keras training loss")
            histories.append({"epoch_in_stage": end, "train_mse": loss})
        seconds = time.perf_counter() - began
        steps = int(self.optimizer.iterations.numpy())
        expected = len(orders) * int(np.ceil(len(x) / self.config.batch_size))
        if steps != expected:
            raise AssertionError("Optimizer update count mismatch")
        return {"fit_seconds": seconds, "updates": steps, "histories": histories,
                "start_stage_sha256": start_hash, "final_sha256": weight_hash(self.model.get_weights())}

    def predict(self, features):
        began = time.perf_counter()
        mean = self.model(self.tf.constant(np.asarray(features, dtype=np.float32)), training=False).numpy()[:, 0].astype(float)
        if not np.isfinite(mean).all():
            raise RuntimeError("Nonfinite Keras prediction")
        return mean, time.perf_counter() - began
