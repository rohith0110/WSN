"""A compact deep Q-network in NumPy: MLP, Adam, experience replay and a target network.

Input: global features of the current node plus padded per-candidate features for the ``K``
ranked slots and a validity mask. Output: one Q-value per slot. Invalid slots are never chosen
and never used as bootstrap targets.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np
import numpy.typing as npt

from wsn.config import DQNConfig
from wsn.models import NodeId
from wsn.network import NetworkView
from wsn.rl.state import Candidate

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

GLOBAL_FEATURES = 4
CANDIDATE_FEATURES = 6


def input_size(candidate_limit: int) -> int:
    return GLOBAL_FEATURES + candidate_limit * (CANDIDATE_FEATURES + 1)


def encode_observation(
    view: NetworkView, current: NodeId, candidates: Sequence[Candidate], candidate_limit: int
) -> FloatArray:
    """[energy, sink distance, utilization, density] + K x [candidate features] + K-slot mask."""
    features = np.zeros(input_size(candidate_limit), dtype=np.float64)
    features[0] = view.energy_fraction(current)
    features[1] = view.distance_to_sink_m(current) / view.field_diagonal_m
    features[2] = view.utilization(current)
    features[3] = view.density(current)
    mask_start = GLOBAL_FEATURES + candidate_limit * CANDIDATE_FEATURES
    for slot, candidate in enumerate(candidates[:candidate_limit]):
        start = GLOBAL_FEATURES + slot * CANDIDATE_FEATURES
        features[start : start + CANDIDATE_FEATURES] = (
            candidate.energy_fraction,
            candidate.link_probability,
            min(max(candidate.progress_m / view.communication_range_m, -1.0), 1.0),
            candidate.utilization,
            candidate.density,
            candidate.tx_cost_ratio - 1.0,
        )
        features[mask_start + slot] = 1.0
    return features


@dataclass(slots=True)
class DenseLayer:
    weight: FloatArray
    bias: FloatArray


class MLP:
    """Fully connected network with ReLU hidden layers and a linear output layer."""

    def __init__(self, layers: list[DenseLayer]) -> None:
        if not layers:
            raise ValueError("An MLP needs at least one layer")
        for previous, layer in pairwise(layers):
            if previous.weight.shape[1] != layer.weight.shape[0]:
                raise ValueError("Layer shapes do not chain")
        self.layers = layers

    @classmethod
    def initialise(cls, sizes: Sequence[int], rng: np.random.Generator) -> MLP:
        """He-normal weights, zero biases."""
        layers = [
            DenseLayer(
                rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out)),
                np.zeros(fan_out, dtype=np.float64),
            )
            for fan_in, fan_out in pairwise(sizes)
        ]
        return cls(layers)

    @property
    def input_size(self) -> int:
        return int(self.layers[0].weight.shape[0])

    @property
    def output_size(self) -> int:
        return int(self.layers[-1].weight.shape[1])

    def copy(self) -> MLP:
        return MLP([DenseLayer(layer.weight.copy(), layer.bias.copy()) for layer in self.layers])

    def forward(self, inputs: FloatArray) -> FloatArray:
        activation = inputs
        for index, layer in enumerate(self.layers):
            activation = activation @ layer.weight + layer.bias
            if index < len(self.layers) - 1:
                activation = np.maximum(activation, 0.0)
        return activation

    def gradients(
        self, inputs: FloatArray, output_gradient_fn: OutputGradient
    ) -> tuple[list[DenseLayer], FloatArray]:
        """Backpropagate ``dLoss/dOutput`` (computed from the outputs) to every parameter."""
        activations = [inputs]
        for index, layer in enumerate(self.layers):
            z = activations[-1] @ layer.weight + layer.bias
            activations.append(np.maximum(z, 0.0) if index < len(self.layers) - 1 else z)
        outputs = activations[-1]
        delta = output_gradient_fn(outputs)
        grads: list[DenseLayer] = []
        for index in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[index]
            grads.append(DenseLayer(activations[index].T @ delta, delta.sum(axis=0)))
            if index > 0:
                delta = (delta @ layer.weight.T) * (activations[index] > 0.0)
        grads.reverse()
        return grads, outputs

    def parameter_count(self) -> int:
        return sum(layer.weight.size + layer.bias.size for layer in self.layers)


class OutputGradient:
    """Huber-loss gradient on the taken action only, averaged over the batch."""

    def __init__(self, actions: IntArray, targets: FloatArray, delta: float) -> None:
        self.actions = actions
        self.targets = targets
        self.delta = delta
        self.loss = 0.0

    def __call__(self, outputs: FloatArray) -> FloatArray:
        rows = np.arange(len(self.actions))
        error = outputs[rows, self.actions] - self.targets
        clipped = np.clip(error, -self.delta, self.delta)
        absolute = np.abs(error)
        quadratic = np.minimum(absolute, self.delta)
        self.loss = float(np.mean(0.5 * quadratic**2 + self.delta * (absolute - quadratic)))
        gradient = np.zeros_like(outputs)
        gradient[rows, self.actions] = clipped / len(self.actions)
        return gradient


@dataclass(slots=True)
class AdamState:
    learning_rate: float
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    step: int = 0
    first: list[DenseLayer] = field(default_factory=list)
    second: list[DenseLayer] = field(default_factory=list)

    def apply(self, network: MLP, grads: list[DenseLayer]) -> None:
        if not self.first:
            self.first = [_zeros_like(layer) for layer in network.layers]
            self.second = [_zeros_like(layer) for layer in network.layers]
        self.step += 1
        correction1 = 1.0 - self.beta1**self.step
        correction2 = 1.0 - self.beta2**self.step
        for layer, grad, m, v in zip(network.layers, grads, self.first, self.second, strict=True):
            self._update(layer.weight, grad.weight, m.weight, v.weight, (correction1, correction2))
            self._update(layer.bias, grad.bias, m.bias, v.bias, (correction1, correction2))

    def _update(
        self,
        param: FloatArray,
        grad: FloatArray,
        first: FloatArray,
        second: FloatArray,
        corrections: tuple[float, float],
    ) -> None:
        """In-place Adam step for one parameter array and its moment estimates."""
        first *= self.beta1
        first += (1.0 - self.beta1) * grad
        second *= self.beta2
        second += (1.0 - self.beta2) * grad * grad
        step = (first / corrections[0]) / (np.sqrt(second / corrections[1]) + self.epsilon)
        param -= self.learning_rate * step


def _zeros_like(layer: DenseLayer) -> DenseLayer:
    return DenseLayer(np.zeros_like(layer.weight), np.zeros_like(layer.bias))


class ReplayBuffer:
    """Fixed-size ring buffer of transitions; ``next_valid = 0`` marks a terminal transition."""

    def __init__(self, capacity: int, feature_count: int) -> None:
        self.capacity = capacity
        self.observations = np.zeros((capacity, feature_count), dtype=np.float64)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float64)
        self.next_observations = np.zeros((capacity, feature_count), dtype=np.float64)
        self.next_valid = np.zeros(capacity, dtype=np.int64)
        self.size = 0
        self._cursor = 0

    def add(
        self,
        observation: FloatArray,
        action: int,
        reward: float,
        next_observation: FloatArray | None,
        next_valid: int,
    ) -> None:
        index = self._cursor
        self.observations[index] = observation
        self.actions[index] = action
        self.rewards[index] = reward
        if next_observation is None:
            self.next_observations[index] = 0.0
            self.next_valid[index] = 0
        else:
            self.next_observations[index] = next_observation
            self.next_valid[index] = next_valid
        self._cursor = (self._cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> IntArray:
        return rng.integers(0, self.size, size=batch_size)


def masked_max(q_values: FloatArray, valid_counts: IntArray) -> FloatArray:
    """Row-wise max over the first ``valid_count`` slots; 0 for terminal rows (count 0)."""
    slots = np.arange(q_values.shape[1])
    mask = slots[None, :] < valid_counts[:, None]
    masked = np.where(mask, q_values, -np.inf)
    best = masked.max(axis=1)
    result: FloatArray = np.where(valid_counts > 0, best, 0.0).astype(np.float64)
    return result


class DQNLearner:
    """Online network + periodically synchronised target network trained from replay."""

    def __init__(self, network: MLP, config: DQNConfig, gamma: float, seed: int) -> None:
        self.online = network
        self.target = network.copy()
        self.config = config
        self.gamma = gamma
        self.optimizer = AdamState(config.learning_rate)
        self.replay = ReplayBuffer(config.replay_capacity, network.input_size)
        self.rng = np.random.default_rng(seed)
        self.transitions = 0
        self.gradient_steps = 0
        self.last_loss: float | None = None

    def q_values(self, observation: FloatArray) -> FloatArray:
        values: FloatArray = self.online.forward(observation[None, :])[0]
        return values

    def record(
        self,
        observation: FloatArray,
        action: int,
        reward: float,
        next_observation: FloatArray | None,
        next_valid: int,
    ) -> None:
        self.replay.add(observation, action, reward, next_observation, next_valid)
        self.transitions += 1
        ready = self.replay.size >= self.config.warmup_transitions
        if ready and self.transitions % self.config.train_every == 0:
            self.train_step()

    def train_step(self) -> None:
        batch = self.replay.sample(self.config.batch_size, self.rng)
        next_q = self.target.forward(self.replay.next_observations[batch])
        targets = self.replay.rewards[batch] + self.gamma * masked_max(
            next_q, self.replay.next_valid[batch]
        )
        gradient = OutputGradient(self.replay.actions[batch], targets, self.config.huber_delta)
        grads, _ = self.online.gradients(self.replay.observations[batch], gradient)
        self.optimizer.apply(self.online, grads)
        self.last_loss = gradient.loss
        self.gradient_steps += 1
        if self.gradient_steps % self.config.target_sync_steps == 0:
            self.target = self.online.copy()
