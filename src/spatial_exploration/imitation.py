"""Masked behavioral cloning on common teacher-labelled exploration states."""

from dataclasses import asdict, dataclass
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .baseline import write_csv
from .exploration_env import SpatialExplorationEnv
from .rollout import rollout_for_environment


@dataclass(frozen=True)
class CloningConfig:
    train_episodes: int = 1024
    validation_episodes: int = 128
    test_episodes: int = 96
    epochs: int = 40
    batch_size: int = 128
    learning_rate: float = 0.0003
    hidden_sizes: tuple = (128, 128)
    seeds: tuple = (0, 1)

    def validate(self):
        if min(self.train_episodes, self.validation_episodes, self.test_episodes,
               self.epochs, self.batch_size) < 1:
            raise ValueError("Episode counts, epochs, and batch size must be positive")
        if not self.seeds or len(self.seeds) != len(set(self.seeds)):
            raise ValueError("Specify distinct training seeds")
        if not self.hidden_sizes or any(size < 1 for size in self.hidden_sizes):
            raise ValueError("Hidden layer sizes must be positive")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Learning rate must be finite and positive")


class ClonedPolicy(nn.Module):
    """Same flat observation and actor layer sizes/activation as the PPO baseline."""

    def __init__(self, observation_size, action_count, hidden_sizes=(128, 128)):
        super().__init__()
        self.observation_size = observation_size
        self.action_count = action_count
        self.hidden_sizes = tuple(hidden_sizes)
        layers, width = [], observation_size
        for size in hidden_sizes:
            layer = nn.Linear(width, size)
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.zeros_(layer.bias)
            layers.extend((layer, nn.Tanh()))
            width = size
        output = nn.Linear(width, action_count)
        nn.init.orthogonal_(output.weight, gain=0.01)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.network = nn.Sequential(*layers)

    def forward(self, observations):
        return self.network(observations)

    def predict(self, observation, action_masks, deterministic=True):
        if not deterministic:
            raise ValueError("This baseline deploys deterministic masked argmax only")
        self.eval()
        with torch.inference_mode():
            states = torch.as_tensor(np.asarray(observation), dtype=torch.float32).unsqueeze(0)
            masks = torch.as_tensor(np.asarray(action_masks), dtype=torch.bool).unsqueeze(0)
            return int(mask_logits(self(states), masks).argmax(dim=1).item()), None

    def save(self, path, metadata=None):
        torch.save({"state_dict": self.state_dict(), "observation_size": self.observation_size,
            "action_count": self.action_count, "hidden_sizes": list(self.hidden_sizes),
            "metadata": metadata or {}}, path)

    @classmethod
    def load(cls, path):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        policy = cls(saved["observation_size"], saved["action_count"], saved["hidden_sizes"])
        policy.load_state_dict(saved["state_dict"])
        policy.eval()
        return policy


def mask_logits(logits, masks):
    if logits.shape != masks.shape or masks.dtype != torch.bool:
        raise ValueError("Boolean masks must have the same shape as logits")
    if not bool(masks.any(dim=-1).all()):
        raise ValueError("Each nonterminal state needs a feasible action")
    return logits.masked_fill(~masks, -torch.inf)


def masked_cross_entropy(logits, masks, labels):
    if labels.ndim != 1 or len(labels) != len(logits):
        raise ValueError("Need one teacher action per observation")
    if not bool(((labels >= 0) & (labels < logits.shape[1])).all()):
        raise ValueError("Teacher action outside action space")
    if not bool(masks.gather(1, labels[:, None]).all()):
        raise ValueError("Teacher actions must be feasible")
    return F.cross_entropy(mask_logits(logits, masks), labels)


def collect_common_states(config, cases, output):
    """Alternate greedy/rollout behavior by episode; label both at every state.

    Both supervised models see exactly the same inputs and masks. No random
    field is sampled: the covariance objective does not depend on measurements.
    Store episode identities so splitting never treats adjacent states as IID.
    """
    output = Path(output)
    observations, masks, greedy_actions, rollout_actions, episode_ids = [], [], [], [], []
    episode_rows = []
    teacher_seconds = {"greedy": 0.0, "rollout": 0.0}
    started = time.perf_counter()
    env = SpatialExplorationEnv(config)
    for index, case in enumerate(cases):
        observation, _ = env.reset(seed=case["field_seed"], options=case)
        behavior = "rollout" if index % 2 == 0 else "greedy"
        count = 0
        while not env.finished:
            began = time.perf_counter()
            greedy = env.greedy_action()
            teacher_seconds["greedy"] += time.perf_counter() - began
            began = time.perf_counter()
            rollout = rollout_for_environment(env).first
            teacher_seconds["rollout"] += time.perf_counter() - began
            mask = env.action_masks()
            if not mask[greedy] or not mask[rollout]:
                raise RuntimeError("Infeasible teacher label")
            observations.append(observation.copy())
            masks.append(mask.copy())
            greedy_actions.append(greedy)
            rollout_actions.append(rollout)
            episode_ids.append(case["scenario"])
            action = rollout if behavior == "rollout" else greedy
            observation, _, _, _, info = env.step(action)
            if info["invalid_action"]:
                raise RuntimeError("Collection behavior violated the action mask")
            count += 1
        episode_rows.append({"scenario": case["scenario"], "behavior": behavior,
            "states": count, "final_mean_variance": env.mean_variance,
            "travel_distance": env.travel_distance, "route": " ".join(map(str, env.route))})
        if (index + 1) % 64 == 0 or index + 1 == len(cases):
            print(f"Collected {output.stem}: {index + 1}/{len(cases)} episodes, {len(observations)} states", flush=True)
    data = {"observations": np.asarray(observations, dtype=np.float32),
        "masks": np.asarray(masks, dtype=bool), "greedy_actions": np.asarray(greedy_actions, dtype=np.int64),
        "rollout_actions": np.asarray(rollout_actions, dtype=np.int64),
        "episode_ids": np.asarray(episode_ids, dtype=np.int64)}
    np.savez_compressed(output, **data)
    write_csv(output.with_suffix(".episodes.csv"), episode_rows)
    return data, {"episodes": len(cases), "states": len(observations),
        "teacher_action_agreement": float(np.mean(data["greedy_actions"] == data["rollout_actions"])),
        "collection_seconds": time.perf_counter() - started, "teacher_label_seconds": teacher_seconds}


def tensor_data(data, teacher):
    return (torch.as_tensor(data["observations"], dtype=torch.float32),
            torch.as_tensor(data["masks"], dtype=torch.bool),
            torch.as_tensor(data[f"{teacher}_actions"], dtype=torch.long))


def classification_metrics(policy, tensors, batch_size=512):
    states, masks, labels = tensors
    total_loss, correct, count = 0.0, 0, len(labels)
    policy.eval()
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            logits = policy(states[start:start + batch_size])
            selected_masks, selected_labels = masks[start:start + batch_size], labels[start:start + batch_size]
            loss = masked_cross_entropy(logits, selected_masks, selected_labels)
            total_loss += float(loss) * len(selected_labels)
            correct += int((mask_logits(logits, selected_masks).argmax(1) == selected_labels).sum())
    return {"cross_entropy": total_loss / count, "action_accuracy": correct / count, "states": count}


def train_policy(train_data, validation_data, teacher, seed, config, output):
    """Fixed epoch budget; select checkpoints by held-out-episode validation CE."""
    config.validate()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    policy = ClonedPolicy(train_data["observations"].shape[1], train_data["masks"].shape[1], config.hidden_sizes)
    train_tensors, validation_tensors = tensor_data(train_data, teacher), tensor_data(validation_data, teacher)
    policy.save(output / "untrained.pt", {"teacher": teacher, "seed": seed})
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(seed + 10000)
    initial_train = classification_metrics(policy, train_tensors)
    initial_validation = classification_metrics(policy, validation_tensors)
    best_loss, best_epoch, history = float("inf"), 0, []
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        policy.train()
        permutation = torch.randperm(len(train_tensors[0]), generator=generator)
        total_loss = 0.0
        for indices in permutation.split(config.batch_size):
            states, masks, labels = (tensor[indices] for tensor in train_tensors)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_cross_entropy(policy(states), masks, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
        validation = classification_metrics(policy, validation_tensors)
        history.append({"epoch": epoch, "train_cross_entropy": total_loss / len(permutation),
                        **{f"validation_{key}": value for key, value in validation.items()}})
        if validation["cross_entropy"] < best_loss:
            best_loss, best_epoch = validation["cross_entropy"], epoch
            policy.save(output / "selected.pt", {"teacher": teacher, "seed": seed,
                "selected_epoch": epoch, "selection": "minimum validation cross entropy",
                "training_config": asdict(config)})
        if epoch % 10 == 0 or epoch == config.epochs:
            print(f"BC {teacher} seed={seed}, epoch={epoch}, val accuracy={validation['action_accuracy']:.3f}", flush=True)
    elapsed = time.perf_counter() - started
    policy.save(output / "final.pt", {"teacher": teacher, "seed": seed, "epoch": config.epochs})
    selected = ClonedPolicy.load(output / "selected.pt")
    write_csv(output / "training_history.csv", history)
    return selected, {"teacher": teacher, "seed": seed, "selected_epoch": best_epoch,
        "training_seconds": elapsed, "parameters": sum(item.numel() for item in policy.parameters()),
        "initial_train": initial_train, "initial_validation": initial_validation,
        "selected_train": classification_metrics(selected, train_tensors),
        "selected_validation": classification_metrics(selected, validation_tensors)}
