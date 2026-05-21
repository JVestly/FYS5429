"""RL and graph-neural-network utilities for CSS quantum-code decoding.

This file contains the reusable machinery used by the notebooks:

* sampling toy Pauli/component errors,
* computing binary syndromes,
* representing a parity-check matrix as a Tanner graph,
* running a single-channel CSS correction environment,
* training a graph actor-critic decoder.

The terminology used here is:

``CSS channel``
    One half of CSS decoding.  To correct Z errors, pass ``Hx`` because X
    stabilizer checks detect Z errors.  To correct X errors, pass ``Hz``.

``syndrome``
    The binary vector ``H @ error mod 2``.  A one means the corresponding
    stabilizer check clicked.

``episode trajectory``
    One complete decoder attempt: syndrome observation, chosen qubit flips,
    rewards, and final success/failure information.  Older reinforcement
    learning texts often call this a "øø"; this module uses the more
    descriptive name ``EpisodeTrajectory`` and keeps ``Rollout`` as an alias
    for backward compatibility.

The module intentionally avoids heavy graph-learning packages.  It uses
NumPy/SciPy for sparse parity-check matrices and plain PyTorch for the GNN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy import sparse
from torch import Tensor, nn
from torch.distributions import Categorical


ArrayLike = np.ndarray
Sampler = Callable[[int, np.random.Generator], np.ndarray]


def to_csr_binary(matrix) -> sparse.csr_matrix:
    """
    Convert a parity-check matrix to binary CSR sparse format.

    Parameters
    ----------
    matrix : array-like or scipy.sparse matrix
        Matrix whose nonzero entries represent parity-check support.

    Returns
    -------
    scipy.sparse.csr_matrix
        Binary matrix with entries reduced modulo 2.
    """

    if sparse.issparse(matrix):
        result = matrix.tocsr().astype(np.uint8)
    else:
        result = sparse.csr_matrix(np.asarray(matrix, dtype=np.uint8))
    result.data %= 2
    result.eliminate_zeros()
    return result


def binary_syndrome(check_matrix: sparse.csr_matrix, error: np.ndarray) -> np.ndarray:
    """
    Compute the binary syndrome ``H @ error mod 2``.

    Parameters
    ----------
    check_matrix : scipy.sparse.csr_matrix
        Parity-check matrix ``H`` for one CSS channel.
    error : numpy.ndarray
        Binary error/component vector.  A one means that qubit has the relevant
        X or Z component.

    Returns
    -------
    numpy.ndarray
        Binary syndrome vector as ``uint8``.
    """

    return np.asarray((check_matrix @ error.astype(np.uint8)) % 2).ravel().astype(np.uint8)


def random_component_error(
    n_qubits: int,
    rng: np.random.Generator,
    p: float = 0.02,
    min_weight: int = 0,
    max_weight: Optional[int] = None,
) -> np.ndarray:
    """
    Sample a binary X-component or Z-component error vector.

    This function samples one CSS channel only.  For example, when training a
    Z-error decoder with ``Hx``, the returned vector marks which physical
    qubits have a Z component.

    Parameters
    ----------
    n_qubits : int
        Number of physical qubits.
    rng : numpy.random.Generator
        Random number generator used for reproducibility.
    p : float, default=0.02
        Independent probability that each qubit has this error component.
    min_weight : int, default=0
        Minimum number of qubits to flip.  Extra support is added uniformly if
        the sampled error is too small.
    max_weight : int or None, default=None
        Optional maximum support size.  If the sampled error is too large, it
        is trimmed uniformly.

    Returns
    -------
    numpy.ndarray
        Binary component-error vector of length ``n_qubits``.
    """

    error = (rng.random(n_qubits) < p).astype(np.uint8)

    if int(error.sum()) < min_weight:
        available = np.flatnonzero(error == 0)
        add = rng.choice(available, size=min_weight - int(error.sum()), replace=False)
        error[add] = 1

    if max_weight is not None and int(error.sum()) > max_weight:
        support = np.flatnonzero(error)
        keep = rng.choice(support, size=max_weight, replace=False)
        trimmed = np.zeros(n_qubits, dtype=np.uint8)
        trimmed[keep] = 1
        error = trimmed

    return error


def random_pauli_error(
    n_qubits: int,
    rng: np.random.Generator,
    p: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample a full Pauli error and split it into X and Z components.

    Parameters
    ----------
    n_qubits : int
        Number of physical qubits.
    rng : numpy.random.Generator
        Random number generator.
    p : float, default=0.02
        Probability that a qubit receives a non-identity Pauli error.  Given an
        error, X, Z, and Y are sampled uniformly.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(x_component, z_component)``.  A Y error appears as one in both
        returned vectors.
    """

    has_error = rng.random(n_qubits) < p
    pauli = rng.integers(1, 4, size=n_qubits)  # 1=X, 2=Z, 3=Y
    x_error = ((pauli == 1) | (pauli == 3)) & has_error
    z_error = ((pauli == 2) | (pauli == 3)) & has_error
    return x_error.astype(np.uint8), z_error.astype(np.uint8)


@dataclass(frozen=True)
class TannerGraph:
    """
    Bipartite Tanner graph representation of a parity-check matrix.

    A Tanner graph has two kinds of nodes:

    * qubit nodes, one for each physical qubit,
    * check nodes, one for each stabilizer/parity check.

    An edge connects qubit ``q`` to check ``c`` exactly when
    ``check_matrix[c, q] = 1``.

    Node ids are arranged as:

    ``0 ... n_qubits - 1``
        Physical-qubit nodes.
    ``n_qubits ... n_qubits + n_checks - 1``
        Stabilizer-check nodes.

    Parameters
    ----------
    n_qubits : int
        Number of physical qubits.
    n_checks : int
        Number of parity/stabilizer checks.
    edge_index : torch.Tensor
        Directed edges with shape ``(2, n_edges)``.  Both graph directions are
        included so message passing can move information both ways.
    check_matrix : scipy.sparse.csr_matrix
        Binary parity-check matrix used to build the graph.
    """

    n_qubits: int
    n_checks: int
    edge_index: Tensor
    check_matrix: sparse.csr_matrix

    @property
    def n_nodes(self) -> int:
        return self.n_qubits + self.n_checks

    @classmethod
    def from_check_matrix(cls, check_matrix, device: Optional[torch.device] = None) -> "TannerGraph":
        """
        Build a Tanner graph from a binary parity-check matrix.

        Parameters
        ----------
        check_matrix : array-like or scipy.sparse matrix
            Parity-check matrix ``H``.
        device : torch.device or None, default=None
            Optional device for the returned edge tensor.

        Returns
        -------
        TannerGraph
            Graph representation of ``H``.
        """
        h = to_csr_binary(check_matrix)
        check_ids, qubit_ids = h.nonzero()

        src_qubits = torch.as_tensor(qubit_ids, dtype=torch.long)
        dst_checks = torch.as_tensor(h.shape[1] + check_ids, dtype=torch.long)

        # Add both directions so simple message passing can move information
        # from checks to qubits and back.
        src = torch.cat([src_qubits, dst_checks], dim=0)
        dst = torch.cat([dst_checks, src_qubits], dim=0)
        edge_index = torch.stack([src, dst], dim=0)
        if device is not None:
            edge_index = edge_index.to(device)

        return cls(
            n_qubits=h.shape[1],
            n_checks=h.shape[0],
            edge_index=edge_index,
            check_matrix=h,
        )

    def to(self, device: torch.device) -> "TannerGraph":
        """
        Move the graph edge tensor to a PyTorch device.

        Parameters
        ----------
        device : torch.device
            Target device, for example ``torch.device("cuda")``.

        Returns
        -------
        TannerGraph
            New graph object with ``edge_index`` on ``device``.
        """
        return TannerGraph(
            n_qubits=self.n_qubits,
            n_checks=self.n_checks,
            edge_index=self.edge_index.to(device),
            check_matrix=self.check_matrix,
        )


class CSSCorrectionEnv:
    """
    Reinforcement-learning environment for one CSS decoding channel.

    The environment exposes syndrome decoding as a small Markov decision
    process.  The current state is the syndrome.  An action either flips one
    physical qubit in the proposed correction or chooses the final ``STOP``
    action.

    For CSS codes:

    * pass ``Hx`` to learn Z-error correction, because X checks detect Z errors,
    * pass ``Hz`` to learn X-error correction, because Z checks detect X errors.

    Parameters
    ----------
    check_matrix : array-like or scipy.sparse matrix
        Parity-check matrix for one CSS channel.
    max_steps : int or None, default=None
        Maximum number of qubit-flip actions before the episode terminates.
    error_sampler : callable or None, default=None
        Function ``error_sampler(n_qubits, rng) -> binary_error``.  If omitted,
        independent component errors are sampled with probability ``p_error``.
    p_error : float, default=0.02
        Component-error probability used by the default sampler.
    seed : int or None, default=None
        Random seed.
    step_penalty : float, default=-0.02
        Reward added for each qubit-flip step.
    flip_reward_scale : float, default=0.2
        Reward multiplier for reducing syndrome weight.  If a flip reduces the
        syndrome weight by 3, the shaped reward contribution is
        ``3 * flip_reward_scale``.
    success_reward : float, default=2.0
        Terminal reward when the syndrome is cleared.
    failure_reward : float, default=-1.0
        Terminal reward when the episode ends without clearing the syndrome.
    allow_stop_when_nonzero : bool, default=True
        If ``False``, the action mask forbids the STOP action until the
        syndrome is already zero.  This prevents a policy from learning the bad
        shortcut of stopping early.
    """

    def __init__(
        self,
        check_matrix,
        *,
        max_steps: Optional[int] = None,
        error_sampler: Optional[Sampler] = None,
        p_error: float = 0.02,
        seed: Optional[int] = None,
        step_penalty: float = -0.02,
        flip_reward_scale: float = 0.2,
        success_reward: float = 2.0,
        failure_reward: float = -1.0,
        allow_stop_when_nonzero: bool = True,
    ) -> None:
        self.h = to_csr_binary(check_matrix)
        self.graph = TannerGraph.from_check_matrix(self.h)
        self.n_checks, self.n_qubits = self.h.shape
        self.max_steps = max_steps or max(4, min(4 * self.n_checks, self.n_qubits))
        self.error_sampler = error_sampler
        self.p_error = p_error
        self.rng = np.random.default_rng(seed)
        self.step_penalty = step_penalty
        self.flip_reward_scale = flip_reward_scale
        self.success_reward = success_reward
        self.failure_reward = failure_reward
        self.allow_stop_when_nonzero = allow_stop_when_nonzero

        # Dense columns are handy for fast syndrome updates in small/medium
        # notebook experiments.
        self._columns = [self.h[:, q].toarray().ravel().astype(np.uint8) for q in range(self.n_qubits)]

        self.error = np.zeros(self.n_qubits, dtype=np.uint8)
        self.correction = np.zeros(self.n_qubits, dtype=np.uint8)
        self.syndrome = np.zeros(self.n_checks, dtype=np.uint8)
        self.initial_syndrome = np.zeros(self.n_checks, dtype=np.uint8)
        self.steps = 0
        self.done = False

    @property
    def stop_action(self) -> int:
        """Index of the special action that means "stop decoding now"."""
        return self.n_qubits

    @property
    def n_actions(self) -> int:
        """Number of available actions: one per qubit plus STOP."""
        return self.n_qubits + 1

    def sample_error(self) -> np.ndarray:
        """
        Sample a new component error for one episode.

        Returns
        -------
        numpy.ndarray
            Binary error vector for the current CSS channel.
        """
        if self.error_sampler is not None:
            return self.error_sampler(self.n_qubits, self.rng).astype(np.uint8)
        return random_component_error(self.n_qubits, self.rng, p=self.p_error)

    def reset(
        self,
        *,
        error: Optional[np.ndarray] = None,
        syndrome: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Start a new decoding episode.

        Parameters
        ----------
        error : numpy.ndarray or None, default=None
            Optional binary component-error vector.  If supplied, the syndrome
            is computed from this error.
        syndrome : numpy.ndarray or None, default=None
            Optional syndrome vector.  Use this when you want to decode a
            measured syndrome without knowing the underlying error.  In that
            case ``self.error`` is set to zero, so exact-residual diagnostics
            are not meaningful.

        Returns
        -------
        numpy.ndarray
            Initial syndrome observation.
        """

        self.correction = np.zeros(self.n_qubits, dtype=np.uint8)
        self.steps = 0
        self.done = False

        if syndrome is not None:
            self.error = np.zeros(self.n_qubits, dtype=np.uint8)
            self.syndrome = syndrome.astype(np.uint8).copy()
        else:
            self.error = self.sample_error() if error is None else error.astype(np.uint8).copy()
            self.syndrome = binary_syndrome(self.h, self.error)

        self.initial_syndrome = self.syndrome.copy()
        return self.syndrome.copy()

    def node_features(self, device: Optional[torch.device] = None) -> Tensor:
        """
        Construct Tanner-graph node features for the current state.

        The returned tensor has one row per Tanner-graph node.  The feature
        columns are:

        ``0``
            One for qubit nodes, zero otherwise.
        ``1``
            One for check nodes, zero otherwise.
        ``2``
            Current syndrome bit on check nodes.  Qubit nodes have zero here.
        ``3``
            Normalized decoding step, ``steps / max_steps``.

        Parameters
        ----------
        device : torch.device or None, default=None
            Optional target device.

        Returns
        -------
        torch.Tensor
            Node-feature tensor with shape ``(n_nodes, 4)``.
        """

        features = np.zeros((self.graph.n_nodes, 4), dtype=np.float32)
        features[: self.n_qubits, 0] = 1.0
        features[self.n_qubits :, 1] = 1.0
        features[self.n_qubits :, 2] = self.syndrome.astype(np.float32)
        features[:, 3] = self.steps / max(1, self.max_steps)
        tensor = torch.from_numpy(features)
        return tensor.to(device) if device is not None else tensor

    def action_mask(self, device: Optional[torch.device] = None) -> Tensor:
        """
        Return a boolean mask over currently valid actions.

        Parameters
        ----------
        device : torch.device or None, default=None
            Optional target device.

        Returns
        -------
        torch.Tensor
            Boolean vector of length ``n_qubits + 1``.  Invalid actions are
            marked ``False`` and are masked out before sampling from the policy.
        """

        mask = torch.ones(self.n_actions, dtype=torch.bool)
        if self.done:
            mask[:-1] = False
        elif not self.allow_stop_when_nonzero and int(self.syndrome.sum()) > 0:
            mask[self.stop_action] = False
        elif not self.allow_stop_when_nonzero and int(self.syndrome.sum()) == 0:
            mask[: self.stop_action] = False
        return mask.to(device) if device is not None else mask

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, object]]:
        """
        Apply one decoder action.

        Parameters
        ----------
        action : int
            Qubit index to flip in the proposed correction, or
            ``env.stop_action`` to stop.

        Returns
        -------
        tuple
            ``(syndrome, reward, done, info)`` where:

            * ``syndrome`` is the updated syndrome,
            * ``reward`` is the scalar RL reward,
            * ``done`` says whether the episode ended,
            * ``info`` contains diagnostics such as syndrome weight, success,
              and residual error weight.
        """
        if self.done:
            raise RuntimeError("Episode already finished. Call reset().")

        old_weight = int(self.syndrome.sum())
        info: Dict[str, object] = {"action": int(action), "old_syndrome_weight": old_weight}

        if action == self.stop_action:
            self.done = True
            success = old_weight == 0
            exact = bool(np.array_equal(self.correction, self.error)) if self.error.any() else success
            reward = self.success_reward if success else self.failure_reward
            info.update(
                {
                    "stopped": True,
                    "success": success,
                    "exact_correction": exact,
                    "residual_weight": int(((self.error + self.correction) % 2).sum()),
                }
            )
            return self.syndrome.copy(), reward, self.done, info

        if action < 0 or action >= self.n_qubits:
            raise ValueError(f"Action must be 0..{self.stop_action}, got {action}.")

        self.correction[action] ^= 1
        self.syndrome ^= self._columns[action]
        self.steps += 1

        new_weight = int(self.syndrome.sum())
        reward = self.step_penalty + self.flip_reward_scale * (old_weight - new_weight)

        if new_weight == 0 or self.steps >= self.max_steps:
            self.done = True
            reward += self.success_reward if new_weight == 0 else self.failure_reward

        info.update(
            {
                "stopped": False,
                "new_syndrome_weight": new_weight,
                "success": new_weight == 0,
                "residual_weight": int(((self.error + self.correction) % 2).sum()),
            }
        )
        return self.syndrome.copy(), float(reward), self.done, info


class GraphMessagePassing(nn.Module):
    """
    One message-passing layer on the Tanner graph.

    Each node sends a learned message to its neighbors.  Incoming messages are
    averaged, combined with the node's current embedding, and added back through
    a residual connection.

    Parameters
    ----------
    hidden_dim : int
        Dimension of every node embedding.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        Run one round of graph message passing.

        Parameters
        ----------
        x : torch.Tensor
            Node embeddings with shape ``(n_nodes, hidden_dim)``.
        edge_index : torch.Tensor
            Directed graph edges with shape ``(2, n_edges)``.

        Returns
        -------
        torch.Tensor
            Updated node embeddings.
        """
        source_nodes, destination_nodes = edge_index
        messages = self.message(x[source_nodes])
        averaged_neighbor_messages = torch.zeros_like(x)
        averaged_neighbor_messages.index_add_(0, destination_nodes, messages)

        node_degrees = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        node_degrees.index_add_(0, destination_nodes, torch.ones_like(destination_nodes, dtype=x.dtype))
        averaged_neighbor_messages = averaged_neighbor_messages / node_degrees.clamp_min(1.0).unsqueeze(-1)

        update = self.update(torch.cat([x, averaged_neighbor_messages], dim=-1))
        return self.norm(x + update)


class GraphActorCritic(nn.Module):
    """
    Graph neural actor-critic decoder.

    The actor outputs logits for ``n_qubits + 1`` actions:

    * action ``q`` flips qubit ``q`` in the proposed correction,
    * action ``n_qubits`` is the STOP action.

    The critic outputs a scalar value estimate for the current syndrome state.

    Parameters
    ----------
    graph : TannerGraph
        Tanner graph built from the CSS-channel parity-check matrix.
    node_feature_dim : int, default=4
        Number of input features per graph node.
    hidden_dim : int, default=96
        Hidden embedding dimension.
    num_layers : int, default=4
        Number of message-passing layers.
    """

    def __init__(
        self,
        graph: TannerGraph,
        node_feature_dim: int = 4,
        hidden_dim: int = 96,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        self.graph = graph
        self.n_qubits = graph.n_qubits
        self.n_actions = graph.n_qubits + 1

        self.encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.layers = nn.ModuleList(GraphMessagePassing(hidden_dim) for _ in range(num_layers))
        self.actor = nn.Linear(hidden_dim, 1)
        self.stop_actor = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.critic = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        device = next(self.parameters()).device
        self.graph = self.graph.to(device)
        return result

    def forward(self, node_features: Tensor, action_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Compute action logits and a state-value estimate.

        Parameters
        ----------
        node_features : torch.Tensor
            Tanner-graph node features from ``CSSCorrectionEnv.node_features``.
        action_mask : torch.Tensor or None, default=None
            Boolean mask of valid actions.  Invalid actions receive a very
            negative logit.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(action_logits, value_estimate)``.
        """
        node_embeddings = self.encoder(node_features)
        for layer in self.layers:
            node_embeddings = layer(node_embeddings, self.graph.edge_index.to(node_embeddings.device))

        qubit_embeddings = node_embeddings[: self.n_qubits]
        graph_embedding = node_embeddings.mean(dim=0)

        qubit_logits = self.actor(qubit_embeddings).squeeze(-1)
        stop_logit = self.stop_actor(graph_embedding).squeeze(-1)
        logits = torch.cat([qubit_logits, stop_logit.view(1)], dim=0)

        if action_mask is not None:
            logits = logits.masked_fill(~action_mask.to(logits.device), -1e9)

        value = self.critic(graph_embedding).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def greedy_action(self, node_features: Tensor, action_mask: Optional[Tensor] = None) -> int:
        """
        Choose the highest-probability valid action.

        Parameters
        ----------
        node_features : torch.Tensor
            Current graph node features.
        action_mask : torch.Tensor or None, default=None
            Optional valid-action mask.

        Returns
        -------
        int
            Greedy action index.
        """
        logits, _ = self.forward(node_features, action_mask)
        return int(torch.argmax(logits).item())


@dataclass
class EpisodeTrajectory:
    """
    Data collected from one complete decoder attempt.

    This is the reinforcement-learning training record for one episode.  It is
    sometimes called a "rollout" in RL literature, but "episode trajectory" is
    more explicit: it stores the sequence of policy decisions, value estimates,
    rewards, and diagnostic information until the decoder stops.

    Parameters
    ----------
    log_probs : list[torch.Tensor]
        Log probability of each chosen action under the policy.
    values : list[torch.Tensor]
        Critic value estimate at each visited state.
    rewards : list[float]
        Reward received after each action.
    entropies : list[torch.Tensor]
        Policy entropy at each step, used as an exploration bonus.
    infos : list[dict]
        Per-step diagnostic dictionaries returned by ``CSSCorrectionEnv.step``.
    """

    log_probs: List[Tensor]
    values: List[Tensor]
    rewards: List[float]
    entropies: List[Tensor]
    infos: List[Dict[str, object]]


def collect_episode_trajectory(
    env: CSSCorrectionEnv,
    model: GraphActorCritic,
    *,
    device: Optional[torch.device] = None,
    greedy: bool = False,
) -> EpisodeTrajectory:
    """
    Run one decoding episode and collect training data.

    Parameters
    ----------
    env : CSSCorrectionEnv
        Single-channel CSS decoding environment.
    model : GraphActorCritic
        Policy/value network.
    device : torch.device or None, default=None
        Device used for PyTorch tensors.
    greedy : bool, default=False
        If ``False``, sample actions from the policy distribution.  If
        ``True``, choose the highest-probability action.  Training normally
        uses sampling; evaluation normally uses greedy actions.

    Returns
    -------
    EpisodeTrajectory
        Complete trajectory containing actions' log-probabilities, rewards,
        critic values, entropies, and diagnostics.
    """

    if device is None:
        device = next(model.parameters()).device

    env.reset()
    log_probs: List[Tensor] = []
    values: List[Tensor] = []
    rewards: List[float] = []
    entropies: List[Tensor] = []
    infos: List[Dict[str, object]] = []

    done = False
    while not done:
        features = env.node_features(device)
        mask = env.action_mask(device)
        logits, value = model(features, mask)
        dist = Categorical(logits=logits)
        action_tensor = torch.argmax(logits) if greedy else dist.sample()
        action = int(action_tensor.item())

        _, reward, done, info = env.step(action)

        log_probs.append(dist.log_prob(action_tensor))
        values.append(value)
        rewards.append(float(reward))
        entropies.append(dist.entropy())
        infos.append(info)

    return EpisodeTrajectory(log_probs, values, rewards, entropies, infos)


# Backward-compatible alias used by older notebook cells.
Rollout = EpisodeTrajectory


def collect_rollout(
    env: CSSCorrectionEnv,
    model: GraphActorCritic,
    *,
    device: Optional[torch.device] = None,
    greedy: bool = False,
) -> EpisodeTrajectory:
    """
    Backward-compatible alias for ``collect_episode_trajectory``.

    Older RL texts often say "rollout"; in this project the preferred name is
    "episode trajectory".
    """

    return collect_episode_trajectory(env, model, device=device, greedy=greedy)


def calculate_discounted_returns(rewards: Sequence[float], gamma: float, device: torch.device) -> Tensor:
    """
    Calculate discounted future rewards for one episode.

    Parameters
    ----------
    rewards : sequence[float]
        Rewards from one episode, ordered from first action to last action.
    gamma : float
        Discount factor.  Smaller values emphasize immediate rewards.
    device : torch.device
        Device for the returned tensor.

    Returns
    -------
    torch.Tensor
        Discounted return at every time step.
    """

    returns: List[float] = []
    running = 0.0
    for reward in reversed(rewards):
        running = float(reward) + gamma * running
        returns.append(running)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32, device=device)


def discounted_returns(rewards: Sequence[float], gamma: float, device: torch.device) -> Tensor:
    """Backward-compatible alias for ``calculate_discounted_returns``."""

    return calculate_discounted_returns(rewards, gamma, device)


def compute_actor_critic_loss(
    trajectory: EpisodeTrajectory,
    *,
    gamma: float = 0.99,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
) -> Tuple[Tensor, Dict[str, float]]:
    """
    Compute the actor-critic loss for one episode trajectory.

    The loss has three terms:

    * policy loss: encourages actions that produced higher-than-expected return,
    * value loss: trains the critic to predict discounted return,
    * entropy bonus: keeps the policy from becoming deterministic too early.

    Parameters
    ----------
    trajectory : EpisodeTrajectory
        Data collected from one episode.
    gamma : float, default=0.99
        Reward discount factor.
    value_coef : float, default=0.5
        Weight of the critic/value loss.
    entropy_coef : float, default=0.01
        Weight of the entropy exploration bonus.

    Returns
    -------
    tuple[torch.Tensor, dict]
        Scalar loss tensor and human-readable training metrics.
    """

    if not trajectory.rewards:
        raise ValueError("Cannot compute a loss from an empty episode trajectory.")

    device = trajectory.values[0].device
    returns = calculate_discounted_returns(trajectory.rewards, gamma, device)
    values = torch.stack(trajectory.values)
    log_probs = torch.stack(trajectory.log_probs)
    entropies = torch.stack(trajectory.entropies)

    advantages = returns - values
    policy_loss = -(log_probs * advantages.detach()).mean()
    value_loss = advantages.pow(2).mean()
    entropy_bonus = entropies.mean()
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_bonus

    metrics = {
        "loss": float(loss.detach().cpu()),
        "policy_loss": float(policy_loss.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
        "entropy": float(entropy_bonus.detach().cpu()),
        "episode_return": float(sum(trajectory.rewards)),
        "episode_length": float(len(trajectory.rewards)),
        "success": float(bool(trajectory.infos[-1].get("success", False))),
        "residual_weight": float(trajectory.infos[-1].get("residual_weight", np.nan)),
    }
    return loss, metrics


def actor_critic_loss(
    rollout: EpisodeTrajectory,
    *,
    gamma: float = 0.99,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
) -> Tuple[Tensor, Dict[str, float]]:
    """Backward-compatible alias for ``compute_actor_critic_loss``."""

    return compute_actor_critic_loss(
        rollout,
        gamma=gamma,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
    )


def train_actor_critic(
    env: CSSCorrectionEnv,
    model: GraphActorCritic,
    optimizer: torch.optim.Optimizer,
    *,
    episodes: int = 100,
    gamma: float = 0.99,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    grad_clip: Optional[float] = 1.0,
    device: Optional[torch.device] = None,
) -> List[Dict[str, float]]:
    """
    Train the GNN decoder with actor-critic reinforcement learning.

    Parameters
    ----------
    env : CSSCorrectionEnv
        CSS-channel correction environment.
    model : GraphActorCritic
        Policy/value network to train.
    optimizer : torch.optim.Optimizer
        PyTorch optimizer, for example ``torch.optim.Adam``.
    episodes : int, default=100
        Number of training episodes.
    gamma : float, default=0.99
        Discount factor for future rewards.
    value_coef : float, default=0.5
        Weight of the critic/value loss.
    entropy_coef : float, default=0.01
        Weight of the entropy exploration bonus.
    grad_clip : float or None, default=1.0
        Optional gradient-norm clipping value.
    device : torch.device or None, default=None
        Training device.  If omitted, CUDA is used when available.

    Returns
    -------
    list[dict]
        One metrics dictionary per episode.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    history: List[Dict[str, float]] = []
    for episode in range(episodes):
        trajectory = collect_episode_trajectory(env, model, device=device, greedy=False)
        loss, metrics = compute_actor_critic_loss(
            trajectory,
            gamma=gamma,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        metrics["episode"] = float(episode)
        history.append(metrics)

    return history


@torch.no_grad()
def evaluate_policy(
    env: CSSCorrectionEnv,
    model: GraphActorCritic,
    *,
    episodes: int = 100,
    greedy: bool = True,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Evaluate a trained decoder policy.

    Parameters
    ----------
    env : CSSCorrectionEnv
        Environment to sample evaluation episodes from.
    model : GraphActorCritic
        Trained policy/value network.
    episodes : int, default=100
        Number of evaluation episodes.
    greedy : bool, default=True
        If ``True``, use the most likely action.  If ``False``, sample from the
        policy distribution.
    device : torch.device or None, default=None
        Device for model inputs.

    Returns
    -------
    dict
        Mean return, episode length, syndrome-clearing success rate, and mean
        residual error weight.
    """

    if device is None:
        device = next(model.parameters()).device

    returns: List[float] = []
    lengths: List[int] = []
    successes: List[float] = []
    residuals: List[float] = []

    for _ in range(episodes):
        trajectory = collect_episode_trajectory(env, model, device=device, greedy=greedy)
        returns.append(float(sum(trajectory.rewards)))
        lengths.append(len(trajectory.rewards))
        successes.append(float(bool(trajectory.infos[-1].get("success", False))))
        residuals.append(float(trajectory.infos[-1].get("residual_weight", np.nan)))

    return {
        "episodes": float(episodes),
        "mean_return": float(np.mean(returns)),
        "mean_length": float(np.mean(lengths)),
        "success_rate": float(np.mean(successes)),
        "mean_residual_weight": float(np.nanmean(residuals)),
    }


def choose_greedy_syndrome_reducing_action(env: CSSCorrectionEnv) -> int:
    """
    Choose the qubit flip that most reduces the syndrome weight.

    This is a simple hand-written decoder step.  It tries every single-qubit
    correction and picks the one with the smallest resulting syndrome weight.
    It does not plan ahead, so it can fail when a correct decoding path must
    temporarily keep or increase syndrome weight.

    Parameters
    ----------
    env : CSSCorrectionEnv
        Current decoding environment.

    Returns
    -------
    int
        Qubit action that gives the best immediate syndrome-weight reduction,
        or ``env.stop_action`` if the syndrome is already zero.
    """

    if int(env.syndrome.sum()) == 0:
        return env.stop_action

    best_action = 0
    best_weight = env.n_checks + 1
    for qubit, column in enumerate(env._columns):
        candidate_weight = int((env.syndrome ^ column).sum())
        if candidate_weight < best_weight:
            best_action = qubit
            best_weight = candidate_weight
    return best_action


def greedy_syndrome_action(env: CSSCorrectionEnv) -> int:
    """Backward-compatible alias for ``choose_greedy_syndrome_reducing_action``."""

    return choose_greedy_syndrome_reducing_action(env)


def train_greedy_imitation(
    env: CSSCorrectionEnv,
    model: GraphActorCritic,
    optimizer: torch.optim.Optimizer,
    *,
    batches: int = 100,
    batch_size: int = 16,
    device: Optional[torch.device] = None,
    max_teacher_steps: Optional[int] = None,
) -> List[Dict[str, float]]:
    """
    Warm-start the actor by imitating a greedy syndrome reducer.

    The teacher used here is not an optimal decoder.  It is a cheap local rule:
    choose the qubit flip that most reduces syndrome weight.  This imitation
    phase is useful because pure reinforcement learning often starts with a
    nearly random policy and can learn bad shortcuts, such as stopping too
    early.

    Parameters
    ----------
    env : CSSCorrectionEnv
        Training environment.
    model : GraphActorCritic
        Actor-critic model to warm-start.
    optimizer : torch.optim.Optimizer
        Optimizer used for supervised imitation steps.
    batches : int, default=100
        Number of imitation batches.
    batch_size : int, default=16
        Number of sampled episodes per imitation batch.
    device : torch.device or None, default=None
        Device for model inputs.
    max_teacher_steps : int or None, default=None
        Maximum number of greedy teacher steps per sampled episode.

    Returns
    -------
    list[dict]
        Batch-level imitation metrics.
    """

    if device is None:
        device = next(model.parameters()).device
    model.to(device)
    max_teacher_steps = max_teacher_steps or env.max_steps

    history: List[Dict[str, float]] = []
    for batch in range(batches):
        losses: List[Tensor] = []
        accuracies: List[float] = []
        mean_syndrome_weights: List[float] = []

        for _ in range(batch_size):
            env.reset()
            for _step in range(max_teacher_steps):
                target = choose_greedy_syndrome_reducing_action(env)
                features = env.node_features(device)
                mask = env.action_mask(device)
                logits, _ = model(features, mask)
                target_tensor = torch.tensor(target, dtype=torch.long, device=device)
                loss = nn.functional.cross_entropy(logits.view(1, -1), target_tensor.view(1))
                losses.append(loss)
                accuracies.append(float(torch.argmax(logits).item() == target))
                mean_syndrome_weights.append(float(env.syndrome.sum()))

                _, _, done, _ = env.step(target)
                if done:
                    break

        batch_loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        batch_loss.backward()
        optimizer.step()

        history.append(
            {
                "batch": float(batch),
                "imitation_loss": float(batch_loss.detach().cpu()),
                "teacher_action_accuracy": float(np.mean(accuracies)),
                "mean_teacher_state_syndrome_weight": float(np.mean(mean_syndrome_weights)),
            }
        )

    return history


def create_decoder_model_for_check_matrix(
    check_matrix,
    *,
    hidden_dim: int = 96,
    num_layers: int = 4,
    device: Optional[torch.device] = None,
) -> Tuple[CSSCorrectionEnv, GraphActorCritic]:
    """
    Create a CSS correction environment and matching GNN decoder.

    Parameters
    ----------
    check_matrix : array-like or scipy.sparse matrix
        Parity-check matrix for one CSS channel.
    hidden_dim : int, default=96
        GNN hidden dimension.
    num_layers : int, default=4
        Number of message-passing layers.
    device : torch.device or None, default=None
        Optional device for the model and graph.

    Returns
    -------
    tuple[CSSCorrectionEnv, GraphActorCritic]
        Environment and actor-critic decoder model.
    """

    env = CSSCorrectionEnv(check_matrix)
    graph = env.graph.to(device) if device is not None else env.graph
    model = GraphActorCritic(graph, hidden_dim=hidden_dim, num_layers=num_layers)
    if device is not None:
        model.to(device)
    return env, model


def make_actor_critic_for_check_matrix(
    check_matrix,
    *,
    hidden_dim: int = 96,
    num_layers: int = 4,
    device: Optional[torch.device] = None,
) -> Tuple[CSSCorrectionEnv, GraphActorCritic]:
    """Backward-compatible alias for ``create_decoder_model_for_check_matrix``."""

    return create_decoder_model_for_check_matrix(
        check_matrix,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        device=device,
    )


__all__ = [
    "CSSCorrectionEnv",
    "EpisodeTrajectory",
    "GraphActorCritic",
    "GraphMessagePassing",
    "Rollout",
    "TannerGraph",
    "actor_critic_loss",
    "binary_syndrome",
    "calculate_discounted_returns",
    "choose_greedy_syndrome_reducing_action",
    "collect_episode_trajectory",
    "collect_rollout",
    "compute_actor_critic_loss",
    "create_decoder_model_for_check_matrix",
    "discounted_returns",
    "evaluate_policy",
    "greedy_syndrome_action",
    "make_actor_critic_for_check_matrix",
    "random_component_error",
    "random_pauli_error",
    "train_greedy_imitation",
    "to_csr_binary",
    "train_actor_critic",
]
