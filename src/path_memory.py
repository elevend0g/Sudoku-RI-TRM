"""
Hebbian Path Memory - K_P (Layer 3) in RI-TRM architecture.

Learns which fixes work for which violations through experience.
Implements biologically-inspired learning:
- Long-Term Potentiation (LTP): Successful paths strengthen
- Long-Term Depression (LTD): Failed paths weaken
- Myelination: Heavily-used paths get boosted
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, FrozenSet
import numpy as np


@dataclass
class Path:
    """
    Represents a reasoning path: state -> action -> result.

    Attributes:
        violation_pattern: Frozenset of violation types at this state
        action: Description of the action taken
        result: Whether violations decreased
        weight: Path strength (success rate), 0-1
        usage_count: Number of times this path has been used
    """
    violation_pattern: FrozenSet[str]
    action: str
    result: bool
    weight: float = 0.5
    usage_count: int = 0

    def __repr__(self):
        return f"Path(pattern={self.violation_pattern}, action={self.action}, weight={self.weight:.3f}, uses={self.usage_count})"

    def __hash__(self):
        return hash((self.violation_pattern, self.action))

    def __eq__(self, other):
        if not isinstance(other, Path):
            return False
        return self.violation_pattern == other.violation_pattern and self.action == other.action


class HebbianPathMemory:
    """
    Hebbian learning for path memory.

    Implements:
    - Long-Term Potentiation (LTP) for successful paths
    - Long-Term Depression (LTD) for failed paths
    - Myelination boost for heavily-used paths
    - ε-greedy exploration/exploitation
    """

    def __init__(
        self,
        alpha: float = 0.1,  # Learning rate
        gamma: float = 0.95,  # Decay rate for failures
        beta: float = 1.1,  # Myelination boost
        theta_myelination: int = 10,  # Usage threshold for myelination
        epsilon: float = 0.3,  # Exploration rate
        epsilon_decay: float = 0.99,  # Decay per epoch
        min_epsilon: float = 0.05,  # Minimum exploration
        min_weight: float = 0.01,  # Minimum path weight
        max_weight: float = 0.99,  # Maximum path weight
    ):
        """
        Initialize Hebbian path memory.

        Args:
            alpha: Learning rate for LTP (typically 0.1)
            gamma: Decay rate for LTD (typically 0.95)
            beta: Myelination boost factor (typically 1.1)
            theta_myelination: Usage threshold for myelination (typically 10)
            epsilon: Initial exploration rate
            epsilon_decay: Decay factor for epsilon per epoch
            min_epsilon: Minimum epsilon value
            min_weight: Minimum path weight (prevents deletion)
            max_weight: Maximum path weight (prevents overflow)
        """
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.theta_myelination = theta_myelination
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.min_weight = min_weight
        self.max_weight = max_weight

        # Storage: (violation_pattern, action) -> Path
        self.paths: Dict[Tuple[FrozenSet[str], str], Path] = {}

        # Statistics
        self.total_updates = 0
        self.successful_updates = 0
        self.failed_updates = 0

    def query(
        self,
        violations: List,  # List of Violation objects
        top_k: int = 5
    ) -> List[Path]:
        """
        Query path memory for candidate actions.

        Args:
            violations: List of current violations
            top_k: Number of top paths to return

        Returns:
            List of candidate paths sorted by weight (descending)
        """
        if len(violations) == 0:
            return []

        # Create violation pattern
        violation_pattern = frozenset(v.type for v in violations)

        # Find all paths matching this pattern
        matching_paths = [
            path for (pattern, action), path in self.paths.items()
            if pattern == violation_pattern
        ]

        # Sort by weight (descending)
        matching_paths.sort(key=lambda p: p.weight, reverse=True)

        return matching_paths[:top_k]

    def update(
        self,
        violation_pattern: FrozenSet[str],
        action: str,
        success: bool
    ) -> Path:
        """
        Update path memory based on outcome.

        Implements Hebbian learning:
        - If success: LTP (strengthen path)
        - If failure: LTD (weaken path)
        - If heavily used: Myelination (boost path)

        Args:
            violation_pattern: Frozenset of violation types
            action: Action that was taken
            success: Whether the action reduced violations

        Returns:
            Updated path object
        """
        self.total_updates += 1

        key = (violation_pattern, action)

        # Get or create path
        if key in self.paths:
            path = self.paths[key]
        else:
            path = Path(
                violation_pattern=violation_pattern,
                action=action,
                result=success,
                weight=0.5,
                usage_count=0
            )
            self.paths[key] = path

        # Update weight based on outcome
        if success:
            # Long-Term Potentiation (LTP)
            path.weight = path.weight + self.alpha * (1 - path.weight)
            path.usage_count += 1
            self.successful_updates += 1

            # Myelination: boost heavily-used successful paths
            if path.usage_count >= self.theta_myelination:
                path.weight = min(path.weight * self.beta, self.max_weight)

        else:
            # Long-Term Depression (LTD)
            path.weight = path.weight * self.gamma
            self.failed_updates += 1

        # Clamp weight to valid range
        path.weight = np.clip(path.weight, self.min_weight, self.max_weight)

        # Update result
        path.result = success

        return path

    def select_path(
        self,
        candidates: List[Path],
        epsilon: Optional[float] = None
    ) -> Optional[Path]:
        """
        Select a path using ε-greedy strategy.

        Args:
            candidates: List of candidate paths
            epsilon: Exploration rate (uses self.epsilon if None)

        Returns:
            Selected path or None if no candidates
        """
        if len(candidates) == 0:
            return None

        if epsilon is None:
            epsilon = self.epsilon

        # ε-greedy selection
        if np.random.random() < epsilon:
            # Explore: random selection
            return np.random.choice(candidates)
        else:
            # Exploit: select highest weight
            return max(candidates, key=lambda p: p.weight)

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

    def get_statistics(self) -> Dict:
        """
        Get memory statistics.

        Returns:
            Dictionary with statistics
        """
        if len(self.paths) == 0:
            return {
                "num_paths": 0,
                "avg_weight": 0.0,
                "max_weight": 0.0,
                "min_weight": 0.0,
                "avg_usage": 0.0,
                "strong_paths": 0,
                "myelinated_paths": 0,
                "total_updates": self.total_updates,
                "successful_updates": self.successful_updates,
                "failed_updates": self.failed_updates,
                "success_rate": 0.0,
                "epsilon": self.epsilon,
            }

        weights = [path.weight for path in self.paths.values()]
        usages = [path.usage_count for path in self.paths.values()]

        # Count strong paths (weight > 0.8)
        strong_paths = sum(1 for w in weights if w > 0.8)

        # Count myelinated paths
        myelinated = sum(1 for u in usages if u >= self.theta_myelination)

        return {
            "num_paths": len(self.paths),
            "avg_weight": np.mean(weights),
            "max_weight": np.max(weights),
            "min_weight": np.min(weights),
            "std_weight": np.std(weights),
            "avg_usage": np.mean(usages),
            "max_usage": np.max(usages),
            "strong_paths": strong_paths,
            "myelinated_paths": myelinated,
            "total_updates": self.total_updates,
            "successful_updates": self.successful_updates,
            "failed_updates": self.failed_updates,
            "success_rate": self.successful_updates / self.total_updates if self.total_updates > 0 else 0.0,
            "epsilon": self.epsilon,
        }

    def get_top_paths(self, n: int = 10) -> List[Path]:
        """
        Get top N strongest paths.

        Args:
            n: Number of paths to return

        Returns:
            List of top paths sorted by weight
        """
        all_paths = list(self.paths.values())
        all_paths.sort(key=lambda p: p.weight, reverse=True)
        return all_paths[:n]

    def get_pattern_diversity(self) -> int:
        """
        Get number of unique violation patterns in memory.

        Returns:
            Number of unique patterns
        """
        patterns = set(path.violation_pattern for path in self.paths.values())
        return len(patterns)

    def prune_weak_paths(self, threshold: float = 0.1):
        """
        Remove paths below weight threshold.

        Args:
            threshold: Weight threshold for pruning
        """
        keys_to_remove = [
            key for key, path in self.paths.items()
            if path.weight < threshold
        ]

        for key in keys_to_remove:
            del self.paths[key]

        return len(keys_to_remove)

    def save_to_dict(self) -> Dict:
        """
        Serialize path memory to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "hyperparameters": {
                "alpha": self.alpha,
                "gamma": self.gamma,
                "beta": self.beta,
                "theta_myelination": self.theta_myelination,
                "epsilon": self.epsilon,
                "min_weight": self.min_weight,
                "max_weight": self.max_weight,
            },
            "paths": [
                {
                    "violation_pattern": list(path.violation_pattern),
                    "action": path.action,
                    "weight": path.weight,
                    "usage_count": path.usage_count,
                    "result": path.result,
                }
                for path in self.paths.values()
            ],
            "statistics": self.get_statistics(),
        }

    @classmethod
    def load_from_dict(cls, data: Dict) -> "HebbianPathMemory":
        """
        Load path memory from dictionary.

        Args:
            data: Dictionary from save_to_dict()

        Returns:
            Loaded HebbianPathMemory
        """
        # Create instance with saved hyperparameters
        hyperparams = data["hyperparameters"]
        memory = cls(
            alpha=hyperparams["alpha"],
            gamma=hyperparams["gamma"],
            beta=hyperparams["beta"],
            theta_myelination=hyperparams["theta_myelination"],
            epsilon=hyperparams["epsilon"],
            min_weight=hyperparams["min_weight"],
            max_weight=hyperparams["max_weight"],
        )

        # Load paths
        for path_data in data["paths"]:
            pattern = frozenset(path_data["violation_pattern"])
            action = path_data["action"]

            path = Path(
                violation_pattern=pattern,
                action=action,
                result=path_data["result"],
                weight=path_data["weight"],
                usage_count=path_data["usage_count"],
            )

            key = (pattern, action)
            memory.paths[key] = path

        # Restore statistics if available
        if "statistics" in data:
            stats = data["statistics"]
            memory.total_updates = stats.get("total_updates", 0)
            memory.successful_updates = stats.get("successful_updates", 0)
            memory.failed_updates = stats.get("failed_updates", 0)

        return memory

    def __len__(self):
        """Return number of paths in memory."""
        return len(self.paths)

    def __repr__(self):
        stats = self.get_statistics()
        return (
            f"HebbianPathMemory("
            f"paths={stats['num_paths']}, "
            f"avg_weight={stats['avg_weight']:.3f}, "
            f"success_rate={stats['success_rate']:.3f}, "
            f"epsilon={stats['epsilon']:.3f})"
        )
