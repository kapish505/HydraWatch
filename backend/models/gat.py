"""
Graph Attention Network (GAT) for leak LOCALISATION.

WHY A GAT? (in plain English):
  A water pipe network is literally a graph — junctions connected by pipes.
  When a pipe leaks, pressure drops at nearby junctions but NOT distant ones.
  A GAT learns which neighbor's information matters most via "attention heads"
  (like the model paying extra attention to the most relevant neighbors).

  The key difference from the old GNN code:
  - OLD: Graph-level classification (is there A leak somewhere? → single yes/no)
  - NEW: Node-level classification (WHICH node has the leak? → probability per node)

  This means we get a leak probability map across the whole network,
  letting us say "the leak is most likely at Node 19" (localisation).

Architecture (matches spec exactly):
  - 3× GATConv layers:
      GATConv(in=3, out=32, heads=4, concat=True)   → output: 128
      GATConv(in=128, out=32, heads=4, concat=True)  → output: 128
      GATConv(in=128, out=32, heads=4, concat=True)  → output: 128
  - Between layers: ELU activation + Dropout(0.3)
  - Final: Linear(128, 1) + Sigmoid → per-node leak probability

Node features (3 per node):
  1. Current pressure (normalized)
  2. Rolling mean pressure 6h (normalized)
  3. Pressure z-score

Edge features used for graph construction:
  - diameter, length, roughness (from WNTR .inp file)

Loss: Binary cross-entropy with positive class weight = 10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

# PyG import
try:
    from torch_geometric.nn import GATConv
    from torch_geometric.data import Data, Batch
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("⚠ torch_geometric not installed. GAT model unavailable.")
    print("  Install: pip install torch-geometric")


MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"


class FocalLoss(nn.Module):
    """
    Focal Loss handles extreme class imbalance by down-weighting the well-classified
    majority class (normal pipes) and focusing training on the hard minority class (leaks).
    """
    def __init__(self, alpha: float = 0.9, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)  # pt is the probability of the true class
        # Add alpha weighting for the positive class
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class LeakGAT(nn.Module):
    """
    Graph Attention Network for per-node leak probability prediction.

    Each node gets a probability between 0 and 1:
      0.0 = definitely no leak here
      1.0 = definitely a leak here

    The top-3 highest-probability nodes are reported as "suspect nodes."
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 32,
        heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()

        if not HAS_PYG:
            raise RuntimeError("torch_geometric required for GAT model")

        self.in_channels = in_channels
        self.dropout = dropout

        # Layer 1: (3) → (32 × 4 = 128)
        self.conv1 = GATConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            concat=True,
            dropout=dropout,
        )

        # Layer 2: (128) → (32 × 4 = 128)
        self.conv2 = GATConv(
            in_channels=hidden_channels * heads,
            out_channels=hidden_channels,
            heads=heads,
            concat=True,
            dropout=dropout,
        )

        # Layer 3: (128) → (32 × 4 = 128)
        self.conv3 = GATConv(
            in_channels=hidden_channels * heads,
            out_channels=hidden_channels,
            heads=heads,
            concat=True,
            dropout=dropout,
        )

        # Final classifier: per-node leak probability
        # 128 → 1 → Sigmoid
        self.classifier = nn.Linear(hidden_channels * heads, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass: 3 GAT layers → per-node leak probability.

        Args:
            x: (N, 3) node features
            edge_index: (2, E) edge connectivity

        Returns:
            (N,) per-node leak probabilities in [0, 1]
        """
        # Layer 1
        h = self.conv1(x, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Layer 2
        h = self.conv2(h, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Layer 3
        h = self.conv3(h, edge_index)
        h = F.elu(h)

        # Per-node classification
        logits = self.classifier(h).squeeze(-1)  # (N,)

        if self.training:
            return logits  # raw logits for BCEWithLogitsLoss (numerically stable)
        else:
            return torch.sigmoid(logits)  # probabilities for inference/ranking


class GATTrainer:
    """
    Training harness for the GAT leak localiser.

    Training data structure:
      - Each sample is one timestep of one scenario
      - Node features: (N, 3) — current_pressure, rolling_mean_6h, z_score
      - Node labels: (N,) — binary, 1.0 for the leaking node, 0.0 for all others
      - Edge index: same for all samples (fixed pipe network topology)

    The major challenge is class imbalance: at any timestep during a leak,
    only 1 out of ~31 nodes is leaking (~3%). We use pos_weight=30 in the
    loss function to compensate (matching the actual class ratio).
    """

    def __init__(
        self,
        edge_index: np.ndarray,
        n_node_features: int = 3,
        hidden_channels: int = 32,
        heads: int = 4,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "auto",
    ):
        if not HAS_PYG:
            raise RuntimeError("torch_geometric required")

        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.edge_index_np = edge_index
        self.edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device)

        self.model = LeakGAT(
            in_channels=n_node_features,
            hidden_channels=hidden_channels,
            heads=heads,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", patience=5, factor=0.5
        )
        self._reference_node_features: Optional[np.ndarray] = None  # stored for ONNX export

    def train_epoch(
        self,
        node_features_list: List[np.ndarray],
        node_labels_list: List[np.ndarray],
        batch_size: int = 32,
    ) -> float:
        """
        Train for one epoch over multiple timestep samples.

        Each sample is one timestep: (N, 3) features + (N,) labels.

        Args:
            node_features_list: list of (N, 3) arrays, one per timestep
            node_labels_list: list of (N,) arrays, one per timestep
            batch_size: number of timestep graphs per mini-batch

        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        # Use a mild pos_weight. Too high (20+) forces the GNN to destroy its attention
        # weights to suppress false positives on neighboring nodes, killing localization.
        pos_weight = torch.tensor([5.0], device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Shuffle samples
        indices = np.random.permutation(len(node_features_list))

        for batch_start in range(0, len(indices), batch_size):
            batch_indices = indices[batch_start:batch_start + batch_size]

            # Build a batched graph
            data_list = []
            for idx in batch_indices:
                feats = torch.tensor(node_features_list[idx], dtype=torch.float)
                labels = torch.tensor(node_labels_list[idx], dtype=torch.float)
                data = Data(
                    x=feats,
                    edge_index=torch.tensor(self.edge_index_np, dtype=torch.long),
                    y=labels,
                )
                data_list.append(data)

            batch = Batch.from_data_list(data_list).to(self.device)

            self.optimizer.zero_grad()

            # Forward — get raw logits (model returns logits during training)
            logits = self.model(batch.x, batch.edge_index)

            # BCEWithLogitsLoss applies sigmoid internally — no double-sigmoid
            loss = criterion(logits, batch.y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def store_reference_sample(self, node_features_list: List[np.ndarray]):
        """Store the first sample as reference for ONNX export tracing."""
        if node_features_list:
            self._reference_node_features = node_features_list[0].copy()

    def evaluate(
        self,
        node_features_list: List[np.ndarray],
        node_labels_list: List[np.ndarray],
    ) -> Dict[str, float]:
        """
        Evaluate the model: compute F1, precision, recall at node level.
        Also compute localisation accuracy (% of leak timesteps where
        the true leak node is in our top-3 suspects).
        """
        self.model.eval()

        all_preds = []
        all_labels = []
        correct_localisations = 0
        total_leak_timesteps = 0

        with torch.no_grad():
            for feats, labels in zip(node_features_list, node_labels_list):
                x = torch.tensor(feats, dtype=torch.float, device=self.device)
                probs = self.model(x, self.edge_index)  # (N,)
                probs_np = probs.cpu().numpy()

                preds = (probs_np > 0.5).astype(int)
                all_preds.append(preds)
                all_labels.append(labels)

                # Localisation: is the true leak node in top-3?
                if labels.max() > 0:
                    total_leak_timesteps += 1
                    true_leak_idx = np.argmax(labels)
                    top3_indices = np.argsort(probs_np)[-3:]
                    if true_leak_idx in top3_indices:
                        correct_localisations += 1

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)

        from sklearn.metrics import f1_score, precision_score, recall_score

        return {
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "localisation_accuracy": (
                correct_localisations / max(total_leak_timesteps, 1)
            ),
            "total_leak_timesteps": total_leak_timesteps,
        }

    def predict(
        self,
        node_features: np.ndarray,
        node_names: List[str],
    ) -> Dict[str, float]:
        """
        Run inference on a single timestep.

        Args:
            node_features: (N, 3) node feature matrix
            node_names: list of node name strings

        Returns:
            Dict mapping node_name → leak probability
        """
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(node_features, dtype=torch.float, device=self.device)
            probs = self.model(x, self.edge_index).cpu().numpy()

        return {name: float(prob) for name, prob in zip(node_names, probs)}

    def get_top_suspects(
        self,
        node_features: np.ndarray,
        node_names: List[str],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Get the top-K most suspicious nodes.

        Returns:
            List of dicts: [{node_id, probability}, ...]
        """
        predictions = self.predict(node_features, node_names)
        sorted_nodes = sorted(predictions.items(), key=lambda x: x[1], reverse=True)

        return [
            {"node_id": name, "probability": prob}
            for name, prob in sorted_nodes[:top_k]
        ]

    def save(self, path: Optional[str] = None, skip_onnx: bool = False) -> str:
        """Save model checkpoint and optionally export to ONNX."""
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        if path is None:
            path = str(MODEL_DIR / "gat_localiser.pt")

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "edge_index": self.edge_index_np,
            "in_channels": self.model.in_channels,
            "hidden_channels": self.model.conv1.out_channels,
            "reference_node_features": self._reference_node_features,
        }, path)

        print(f"  GAT model saved to {path}")

        if skip_onnx:
            return path

        # ONNX export using real data
        onnx_path = str(MODEL_DIR / "gat_localiser.onnx")
        self.model.eval()
        self.model.to("cpu")
        N = self.edge_index_np.max() + 1  # number of nodes

        # Use stored reference sample from training; fall back to zeros
        if self._reference_node_features is not None:
            trace_x = torch.tensor(self._reference_node_features, dtype=torch.float32)
        else:
            trace_x = torch.zeros(N, self.model.in_channels, dtype=torch.float32)
        trace_edge = torch.tensor(self.edge_index_np, dtype=torch.long)

        try:
            torch.onnx.export(
                self.model, (trace_x, trace_edge), onnx_path,
                input_names=["node_features", "edge_index"],
                output_names=["node_probabilities"],
                opset_version=16,
            )
            print(f"  ONNX exported to {onnx_path}")

            # Validate: compare ONNX output to PyTorch output on same input
            try:
                import onnxruntime as ort
                sess = ort.InferenceSession(onnx_path)
                onnx_out = sess.run(None, {
                    "node_features": trace_x.numpy(),
                    "edge_index": trace_edge.numpy(),
                })[0]
                with torch.no_grad():
                    pt_out = self.model(trace_x, trace_edge).numpy()
                max_diff = float(np.abs(onnx_out - pt_out).max())
                print(f"  ONNX validation: max abs diff = {max_diff:.6e} {'✓' if max_diff < 1e-4 else '⚠ MISMATCH'}")
            except ImportError:
                print("  ℹ onnxruntime not installed — skipping ONNX validation")

        except Exception as e:
            print(f"  ⚠ ONNX export failed: {e}")
        finally:
            self.model.to(self.device)

        return path

    @classmethod
    def load(cls, path: Optional[str] = None, device: str = "auto") -> "GATTrainer":
        """Load a trained model checkpoint."""
        if path is None:
            path = str(MODEL_DIR / "gat_localiser.pt")

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        trainer = cls(
            edge_index=checkpoint["edge_index"],
            n_node_features=checkpoint["in_channels"],
            hidden_channels=checkpoint.get("hidden_channels", 32),
            device=device,
        )
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer._reference_node_features = checkpoint.get("reference_node_features", None)

        return trainer
