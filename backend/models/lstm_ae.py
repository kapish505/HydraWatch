"""
LSTM Autoencoder for anomaly detection in water pressure time series.

HOW IT WORKS (in plain English):
  Imagine a person who watches water pressure readings all day, every day,
  for months. They learn what "normal" looks like — the morning dip, the
  evening recovery, the weekend pattern. Now show them a NEW set of readings.
  If they can easily predict/reconstruct what they see, it's normal.
  If they're surprised (high reconstruction error), something is wrong.

  That's exactly what this LSTM Autoencoder does:
  1. TRAINING: We feed it ONLY normal pressure data (no leaks).
  2. It learns to compress 24 timesteps into a tiny 16-number "summary"
     (the bottleneck) and then reconstruct the original 24 timesteps.
  3. INFERENCE: We feed it new data. If it can't reconstruct well
     (high MSE), that data looks different from normal → anomaly!

Architecture:
  Encoder: LSTM(N_sensors, 64, 2 layers) → last hidden → Linear(64, 16)
  Decoder: Linear(16, 64) → repeat 24 times → LSTM(64, N_sensors) → output

Input shape:  (batch, 24, N_sensors)
Output shape: (batch, 24, N_sensors) — reconstruction of input
Loss: MSE between input and reconstruction
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List


MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"


class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder for time series anomaly detection.

    The "autoencoder" part means it learns to copy its input to its output
    through a narrow bottleneck. If it can copy successfully, the input
    is "normal". If it can't, the input is anomalous.
    """

    def __init__(self, n_sensors: int, hidden_size: int = 64, bottleneck_size: int = 16):
        """
        Args:
            n_sensors: number of pressure sensor nodes (e.g., 32 for Hanoi)
            hidden_size: LSTM hidden dimension (64)
            bottleneck_size: compressed representation size (16)
        """
        super().__init__()

        self.n_sensors = n_sensors
        self.hidden_size = hidden_size
        self.bottleneck_size = bottleneck_size

        # ── Encoder ──────────────────────────────────────────────────
        # Reads the 24-step pressure sequence and summarizes it
        self.encoder_lstm = nn.LSTM(
            input_size=n_sensors,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )
        # Compress the LSTM's final hidden state to the bottleneck
        self.encoder_fc = nn.Linear(hidden_size, bottleneck_size)

        # ── Decoder ──────────────────────────────────────────────────
        # Expands the bottleneck back to LSTM-sized hidden state
        self.decoder_fc = nn.Linear(bottleneck_size, hidden_size)
        # Reconstructs the original sequence from the expanded representation
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=n_sensors,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode → bottleneck → decode.

        Args:
            x: (batch, seq_len, n_sensors) input pressure windows

        Returns:
            (batch, seq_len, n_sensors) reconstructed pressure windows
        """
        batch_size, seq_len, _ = x.shape

        # ENCODE: run LSTM over all 24 timesteps
        # enc_output: (batch, seq_len, hidden_size)
        # h_n: (num_layers, batch, hidden_size) — last hidden state
        _, (h_n, _) = self.encoder_lstm(x)

        # Take the last layer's hidden state
        # h_n[-1] shape: (batch, hidden_size)
        encoded = self.encoder_fc(h_n[-1])  # (batch, bottleneck_size)

        # DECODE: expand bottleneck and repeat for each timestep
        decoded = self.decoder_fc(encoded)  # (batch, hidden_size)

        # Repeat the decoded vector for each timestep in the sequence
        # Shape: (batch, seq_len, hidden_size)
        decoded_repeated = decoded.unsqueeze(1).repeat(1, seq_len, 1)

        # Run decoder LSTM to produce the reconstructed sequence
        # output: (batch, seq_len, n_sensors)
        output, _ = self.decoder_lstm(decoded_repeated)

        return output

    def get_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-window anomaly score = mean MSE of reconstruction.

        A higher score means the model was more "surprised" by this input.

        Args:
            x: (batch, seq_len, n_sensors)

        Returns:
            (batch,) anomaly scores
        """
        with torch.no_grad():
            reconstruction = self.forward(x)
            # MSE per element, then mean across time and sensors
            mse = ((x - reconstruction) ** 2).mean(dim=(1, 2))  # (batch,)
        return mse


class LSTMAutoencoderTrainer:
    """
    Training harness for the LSTM Autoencoder.

    Key design decisions:
    - Train ONLY on normal data (no leak windows)
    - Threshold = 99th percentile of reconstruction error on held-out normal data
    - This means ~1% of normal data will be flagged (acceptable false alarm rate)
    """

    def __init__(
        self,
        n_sensors: int,
        hidden_size: int = 64,
        bottleneck_size: int = 16,
        lr: float = 1e-3,
        device: str = "auto",
    ):
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model = LSTMAutoencoder(n_sensors, hidden_size, bottleneck_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=5, factor=0.5
        )
        self.threshold = 0.0  # Set after training
        self.n_sensors = n_sensors

        self._normalization = {"mean": None, "std": None}
        self._reference_sample: Optional[np.ndarray] = None  # stored during training for ONNX export

    def train(
        self,
        train_windows: np.ndarray,
        val_windows: np.ndarray,
        epochs: int = 50,
        batch_size: int = 64,
        verbose: bool = True,
        val_all_windows: Optional[np.ndarray] = None,
        val_all_labels: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Train the autoencoder on normal-only windows.

        Args:
            train_windows: (N_train, 24, N_sensors) normal-only training data
            val_windows: (N_val, 24, N_sensors) normal-only validation data
            epochs: number of training epochs
            batch_size: mini-batch size

        Returns:
            Dict of training metrics
        """
        # Convert to tensors
        train_tensor = torch.tensor(train_windows, dtype=torch.float32)
        val_tensor = torch.tensor(val_windows, dtype=torch.float32)

        train_dataset = torch.utils.data.TensorDataset(train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )

        if verbose:
            print(f"  Training LSTM Autoencoder on {len(train_windows)} normal windows")
            print(f"  Validation: {len(val_windows)} normal windows")
            print(f"  Device: {self.device}")
            params = sum(p.numel() for p in self.model.parameters())
            print(f"  Parameters: {params:,}")

        criterion = nn.MSELoss()
        best_val_loss = float("inf")
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(1, epochs + 1):
            # ── Training ──────────────────────────────────────────────
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for (batch_x,) in train_loader:
                batch_x = batch_x.to(self.device)

                self.optimizer.zero_grad()
                reconstruction = self.model(batch_x)
                loss = criterion(reconstruction, batch_x)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)
            history["train_loss"].append(avg_train_loss)

            # ── Validation ────────────────────────────────────────────
            self.model.eval()
            val_loss_sum = 0.0
            n_val_batches = 0
            with torch.no_grad():
                val_loader = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(val_tensor),
                    batch_size=batch_size, shuffle=False
                )
                for (batch_v,) in val_loader:
                    batch_v = batch_v.to(self.device)
                    val_recon = self.model(batch_v)
                    val_loss_sum += criterion(val_recon, batch_v).item()
                    n_val_batches += 1
            
            val_loss = val_loss_sum / max(n_val_batches, 1)
            history["val_loss"].append(val_loss)

            self.scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save best model state
                self._best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            if verbose and (epoch % 5 == 0 or epoch == 1):
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"  Epoch {epoch:3d} | Train Loss: {avg_train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f} | LR: {lr:.2e}")

        # Restore best model
        if hasattr(self, "_best_state"):
            self.model.load_state_dict(self._best_state)

        # ── Set threshold ─────────────────────────────────────────────
        if val_all_windows is not None and val_all_labels is not None:
            self._set_threshold(val_tensor, val_all_windows, val_all_labels)
        else:
            self._set_threshold(val_tensor)

        if verbose:
            print(f"\n  Best val loss: {best_val_loss:.6f}")
            print(f"  Anomaly threshold (99th percentile): {self.threshold:.6f}")

        # Store a representative sample from validation for ONNX export
        self._reference_sample = val_windows[:1].copy()  # (1, 24, N)

        return {
            "best_val_loss": best_val_loss,
            "threshold": self.threshold,
            "history": history,
        }

    def _set_threshold(
        self, 
        val_windows_normal: torch.Tensor,
        val_all_windows: Optional[np.ndarray] = None,
        val_all_labels: Optional[np.ndarray] = None,
    ):
        """
        Set the anomaly threshold.
        
        If validation leak data is provided, it uses the precision-recall curve
        to mathematically find the exact threshold maximizing the F1 score.
        Otherwise, it falls back to the 95th percentile of normal data.
        """
        self.model.eval()
        
        if val_all_windows is not None and val_all_labels is not None:
            # Optimize F1 score explicitly
            predict_results = self.predict(val_all_windows)
            scores = predict_results["anomaly_scores"]
            
            from sklearn.metrics import precision_recall_curve
            precisions, recalls, thresholds = precision_recall_curve(val_all_labels, scores)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
            best_idx = np.argmax(f1_scores)
            
            if best_idx < len(thresholds):
                self.threshold = float(thresholds[best_idx])
            else:
                self.threshold = 1.0
            return

        # Fallback 95th percentile logic
        all_scores = []
        with torch.no_grad():
            dl = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(val_windows_normal),
                batch_size=2048, shuffle=False
            )
            for (bx,) in dl:
                bx = bx.to(self.device)
                scores = self.model.get_anomaly_score(bx).cpu().numpy()
                all_scores.append(scores)

        if all_scores:
            scores_concat = np.concatenate(all_scores, axis=0)
            self.threshold = float(np.percentile(scores_concat, 95))
        else:
            self.threshold = 1.0

    def predict(
        self,
        windows: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Run inference on pressure windows.

        Args:
            windows: (N, 24, N_sensors) pressure windows (already normalized)

        Returns:
            Dict with:
              'anomaly_scores': (N,) raw MSE scores
              'normalized_scores': (N,) scores in [0, 1] range
              'is_anomaly': (N,) boolean predictions
        """
        self.model.eval()
        tensor = torch.tensor(windows, dtype=torch.float32)
        all_scores = []

        with torch.no_grad():
            dl = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(tensor),
                batch_size=2048, shuffle=False
            )
            for (bx,) in dl:
                bx = bx.to(self.device)
                s = self.model.get_anomaly_score(bx).cpu().numpy()
                all_scores.append(s)

        scores = np.concatenate(all_scores, axis=0) if all_scores else np.array([])

        # Normalize to 0-1 using threshold as reference
        # Score of threshold → ~0.85 (the alert boundary)
        max_score = max(self.threshold * 3, scores.max() + 1e-8)
        normalized = np.clip(scores / max_score, 0, 1)

        return {
            "anomaly_scores": scores,
            "normalized_scores": normalized,
            "is_anomaly": scores > self.threshold,
        }

    def save(self, path: Optional[str] = None) -> str:
        """Save model and threshold to disk."""
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        if path is None:
            path = str(MODEL_DIR / "lstm_autoencoder.pt")

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "n_sensors": self.n_sensors,
            "hidden_size": self.model.hidden_size,
            "bottleneck_size": self.model.bottleneck_size,
            "threshold": self.threshold,
            "reference_sample": self._reference_sample,
        }, path)

        print(f"  LSTM Autoencoder saved to {path}")

        # Also export to ONNX using a real data sample
        onnx_path = str(MODEL_DIR / "lstm_autoencoder.onnx")
        self.model.eval()
        self.model.to("cpu")

        # Use stored reference sample from training; fall back to zeros
        # with correct shape if no reference is available (e.g. loaded model)
        if self._reference_sample is not None:
            trace_input = torch.tensor(self._reference_sample[:1], dtype=torch.float32)
        else:
            trace_input = torch.zeros(1, 24, self.n_sensors, dtype=torch.float32)

        try:
            torch.onnx.export(
                self.model, trace_input, onnx_path,
                input_names=["pressure_window"],
                output_names=["reconstruction"],
                dynamic_axes={
                    "pressure_window": {0: "batch_size"},
                    "reconstruction": {0: "batch_size"},
                },
                opset_version=14,
            )
            print(f"  ONNX exported to {onnx_path}")

            # Validate: compare ONNX output to PyTorch output on the same input
            try:
                import onnxruntime as ort
                sess = ort.InferenceSession(onnx_path)
                onnx_out = sess.run(None, {"pressure_window": trace_input.numpy()})[0]
                with torch.no_grad():
                    pt_out = self.model(trace_input).numpy()
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
    def load(cls, path: Optional[str] = None, device: str = "auto") -> "LSTMAutoencoderTrainer":
        """Load a trained model from disk."""
        if path is None:
            path = str(MODEL_DIR / "lstm_autoencoder.pt")

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        trainer = cls(
            n_sensors=checkpoint["n_sensors"],
            hidden_size=checkpoint["hidden_size"],
            bottleneck_size=checkpoint["bottleneck_size"],
            device=device,
        )
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.threshold = checkpoint["threshold"]
        trainer._reference_sample = checkpoint.get("reference_sample", None)

        return trainer
