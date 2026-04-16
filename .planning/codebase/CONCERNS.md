# Architectural Concerns & Tech Debt

- **Model Training Speed vs Feedback Loop**: `scripts/train.py` builds all models (XGBoost, LSTM Autoencoder, and PyTorch Geometric GAT Localiser). Background execution cycles for the GAT node iterations on standard hardware configurations can easily take a few hours. Changing network feature assumptions heavily bottlenecks quick feedback operations.
- **Memory vs Telemetry State Tracking**: `IngestBuffer` utilizes a constrained double-ended queue. Extreme scale operations processing larger networks (>100k nodes) directly through memory in real-time will eventually bottleneck the Python thread pool natively; Redis or Time-Series specialized architectures might be needed for next stage scale.
- **Hardware Agnosticism**: Deep learning checks check for `torch.backends.mps` manually (for Apple Sillicon support), CUDA, or CPU fallback explicitly inside model trainers. Ensure PyTorch builds perfectly match operating configurations on target deployments.
