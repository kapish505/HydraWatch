# Testing & Evaluation

Testing is primarily focused on evaluating the Machine Learning aspect of the pipeline.

- `scripts/evaluate.py`: Main runner. Initiates a comprehensive replay of the system against BattLeDIM 2019 reference simulation outputs. 
  - Generates ground-truth comparative metrics indicating True Positives/False Positives. 
  - Produces F1 scores against legacy system heuristics (Minimum Night Flow detection, CUSUM controllers) which evaluate performance.

- **ONNX Verification**: Model checkpoints built by `scripts/train.py` automatically inject test arrays into exported ONNX engines. Post-export routines explicitly crash or warn explicitly if floating point representations structurally distort during translation.
