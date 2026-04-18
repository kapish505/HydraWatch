import asyncio
from pathlib import Path
from backend.main import _state, build_pressure_matrix, build_xgboost_features, build_gat_node_features
from backend.data.loader import load_leakdb_scenario, get_node_names
from backend.models.ensemble import LeakEnsemble
from backend.models.gat import LeakGAT, GATTrainer
import numpy as np
import torch

async def test():
    _state["hanoi_dir"] = Path("/Users/kapish/Work/HydraWatch/data/raw/leakdb/LeakDB/Hanoi_CMH/Hanoi_CMH")
    sc = load_leakdb_scenario(1, _state["hanoi_dir"])
    pressures = build_pressure_matrix(sc)
    
    # Try GAT prediction
    gat_model = GATTrainer(
        edge_index=np.array([[0, 1], [1, 0]]), # Mock
        n_node_features=13,
        hidden_channels=64,
        heads=4,
        lr=1e-3,
    )
    # Load the real model
    gat_model.load("/Users/kapish/Work/HydraWatch/models/gat_localiser.pt")
    
    gat_feats = build_gat_node_features(pressures, 24)
    print("GAT FEATS SHAPE:", gat_feats.shape)
    
    node_names = get_node_names(sc)
    probs = gat_model.predict(gat_feats, node_names)
    print("PROBS:", list(probs.values())[:3])
    
asyncio.run(test())
