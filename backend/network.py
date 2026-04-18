"""
WNTR graph utilities for HydraWatch.

Loads EPANET .inp network files using WNTR (Water Network Tool for Resilience)
and provides helper functions to convert the graph into JSON for the frontend
and to compute adjacency information for feature engineering.

Think of a water network like a road map:
  - Junctions (nodes) = intersections where water flows
  - Pipes (edges) = roads connecting intersections
  - Tanks/Reservoirs = water sources
  
WNTR reads the standard EPANET .inp file format used by water utilities worldwide.
"""

import wntr
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


def load_network(inp_file: str) -> wntr.network.WaterNetworkModel:
    """
    Load a water network from an EPANET .inp file.
    
    The .inp file is the standard format for describing water distribution
    networks — it contains all the pipes, junctions, tanks, and their
    physical properties (length, diameter, roughness, elevation, etc.).
    
    Args:
        inp_file: Path to the EPANET .inp file
        
    Returns:
        WNTR WaterNetworkModel object
    """
    inp_path = Path(inp_file)
    if not inp_path.exists():
        raise FileNotFoundError(f"Network file not found: {inp_file}")
    
    wn = wntr.network.WaterNetworkModel(str(inp_path))
    print(f"  Loaded network: {inp_path.name}")
    print(f"  Junctions: {wn.num_nodes}")
    print(f"  Pipes: {wn.num_links}")
    return wn


def get_junction_names(wn: wntr.network.WaterNetworkModel) -> List[str]:
    """Get list of junction (sensor node) names, sorted."""
    return sorted(wn.junction_name_list)


def get_pipe_names(wn: wntr.network.WaterNetworkModel) -> List[str]:
    """Get list of pipe names, sorted."""
    return sorted(wn.pipe_name_list)


def get_adjacency(wn: wntr.network.WaterNetworkModel) -> Dict[str, List[str]]:
    """
    Build adjacency dict: each node maps to its list of neighbour nodes.
    
    This is used in feature engineering — when a pipe leaks, pressure drops
    in the leaking node AND its neighbours. So knowing who is connected to
    whom helps us compute "neighbour pressure delta" features.
    
    Args:
        wn: WNTR water network model
        
    Returns:
        Dict mapping node_name -> [list of neighbour node_names]
    """
    adjacency = {}
    junction_names = set(wn.junction_name_list)
    
    for jname in wn.junction_name_list:
        neighbors = []
        # Get the junction node object
        node = wn.get_node(jname)
        
        # Find all pipes connected to this junction
        for link_name, link in wn.links():
            start_name = link.start_node_name
            end_name = link.end_node_name
            
            if start_name == jname and end_name in junction_names:
                neighbors.append(end_name)
            elif end_name == jname and start_name in junction_names:
                neighbors.append(start_name)
        
        adjacency[jname] = neighbors
    
    return adjacency


def network_to_json(wn: wntr.network.WaterNetworkModel) -> Dict[str, Any]:
    """
    Convert the WNTR network model into JSON-serializable format for the 
    frontend D3.js graph visualization.
    
    Each node gets: id, type, x/y position, elevation
    Each edge gets: source, target, diameter, length, roughness
    
    Args:
        wn: WNTR water network model
        
    Returns:
        Dict with 'nodes' and 'edges' lists
    """
    nodes = []
    edges = []
    
    # Build lookup of which names are junctions vs reservoirs/tanks
    junction_set = set(wn.junction_name_list)
    
    def _prefix(name):
        """Prefix junction names with Node_ to match data pipeline naming."""
        return f"Node_{name}" if name in junction_set else name
    
    # ── Nodes (junctions, tanks, reservoirs) ─────────────────────────
    for name, node in wn.nodes():
        node_data = {
            "id": _prefix(name),
            "type": node.node_type,
            "elevation": float(node.elevation) if hasattr(node, "elevation") else 0.0,
        }
        
        # Try to get coordinates if they exist in the .inp file
        try:
            coords = wn.get_node(name).coordinates
            if coords:
                node_data["x"] = float(coords[0])
                node_data["y"] = float(coords[1])
        except (AttributeError, TypeError):
            pass
        
        # Add base demand for junctions
        if hasattr(node, "base_demand"):
            node_data["base_demand"] = float(node.base_demand) if node.base_demand else 0.0
        
        nodes.append(node_data)
    
    # ── Edges (pipes, pumps, valves) ─────────────────────────────────
    for name, link in wn.links():
        edge_data = {
            "id": name,
            "source": _prefix(link.start_node_name),
            "target": _prefix(link.end_node_name),
            "type": link.link_type,
        }
        
        # Physical properties (pipes have diameter, length, roughness)
        if hasattr(link, "diameter"):
            edge_data["diameter"] = float(link.diameter) if link.diameter else 0.0
        if hasattr(link, "length"):
            edge_data["length"] = float(link.length) if link.length else 0.0
        if hasattr(link, "roughness"):
            edge_data["roughness"] = float(link.roughness) if link.roughness else 0.0
        
        edges.append(edge_data)
    
    return {"nodes": nodes, "edges": edges}


def get_edge_index_and_features(
    wn: wntr.network.WaterNetworkModel,
    junction_names: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build edge_index and edge features for PyTorch Geometric.
    
    In graph neural networks, we represent the graph as:
    - edge_index: [2, num_edges] array of (source, target) pairs
    - edge_attr: [num_edges, num_edge_features] array
    
    Edge features: [diameter, length, roughness] (normalized)
    
    Args:
        wn: WNTR water network model
        junction_names: Optional list of junction names (for consistent ordering)
        
    Returns:
        (edge_index, edge_attr) numpy arrays
    """
    if junction_names is None:
        junction_names = sorted(wn.junction_name_list)
    
    name_to_idx = {name: idx for idx, name in enumerate(junction_names)}
    junction_set = set(junction_names)
    
    src_list = []
    dst_list = []
    edge_features = []
    
    for link_name, link in wn.links():
        start = link.start_node_name
        end = link.end_node_name
        
        # Only include edges between junctions (skip tank/reservoir connections)
        if start not in junction_set or end not in junction_set:
            continue
        
        diameter = float(link.diameter) if hasattr(link, "diameter") and link.diameter else 0.0
        length = float(link.length) if hasattr(link, "length") and link.length else 0.0
        roughness = float(link.roughness) if hasattr(link, "roughness") and link.roughness else 0.0
        
        # Add both directions (undirected graph)
        src_list.extend([name_to_idx[start], name_to_idx[end]])
        dst_list.extend([name_to_idx[end], name_to_idx[start]])
        edge_features.extend([[diameter, length, roughness]] * 2)
    
    edge_index = np.array([src_list, dst_list], dtype=np.int64)
    edge_attr = np.array(edge_features, dtype=np.float32)
    
    # Normalize edge features to [0, 1] range
    if len(edge_attr) > 0:
        for col in range(edge_attr.shape[1]):
            col_max = edge_attr[:, col].max()
            if col_max > 0:
                edge_attr[:, col] /= col_max
    
    return edge_index, edge_attr
