"""
Spatial Graph Construction and Reasoning for MGA-VQA.
Implements graph construction from OCR results and GNN-based reasoning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from typing import List, Dict, Tuple, Optional
import math
import numpy as np
from transformers import AutoTokenizer, AutoModel


class SpatialGraphConstructor(nn.Module):
    """
    Constructs spatial graphs from OCR results and visual features.
    Implements the graph construction logic from the MGA-VQA paper.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.node_dim = config.node_dim
        self.edge_dim = config.edge_dim
        self.max_nodes = config.max_nodes
        
        # Text encoder for semantic embeddings
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.text_encoder = AutoModel.from_pretrained('bert-base-uncased')
        
        # Node feature fusion layers
        self.text_proj = nn.Linear(768, self.node_dim // 3)
        self.visual_proj = nn.Linear(768, self.node_dim // 3)  
        self.position_proj = nn.Linear(4, self.node_dim // 3)
        
        self.node_fusion = nn.Sequential(
            nn.Linear(self.node_dim, self.node_dim),
            nn.LayerNorm(self.node_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # Edge weight computation
        self.edge_mlp = nn.Sequential(
            nn.Linear(self.node_dim * 2 + 6, self.edge_dim),  # 6 for spatial features
            nn.ReLU(),
            nn.Linear(self.edge_dim, 1),
            nn.Sigmoid()
        )
        
    def normalize_bbox(self, bbox: List[float], image_size: Tuple[int, int]) -> List[float]:
        """Normalize bounding box coordinates to [0, 1]."""
        width, height = image_size
        x1, y1, x2, y2 = bbox
        return [x1 / width, y1 / height, x2 / width, y2 / height]
    
    def compute_spatial_distance(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Compute spatial distance between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Centers
        center1 = ((x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2)
        center2 = ((x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2)
        
        # Euclidean distance
        distance = math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        return distance
    
    def compute_alignment_score(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Compute alignment score between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Horizontal alignment
        h_overlap = max(0, min(x2_1, x2_2) - max(x1_1, x1_2))
        h_union = max(x2_1, x2_2) - min(x1_1, x1_2)
        h_align = h_overlap / h_union if h_union > 0 else 0
        
        # Vertical alignment  
        v_overlap = max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
        v_union = max(y2_1, y2_2) - min(y1_1, y1_2)
        v_align = v_overlap / v_union if v_union > 0 else 0
        
        return max(h_align, v_align)
    
    def compute_semantic_similarity(self, text1_emb: torch.Tensor, text2_emb: torch.Tensor) -> float:
        """Compute semantic similarity between text embeddings."""
        similarity = F.cosine_similarity(text1_emb, text2_emb, dim=-1)
        return similarity.item()
    
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode text using BERT."""
        if not texts:
            return torch.zeros(0, 768)
        
        # Tokenize texts
        inputs = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=64
        )
        
        # Move to device
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.text_encoder(**inputs)
            text_embeddings = outputs.last_hidden_state.mean(dim=1)  # [num_texts, 768]
        
        return text_embeddings
    
    def create_node_features(self, ocr_results: List[Dict], visual_features: torch.Tensor,
                           image_size: Tuple[int, int]) -> torch.Tensor:
        """
        Create node features by fusing text, visual, and positional information.
        
        Args:
            ocr_results: List of OCR results with 'text' and 'bbox' keys
            visual_features: Visual features from encoder [B, N, D]
            image_size: (width, height) of the image
            
        Returns:
            node_features: [num_nodes, node_dim]
        """
        if not ocr_results:
            return torch.zeros(0, self.node_dim).to(next(self.parameters()).device)
        
        # Extract text and bboxes
        texts = [result['text'] for result in ocr_results]
        bboxes = [result['bbox'] for result in ocr_results]
        
        # Encode texts
        text_embeddings = self.encode_text(texts)
        text_features = self.text_proj(text_embeddings)
        
        # Process visual features (use global average for now)
        visual_global = visual_features.mean(dim=1)  # [B, D] -> [D]
        if visual_global.dim() > 1:
            visual_global = visual_global[0]  # Take first batch
        visual_expanded = visual_global.unsqueeze(0).repeat(len(ocr_results), 1)
        visual_features_proj = self.visual_proj(visual_expanded)
        
        # Normalize and encode positional features
        normalized_bboxes = [self.normalize_bbox(bbox, image_size) for bbox in bboxes]
        position_features = torch.tensor(normalized_bboxes, dtype=torch.float32)
        position_features = position_features.to(next(self.parameters()).device)
        position_features_proj = self.position_proj(position_features)
        
        # Fuse all features
        fused_features = torch.cat([
            text_features,
            visual_features_proj, 
            position_features_proj
        ], dim=-1)
        
        node_features = self.node_fusion(fused_features)
        return node_features
    
    def compute_edge_weights(self, ocr_results: List[Dict], node_features: torch.Tensor,
                           image_size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute edge indices and weights based on spatial and semantic relationships.
        
        Args:
            ocr_results: List of OCR results
            node_features: Node features [num_nodes, node_dim]
            image_size: (width, height) of image
            
        Returns:
            edge_index: [2, num_edges]
            edge_weights: [num_edges]
        """
        num_nodes = len(ocr_results)
        if num_nodes <= 1:
            return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0)
        
        edge_indices = []
        edge_weights = []
        
        # Get text embeddings for semantic similarity
        texts = [result['text'] for result in ocr_results]
        text_embeddings = self.encode_text(texts)
        
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                bbox1 = ocr_results[i]['bbox']
                bbox2 = ocr_results[j]['bbox']
                
                # Normalize bboxes
                norm_bbox1 = self.normalize_bbox(bbox1, image_size)
                norm_bbox2 = self.normalize_bbox(bbox2, image_size)
                
                # Compute components of edge weight (Equation 2)
                d_spatial = self.compute_spatial_distance(norm_bbox1, norm_bbox2)
                a_alignment = self.compute_alignment_score(norm_bbox1, norm_bbox2)
                s_semantic = self.compute_semantic_similarity(
                    text_embeddings[i:i+1], text_embeddings[j:j+1]
                )
                
                # Combined edge weight
                edge_weight = (
                    self.config.spatial_weight_alpha * (1 - d_spatial) +  # Inverse distance
                    self.config.alignment_weight_beta * a_alignment +
                    self.config.semantic_weight_gamma * s_semantic
                )
                
                # Alternative: Use MLP for edge weight computation
                spatial_features = torch.tensor([
                    d_spatial, a_alignment, s_semantic,
                    abs(norm_bbox1[0] - norm_bbox2[0]),  # x difference
                    abs(norm_bbox1[1] - norm_bbox2[1]),  # y difference  
                    abs((norm_bbox1[2] - norm_bbox1[0]) - (norm_bbox2[2] - norm_bbox2[0]))  # width diff
                ], dtype=torch.float32).to(next(self.parameters()).device)
                
                edge_input = torch.cat([
                    node_features[i],
                    node_features[j], 
                    spatial_features
                ], dim=-1).unsqueeze(0)
                
                mlp_weight = self.edge_mlp(edge_input).squeeze()
                
                # Use MLP weight if above threshold
                if mlp_weight > 0.1:  # Threshold for edge existence
                    edge_indices.extend([[i, j], [j, i]])  # Undirected graph
                    edge_weights.extend([mlp_weight.item(), mlp_weight.item()])
        
        if not edge_indices:
            return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0)
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).T
        edge_weights = torch.tensor(edge_weights, dtype=torch.float32)
        
        return edge_index, edge_weights
    
    def forward(self, ocr_results: List[Dict], visual_features: torch.Tensor,
                image_size: Tuple[int, int]) -> Data:
        """
        Construct spatial graph from OCR results and visual features.
        
        Args:
            ocr_results: List of dicts with 'text' and 'bbox' keys
            visual_features: Visual features from encoder [B, N, D]
            image_size: (width, height) of image
            
        Returns:
            graph_data: PyTorch Geometric Data object
        """
        # Limit number of nodes
        if len(ocr_results) > self.max_nodes:
            ocr_results = ocr_results[:self.max_nodes]
        
        # Create node features
        node_features = self.create_node_features(ocr_results, visual_features, image_size)
        
        if node_features.shape[0] == 0:
            # Return empty graph
            return Data(x=torch.zeros(1, self.node_dim), 
                       edge_index=torch.zeros(2, 0, dtype=torch.long),
                       edge_attr=torch.zeros(0, 1))
        
        # Compute edges and weights
        edge_index, edge_weights = self.compute_edge_weights(
            ocr_results, node_features, image_size
        )
        
        # Create graph data
        graph_data = Data(
            x=node_features,
            edge_index=edge_index.to(next(self.parameters()).device),
            edge_attr=edge_weights.unsqueeze(-1).to(next(self.parameters()).device)
        )
        
        return graph_data


class GraphNeuralNetwork(nn.Module):
    """
    Graph Neural Network for spatial reasoning.
    Implements 3-layer GCN with residual connections.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.node_dim = config.node_dim
        self.hidden_dim = config.gnn_hidden_dim
        self.num_layers = config.num_gnn_layers
        
        # Graph convolutional layers
        self.gcn_layers = nn.ModuleList()
        self.residual_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        
        # First layer
        self.gcn_layers.append(GCNConv(self.node_dim, self.hidden_dim))
        self.residual_layers.append(nn.Linear(self.node_dim, self.hidden_dim))
        self.norm_layers.append(nn.LayerNorm(self.hidden_dim))
        
        # Hidden layers
        for _ in range(self.num_layers - 2):
            self.gcn_layers.append(GCNConv(self.hidden_dim, self.hidden_dim))
            self.residual_layers.append(nn.Identity())
            self.norm_layers.append(nn.LayerNorm(self.hidden_dim))
        
        # Output layer
        self.gcn_layers.append(GCNConv(self.hidden_dim, self.node_dim))
        self.residual_layers.append(nn.Linear(self.hidden_dim, self.node_dim))
        self.norm_layers.append(nn.LayerNorm(self.node_dim))
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Global graph pooling
        self.graph_pooling = nn.Sequential(
            nn.Linear(self.node_dim, self.node_dim),
            nn.GELU(),
            nn.Linear(self.node_dim, self.node_dim)
        )
        
    def forward(self, graph_data: Data) -> Dict[str, torch.Tensor]:
        """
        Forward pass through graph neural network.
        
        Args:
            graph_data: PyTorch Geometric Data object
            
        Returns:
            dict containing:
                - node_embeddings: Updated node features
                - graph_embedding: Global graph representation
        """
        x = graph_data.x
        edge_index = graph_data.edge_index
        edge_weight = graph_data.edge_attr.squeeze(-1) if graph_data.edge_attr is not None else None
        
        # Handle empty graphs
        if x.shape[0] == 0:
            return {
                'node_embeddings': torch.zeros(0, self.node_dim).to(x.device),
                'graph_embedding': torch.zeros(1, self.node_dim).to(x.device)
            }
        
        # Apply GCN layers with residual connections
        for i, (gcn, residual, norm) in enumerate(zip(
            self.gcn_layers, self.residual_layers, self.norm_layers
        )):
            x_residual = residual(x)
            
            # Graph convolution (Equation 3 from paper)
            if edge_index.shape[1] > 0:
                x = gcn(x, edge_index, edge_weight)
            else:
                # No edges, just apply linear transformation
                x = gcn.lin(x)
            
            # Residual connection and normalization
            if self.config.use_residual:
                x = x + x_residual
            x = norm(x)
            
            # Apply activation and dropout (except last layer)
            if i < len(self.gcn_layers) - 1:
                x = F.gelu(x)
                x = self.dropout(x)
        
        # Global graph embedding
        if hasattr(graph_data, 'batch'):
            graph_embedding = global_mean_pool(x, graph_data.batch)
        else:
            graph_embedding = x.mean(dim=0, keepdim=True)
        
        graph_embedding = self.graph_pooling(graph_embedding)
        
        return {
            'node_embeddings': x,
            'graph_embedding': graph_embedding
        }


def create_spatial_modules(config):
    """Factory function to create spatial graph modules."""
    return {
        'constructor': SpatialGraphConstructor(config),
        'reasoner': GraphNeuralNetwork(config)
    }
