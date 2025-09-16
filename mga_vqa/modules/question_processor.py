"""
Memory-Augmented Question Processing for MGA-VQA.
Implements dual memory system with Direct Memory (DM) and Indirect Memory (IM).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModel
import math


class DualMemorySystem(nn.Module):
    """
    Dual Memory System with Direct Memory (DM) and Indirect Memory (IM).
    DM stores high-confidence answer candidates.
    IM stores contextual patterns and layout-aware dependencies.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.memory_dim = config.memory_dim
        self.dm_size = config.direct_memory_size
        self.im_size = config.indirect_memory_size
        
        # Direct Memory - high-confidence answer candidates
        self.direct_memory = nn.Parameter(
            torch.randn(self.dm_size, self.memory_dim) * 0.1
        )
        
        # Indirect Memory - contextual patterns and layout dependencies  
        self.indirect_memory = nn.Parameter(
            torch.randn(self.im_size, self.memory_dim) * 0.1
        )
        
        # Memory access controllers
        self.dm_controller = nn.Sequential(
            nn.Linear(self.memory_dim, self.memory_dim),
            nn.Tanh()
        )
        
        self.im_controller = nn.Sequential(
            nn.Linear(self.memory_dim, self.memory_dim),
            nn.Tanh()
        )
        
        # Memory gating mechanisms
        self.dm_gate = nn.Sequential(
            nn.Linear(self.memory_dim * 2, self.memory_dim),
            nn.Sigmoid()
        )
        
        self.im_gate = nn.Sequential(
            nn.Linear(self.memory_dim * 2, self.memory_dim),
            nn.Sigmoid()
        )
    
    def access_direct_memory(self, query: torch.Tensor) -> torch.Tensor:
        """
        Access Direct Memory based on query.
        
        Args:
            query: Query tensor [B, D] or [D]
            
        Returns:
            dm_output: Retrieved information from Direct Memory
        """
        if query.dim() == 1:
            query = query.unsqueeze(0)
        
        batch_size = query.shape[0]
        
        # Compute attention scores
        query_proj = self.dm_controller(query)  # [B, D]
        scores = torch.matmul(query_proj, self.direct_memory.T)  # [B, dm_size]
        attention_weights = F.softmax(scores, dim=-1)  # [B, dm_size]
        
        # Retrieve memory content
        dm_content = torch.matmul(attention_weights, self.direct_memory)  # [B, D]
        
        # Apply gating
        gate_input = torch.cat([query, dm_content], dim=-1)
        gate = self.dm_gate(gate_input)
        dm_output = gate * dm_content + (1 - gate) * query
        
        return dm_output
    
    def access_indirect_memory(self, query: torch.Tensor) -> torch.Tensor:
        """
        Access Indirect Memory based on query.
        
        Args:
            query: Query tensor [B, D] or [D]
            
        Returns:
            im_output: Retrieved information from Indirect Memory
        """
        if query.dim() == 1:
            query = query.unsqueeze(0)
        
        batch_size = query.shape[0]
        
        # Compute attention scores
        query_proj = self.im_controller(query)  # [B, D]
        scores = torch.matmul(query_proj, self.indirect_memory.T)  # [B, im_size]
        attention_weights = F.softmax(scores, dim=-1)  # [B, im_size]
        
        # Retrieve memory content
        im_content = torch.matmul(attention_weights, self.indirect_memory)  # [B, D]
        
        # Apply gating
        gate_input = torch.cat([query, im_content], dim=-1)
        gate = self.im_gate(gate_input)
        im_output = gate * im_content + (1 - gate) * query
        
        return im_output
    
    def forward(self, query: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through dual memory system.
        
        Args:
            query: Question embedding [B, D] or [D]
            
        Returns:
            dict containing:
                - dm_output: Direct memory output
                - im_output: Indirect memory output  
                - combined_output: Combined memory output
        """
        dm_output = self.access_direct_memory(query)
        im_output = self.access_indirect_memory(query)
        
        # Combine memories
        combined_output = (dm_output + im_output) / 2
        
        return {
            'dm_output': dm_output,
            'im_output': im_output, 
            'combined_output': combined_output
        }


class MemoryIntegrationModule(nn.Module):
    """
    Memory Integration via Cross-Attention mechanism.
    Implements Equation (4): M_integrated = ATTENTION(Q, [DM; IM], [DM; IM])
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.memory_dim = config.memory_dim
        self.num_heads = config.num_attention_heads
        self.dropout = config.dropout
        
        # Cross-attention for memory integration
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.memory_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(self.memory_dim)
        self.norm2 = nn.LayerNorm(self.memory_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(self.memory_dim, self.memory_dim * 4),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.memory_dim * 4, self.memory_dim)
        )
    
    def forward(self, question_emb: torch.Tensor, direct_memory: torch.Tensor,
                indirect_memory: torch.Tensor) -> torch.Tensor:
        """
        Integrate memories using cross-attention.
        
        Args:
            question_emb: Question embedding [B, D]
            direct_memory: Direct memory content [dm_size, D]
            indirect_memory: Indirect memory content [im_size, D]
            
        Returns:
            integrated_memory: Memory-integrated question representation [B, D]
        """
        if question_emb.dim() == 1:
            question_emb = question_emb.unsqueeze(0)
        
        batch_size = question_emb.shape[0]
        
        # Concatenate memories
        memories = torch.cat([direct_memory, indirect_memory], dim=0)  # [dm_size + im_size, D]
        memories = memories.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, dm_size + im_size, D]
        
        # Query, Key, Value
        query = question_emb.unsqueeze(1)  # [B, 1, D]
        key = value = memories  # [B, dm_size + im_size, D]
        
        # Cross-attention
        attended_output, attention_weights = self.cross_attention(
            query, key, value
        )  # attended_output: [B, 1, D]
        
        # Remove sequence dimension
        attended_output = attended_output.squeeze(1)  # [B, D]
        
        # Residual connection and layer norm
        integrated = self.norm1(attended_output + question_emb)
        
        # Feed-forward network
        ffn_output = self.ffn(integrated)
        integrated = self.norm2(ffn_output + integrated)
        
        return integrated


class MemoryAugmentedQuestionProcessor(nn.Module):
    """
    Memory-Augmented Question Processor combining question encoding 
    with dual memory system for enhanced reasoning.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.memory_dim = config.memory_dim
        self.max_length = config.max_question_length
        
        # Question encoder - Updated to use Gemma backbone
        self.tokenizer = AutoTokenizer.from_pretrained(config.question_encoder)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.question_encoder = AutoModel.from_pretrained(config.question_encoder)
        
        # Project question embeddings to memory dimension
        # Gemma models typically have larger hidden sizes
        question_dim = self.question_encoder.config.hidden_size if hasattr(self.question_encoder.config, 'hidden_size') else 4096
        self.question_proj = nn.Linear(question_dim, self.memory_dim)
        
        # Dual memory system
        self.dual_memory = DualMemorySystem(config)
        
        # Memory integration module
        self.memory_integration = MemoryIntegrationModule(config)
        
        # Question processing layers
        self.question_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.memory_dim,
                nhead=config.num_attention_heads,
                dim_feedforward=self.memory_dim * 4,
                dropout=config.dropout,
                batch_first=True
            )
            for _ in range(2)
        ])
        
        # Final processing
        self.final_proj = nn.Sequential(
            nn.Linear(self.memory_dim, self.memory_dim),
            nn.LayerNorm(self.memory_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
    
    def encode_question(self, questions: List[str]) -> torch.Tensor:
        """
        Encode questions using pre-trained language model.
        
        Args:
            questions: List of question strings
            
        Returns:
            question_embeddings: [B, D] question embeddings
        """
        # Tokenize questions
        inputs = self.tokenizer(
            questions,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        # Move to device
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Encode questions
        with torch.no_grad():
            outputs = self.question_encoder(**inputs)
            question_emb = outputs.last_hidden_state.mean(dim=1)  # [B, hidden_size]
        
        # Project to memory dimension
        question_emb = self.question_proj(question_emb)  # [B, memory_dim]
        
        return question_emb
    
    def process_with_memory(self, question_emb: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process question with dual memory system.
        
        Args:
            question_emb: Question embeddings [B, D]
            
        Returns:
            dict containing processed representations
        """
        # Access dual memory
        memory_outputs = self.dual_memory(question_emb)
        
        # Integrate memories using cross-attention (Equation 4)
        integrated_memory = self.memory_integration(
            question_emb,
            self.dual_memory.direct_memory,
            self.dual_memory.indirect_memory
        )
        
        return {
            'question_emb': question_emb,
            'dm_output': memory_outputs['dm_output'],
            'im_output': memory_outputs['im_output'],
            'integrated_memory': integrated_memory
        }
    
    def apply_question_layers(self, question_emb: torch.Tensor) -> torch.Tensor:
        """Apply transformer layers for question processing."""
        if question_emb.dim() == 2:
            question_emb = question_emb.unsqueeze(1)  # Add sequence dimension
        
        processed = question_emb
        for layer in self.question_layers:
            processed = layer(processed)
        
        if processed.shape[1] == 1:
            processed = processed.squeeze(1)  # Remove sequence dimension
        
        return processed
    
    def forward(self, questions: List[str]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Memory-Augmented Question Processor.
        
        Args:
            questions: List of question strings
            
        Returns:
            dict containing:
                - question_processed: Final processed question representation
                - integrated_memory: Memory-integrated representation  
                - memory_components: Individual memory components
        """
        # Encode questions
        question_emb = self.encode_question(questions)
        
        # Process with memory system
        memory_results = self.process_with_memory(question_emb)
        
        # Apply question processing layers
        processed_question = self.apply_question_layers(memory_results['integrated_memory'])
        
        # Final projection
        final_processed = self.final_proj(processed_question)
        
        return {
            'question_processed': final_processed,
            'integrated_memory': memory_results['integrated_memory'],
            'memory_components': {
                'dm_output': memory_results['dm_output'],
                'im_output': memory_results['im_output'],
                'original_question': memory_results['question_emb']
            }
        }


def create_question_processor(config):
    """Factory function to create question processor."""
    return MemoryAugmentedQuestionProcessor(config)
