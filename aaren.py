# author: gyuntian
# 

# cython: language_level=3

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import aaren_cuda

# Load the CUDA kernel
# aaren_cuda = load(name="aaren_cuda", sources=["./aaren.cu"])

# to enhance
#   rotary or alibi
#   multi head
#   kv cache
#   global awareness
class AARENLayer(nn.Module):
    def __init__(self, vocab_size, embedding_dim=1024, query_dim=1024):
        super(AARENLayer, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.query_vector = nn.Parameter(torch.empty(1, query_dim)) # Trainable Q
        
        self.key_layer = nn.Linear(embedding_dim, query_dim)  # Linear layer for K
        self.value_layer = nn.Linear(embedding_dim, query_dim)  # Linear layer for V

        self._init()
    
    def _init(self):
        # Initialize query_vector
        nn.init.xavier_uniform_(self.query_vector)
        
        # List of linear layers
        for layer in [self.key_layer, self.value_layer]:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:  # Check if there's a bias
                nn.init.zeros_(layer.bias)
        
    def forward(self, input_ids, attention_mask=None):

        # Embed input IDs
        embedded_inputs = self.embedding(input_ids)  # Shape: (batch_size, seq_length, embedding_dim)
        
        # Compute K and V
        # K = self.key_layer(embedded_inputs)  # Shape: (batch_size, seq_length, query_dim)
        V = self.value_layer(embedded_inputs)  # Shape: (batch_size, seq_length, query_dim)

        # Compute scores s_i
        s_i = self.key_layer(embedded_inputs) @ self.query_vector.t() # Shape: (batch_size, seq_length, 1)

        # Call CUDA kernel for output
        output_uA, output_wA = aaren_cuda.launchKernel(s_i, V)
        
        return output_uA, output_wA


# Example usage
if __name__ == "__main__":
    # Parameters
    vocab_size = 1000
    embedding_dim = 128
    query_dim = 32

    model = AARENLayer(vocab_size, embedding_dim, query_dim)
    input_ids = torch.randint(0, vocab_size, (10, 50))  # Example batch of input IDs
    output_uA, output_wA = model(input_ids)
    print(output_uA.shape, output_wA.shape)
    
