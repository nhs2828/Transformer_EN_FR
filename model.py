import torch
import torch.nn as nn
import math
import numpy as np


def get_positional_embeddings(sequence_length, d):
    result = torch.ones(1,sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[0][i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result

class MLP(nn.Module):
    """
        Multi-Layer Perceptron
    """
    def __init__(self, emb_dim, hidden_dim, p_dropOut = 0.5):
        super().__init__()
        self.seq = nn.Sequential(nn.Linear(in_features=emb_dim, out_features=hidden_dim),\
                                 nn.ReLU(),\
                                 nn.Dropout(p = p_dropOut),\
                                 nn.Linear(in_features=hidden_dim, out_features=emb_dim)
                                )
        
    def forward(self, input):
        return self.seq(input)
    
class MHA(nn.Module):
    def __init__(self, emb_dim, num_heads, mask=False):
        super().__init__()
        self.num_heads = num_heads
        assert emb_dim % self.num_heads == 0
        self.head_dim = emb_dim//self.num_heads
        self.scale = 1/math.sqrt(self.head_dim)
        self.mask = mask
        # First projection for Query, Key, Value
        self.Q = nn.Linear(in_features=emb_dim, out_features=emb_dim)
        self.K = nn.Linear(in_features=emb_dim, out_features=emb_dim)
        self.V = nn.Linear(in_features=emb_dim, out_features=emb_dim)
        # Final projection for output of the block
        self.O = nn.Linear(in_features=emb_dim, out_features=emb_dim)

    def forward(self, Q, K, V):
        """
        Input:
            input: tensor 3-D (Batch, Length, Embedding)
        """
        Q = self.Q(Q)
        K = self.K(K)
        V = self.V(V)
        if self.mask:
            length = input.size(1)
            mask = torch.ones((length,length))
            mask = torch.tril(mask)
            mask = torch.where(mask == 0, float('-inf'), 0.0)

        attention = None
        for head_i in range(self.num_heads):
            Q_head = Q[..., head_i*self.head_dim: (head_i+1)*self.head_dim] # (batch, len, dk)
            K_head = K[..., head_i*self.head_dim: (head_i+1)*self.head_dim] # (batch, len, dk)
            V_head = V[..., head_i*self.head_dim: (head_i+1)*self.head_dim] # (batch, len, dk)

            if not self.mask:
                attention_head = torch.matmul(torch.softmax(self.scale*torch.matmul(Q_head, K_head.permute(0, 2, 1)), dim=-1), V_head)
            else:
                attention_head = torch.matmul(torch.softmax(self.scale*torch.matmul(Q_head, K_head.permute(0, 2, 1))+mask, dim=-1), V_head)

            if head_i == 0:
                attention = attention_head
            else:
                attention = torch.concat([attention, attention_head], dim=-1)
        return self.O(attention)
    
class EncoderBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, hidden_dim, p_dropOut = 0.5):
        super().__init__()
        self.MHA = MHA(emb_dim=emb_dim, num_heads=num_heads, mask=False) # encoder does not need to apply masking onn attention
        self.MLP = MLP(emb_dim=emb_dim, hidden_dim=hidden_dim, p_dropOut=p_dropOut)
        self.norm_1 = nn.LayerNorm(emb_dim)
        self.norm_2 = nn.LayerNorm(emb_dim)

    def forward(self, input):
        # MHA
        out_MHA = self.MHA(input, input, input)
        # Add & Norm
        input_ff = self.norm_1(torch.add(out_MHA, input))
        # Feed forward
        out_ff = self.MLP(input_ff)
        # Add & Norm
        output = self.norm_1(torch.add(out_ff, input_ff))
        return output
    
class DecodeBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, hidden_dim, p_dropOut = 0.5):
        super().__init__()
        self.masked_MHA = MHA(emb_dim=emb_dim, num_heads=num_heads, mask=True)
        self.MHA = MHA(emb_dim=emb_dim, num_heads=num_heads, mask=False)
        self.MLP = MLP(emb_dim=emb_dim, hidden_dim=hidden_dim, p_dropOut=p_dropOut)
        self.norm_1 = nn.LayerNorm(emb_dim)
        self.norm_2 = nn.LayerNorm(emb_dim)
        self.norm_3 = nn.LayerNorm(emb_dim)

    def forward(self, output_emb, output_encoder):
        # MHA masked
        out_MHA_masked = self.masked_MHA(output_emb, output_emb, output_emb)
        # Add & Norm
        in_MHA = self.norm_1(torch.add(output_emb, out_MHA_masked))
        # MHA
        out_MHA = self.MHA(in_MHA, output_encoder, output_encoder) # key and value are from the encoder
        # Add & Norm
        in_ff = self.norm_2(torch.add(in_MHA, out_MHA))
        # Feed forward
        out_ff = self.MLP(in_ff)
        # Add & Norm
        output = self.norm_3(torch.add(in_ff, out_ff))
        return output
    
class Transformer(nn.Module):
    def __init__(self, vocab_size_in, vocab_size_out, emb_dim, hidden_encoder, hidden_decoder, num_heads, num_blocks, p_dropOut=0.5):
        super().__init__()
        self.emb_dim = emb_dim
        encoderBlocks = []
        decoderBlocks = []
        for _ in num_blocks:
            encoderBlocks.append(EncoderBlock(emb_dim=emb_dim, num_heads=num_heads, hidden_dim=hidden_encoder, p_dropOut=p_dropOut))
            decoderBlocks.append(DecodeBlock(emb_dim=emb_dim, num_heads=num_heads, hidden_dim=hidden_decoder, p_dropOut=p_dropOut))
        self.Encoder = nn.Sequential(*encoderBlocks)
        self.Decoder = nn.Sequential(*decoderBlocks)
        self.EncoderEmb = nn.Embedding(num_embeddings=vocab_size_in,embedding_dim=emb_dim)
        self.DecoderEmb = nn.Embedding(num_embeddings=vocab_size_out,embedding_dim=emb_dim)
        self.linear = nn.Linear(in_features=emb_dim, out_features=vocab_size_out)

    def forward(self, input, output):
        batch_size, seq_len_input = input.size()
        seq_len_output = output.size(1)
        # Embedding
        input_emb = self.EncoderEmb(input)
        output_emb = self.DecoderEmb(output)
        # PE
        pos_emb_input = get_positional_embeddings(seq_len_input, self.emb_dim)
        pos_emb_output = get_positional_embeddings(seq_len_output, self.emb_dim)
        input_emb_pe = torch.add(input_emb, pos_emb_input.repeat(batch_size, 1, 1))
        output_emb_pe = torch.add(output_emb, pos_emb_output.repeat(batch_size, 1, 1))
        # encode
        out_encoder = self.Encoder(input_emb_pe)
        # decode
        out_decoder = self.Encoder(output_emb_pe, out_encoder)
        # final linear projection
        output = self.linear(out_decoder)
        return torch.softmax(output, dim=-1)
