import torch
import torch.nn as nn
import math
import numpy as np


# def get_positional_embeddings(sequence_length, d):
#     result = torch.ones(1,sequence_length, d)
#     for i in range(sequence_length):
#         for j in range(d):
#             result[0][i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
#     return result

class PositionalEmbedding():
    def __init__(self, max_len, emb_dim):
        assert emb_dim % 2 == 0
        pos = torch.arange(max_len).reshape(-1,1)
        i = torch.arange(emb_dim//2).reshape(1,-1)
        self.PE = torch.empty((max_len, emb_dim))
        self.PE[..., 0::2] = torch.sin(pos/torch.pow(10000, 2*i/emb_dim))
        self.PE[..., 1::2] = torch.cos(pos/torch.pow(10000, 2*i/emb_dim))
        self.PE = self.PE.cuda()

    def embed(self, input):
        _, N, _ = input.size()
        return torch.add(input, self.PE[:N])
    

class MLP(nn.Module):
    """
        Multi-Layer Perceptron
    """
    def __init__(self, emb_dim, hidden_ratio, p_dropOut = 0.5):
        super().__init__()
        self.seq = nn.Sequential(nn.Linear(in_features=emb_dim, out_features=emb_dim*hidden_ratio),\
                                 nn.ReLU(),\
                                 nn.Dropout(p = p_dropOut),\
                                 nn.Linear(in_features=emb_dim*hidden_ratio, out_features=emb_dim)
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
            length = Q.size(1)
            mask = torch.ones((length,length))
            mask = torch.tril(mask)
            mask = torch.where(mask == 0, float('-inf'), 0.0)
            mask = mask.cuda()

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
    def __init__(self, emb_dim, num_heads, hidden_ratio, p_dropOut = 0.5):
        super().__init__()
        self.MHA = MHA(emb_dim=emb_dim, num_heads=num_heads, mask=False) # encoder does not need to apply masking onn attention
        self.MLP = MLP(emb_dim=emb_dim, hidden_ratio=hidden_ratio, p_dropOut=p_dropOut)
        self.norm_1 = nn.LayerNorm(emb_dim)
        self.norm_2 = nn.LayerNorm(emb_dim)

    def forward(self, inputs):
        # MHA
        print("encoder", inputs.shape)
        out_MHA = self.MHA(inputs, inputs, inputs)
        # Add & Norm
        input_ff = self.norm_1(torch.add(out_MHA, inputs))
        # Feed forward
        out_ff = self.MLP(input_ff)
        # Add & Norm
        output = self.norm_1(torch.add(out_ff, input_ff))
        return output
    
class DecodeBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, hidden_ratio, p_dropOut = 0.5):
        super().__init__()
        self.masked_MHA = MHA(emb_dim=emb_dim, num_heads=num_heads, mask=True)
        self.MHA = MHA(emb_dim=emb_dim, num_heads=num_heads, mask=False)
        self.MLP = MLP(emb_dim=emb_dim, hidden_ratio=hidden_ratio, p_dropOut=p_dropOut)
        self.norm_1 = nn.LayerNorm(emb_dim)
        self.norm_2 = nn.LayerNorm(emb_dim)
        self.norm_3 = nn.LayerNorm(emb_dim)

    def forward(self, inputs):
        output_emb, output_encoder = inputs
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
        return (output_emb, output)
    
class Transformer(nn.Module):
    def __init__(self, vocab_size_in, vocab_size_out, emb_dim, hidden_ratio_encoder, hidden_ratio_decoder, num_heads, num_blocks,max_len, p_dropOut=0.5):
        super().__init__()
        self.emb_dim = emb_dim
        encoderBlocks = []
        decoderBlocks = []
        for _ in range(num_blocks):
            encoderBlocks.append(EncoderBlock(emb_dim=emb_dim, num_heads=num_heads, hidden_ratio=hidden_ratio_encoder, p_dropOut=p_dropOut))
            decoderBlocks.append(DecodeBlock(emb_dim=emb_dim, num_heads=num_heads, hidden_ratio=hidden_ratio_decoder, p_dropOut=p_dropOut))
        self.Encoder = nn.Sequential(*encoderBlocks)
        self.Decoder = nn.Sequential(*decoderBlocks)
        self.EncoderEmb = nn.Embedding(num_embeddings=vocab_size_in,embedding_dim=emb_dim)
        self.DecoderEmb = nn.Embedding(num_embeddings=vocab_size_out,embedding_dim=emb_dim)
        self.PE = PositionalEmbedding(max_len=max_len, emb_dim=emb_dim)
        self.linear = nn.Linear(in_features=emb_dim, out_features=vocab_size_out)

    def forward(self, inputs, outputs):
        # Embedding
        input_emb = self.EncoderEmb(inputs)
        output_emb = self.DecoderEmb(outputs)
        # PE
        input_emb_pe = self.PE.embed(input_emb)
        output_emb_pe = self.PE.embed(output_emb)
        # encode
        out_encoder = self.Encoder(input_emb_pe)
        # decode
        _, out_decoder = self.Decoder((output_emb_pe, out_encoder))
        # final linear projection
        output = self.linear(out_decoder)
        return output
        
def maskedCrossEntropy(output: torch.Tensor, target: torch.LongTensor, padcar: int):
    """
    :param output: Tensor batch x length x output_dim,
    :param target: Tensor batch x length
    :param padcar: index of PAD character
    """
    sm_output = torch.log(torch.softmax(output, dim=-1)) # log softmax
    masque_target = torch.where(target == padcar, 0., 1.).cuda()
    index_target = target.unsqueeze(-1) # batch, leng, 1
    loss = -torch.gather(sm_output, dim=-1, index=index_target).squeeze(-1)*masque_target
    return torch.mean(loss)
