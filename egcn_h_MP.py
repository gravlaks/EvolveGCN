import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import utils as u
import torch_geometric
from torch_geometric.nn.conv import MessagePassing
import torch.nn as nn
import torch_scatter
    
class GAT_MP(MessagePassing):

    def __init__(self, in_channels, out_channels, heads = 2,
                 negative_slope = 0.2, dropout = 0., **kwargs):
        super(GAT_MP, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout


        self.att_l = Parameter(torch.zeros((1, heads, out_channels)))
        self.att_r = Parameter(torch.zeros((1, heads, out_channels)))
        print("Shape", self.att_l.shape)
        self.reset_parameters()


    def reset_parameters(self):

        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)

    def forward(self, x, edge_index, size = None, weights=None):
        

        ############################################################################
        # TODO: Your code here! 
        # Implement message passing, as well as any pre- and post-processing (our update rule).
        # 1. First apply linear transformation to node embeddings, and split that 
        #    into multiple heads. We use the same representations for source and
        #    target nodes, but apply different linear weights (W_l and W_r)
        # 2. Calculate alpha vectors for central nodes (alpha_l) and neighbor nodes (alpha_r).
        # 3. Call propagate function to conduct the message passing. 
        #    3.1 Remember to pass alpha = (alpha_l, alpha_r) as a parameter.
        #    3.2 See there for more information: https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html
        # 4. Transform the output back to the shape of [N, H * C].
        # Our implementation is ~5 lines, but don't worry if you deviate from this.


        ############################################################################
        
        H, C = self.heads, self.out_channels

        

        N = x.shape[0]
        h_prime = x.matmul(weights) 
        
        alpha_l = torch.sum(h_prime*self.att_l, dim=-1)
        alpha_r = torch.sum(h_prime*self.att_r, dim=-1)

         
        out = self.propagate(edge_index=edge_index, x=(h_prime, h_prime), alpha=(alpha_l, alpha_r), size=size)
        out = out.view((-1, H*C))
        

        return out


    def message(self,index, x_j, alpha_j, alpha_i, ptr, size_i):

        ############################################################################
        # TODO: Your code here! 
        # Implement your message function. Putting the attention in message 
        # instead of in update is a little tricky.
        # 1. Calculate the final attention weights using alpha_i and alpha_j,
        #    and apply leaky Relu.
        # 2. Calculate softmax over the neighbor nodes for all the nodes. Use 
        #    torch_geometric.utils.softmax instead of the one in Pytorch.
        # 3. Apply dropout to attention weights (alpha).
        # 4. Multiply embeddings and attention weights. As a sanity check, the output
        #    should be of shape [E, H, C].
        # 5. ptr (LongTensor, optional): If given, computes the softmax based on
        #    sorted inputs in CSR representation. You can simply pass it to softmax.
        # Our implementation is ~4-5 lines, but don't worry if you deviate from this.
        E, H, C = self.in_channels, self.heads, self.out_channels
       
        
        final_attention_weights = torch.add(alpha_i, alpha_j)
        att_unnormalized = F.leaky_relu(final_attention_weights)
        att_weights = torch_geometric.utils.softmax(att_unnormalized, index=index, num_nodes=size_i, ptr=ptr, dim=-2)
        att_weights = torch.nn.functional.dropout(att_weights, p=self.dropout)
        out = x_j*att_weights.unsqueeze(-1)
        #assert(out.shape ==(E, H, C)), f"out shape: {out.shape}"
        ############################################################################

        return out


    def aggregate(self, inputs, index, dim_size = None):

        ############################################################################
        # TODO: Your code here! 
        # Implement your aggregate function here.
        # See here as how to use torch_scatter.scatter: https://pytorch-scatter.readthedocs.io/en/latest/_modules/torch_scatter/scatter.html
        # Pay attention to "reduce" parameter is different from that in GraphSage.
        # Our implementation is ~1 lines, but don't worry if you deviate from this.

        ############################################################################
        node_dim = self.node_dim
        out = torch_scatter.scatter(inputs, index, dim=node_dim, reduce="sum")

        return out




class GAT(torch.nn.Module):
    def __init__(self, activation, out_features):
        super().__init__()
        self.activation = activation
        attention_dim = 2
        self.w_a = Parameter(torch.Tensor(out_features, attention_dim))
        self.a = Parameter(torch.Tensor(2*attention_dim, 1))

        u.reset_param(self.w_a)
        u.reset_param(self.a)
        self.alpha = 0.001

    def forward(self, node_feats, Ahat, w):
        N = Ahat.shape[0]
        h_prime = node_feats.matmul(w) 
        h_reduced = h_prime.matmul(self.w_a)
        H1 = h_reduced.unsqueeze(1).repeat(1,N,1)
        H2 = h_reduced.unsqueeze(0).repeat(N,1,1)
        attn_input = torch.cat([H1, H2], dim = -1) # (N, N, F)
        e = attn_input.matmul(self.a).squeeze(-1) # [N, N]
        attn_mask = -1e18*torch.ones_like(e)
        masked_e = torch.where(Ahat.to_dense() > 0, e, attn_mask)
        attn_scores = F.softmax(masked_e, dim = -1) # [N, N]

        h_prime = torch.mm(attn_scores, h_prime)
        out = self.activation(h_prime)
        return out