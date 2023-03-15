import torch
print(torch.__version__)
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import utils as u
import torch_geometric
from torch_geometric.nn.conv import MessagePassing
import torch.nn as nn
import torch_scatter
    
class GAT_MP(MessagePassing):

    def __init__(self, out_channels, heads = 1,
                 negative_slope = 0.2, dropout = 0., **kwargs):
        super(GAT_MP, self).__init__(node_dim=0, **kwargs)

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
        

        H, C = self.heads, self.out_channels

        N = x.shape[0]

        print("HCN", H, C, N)
        print("x", x.shape)
        print("edge index", edge_index.shape)
        print("weights", weights.shape)
        print("matmul temp", (x.matmul(weights)).shape )
        h_prime = x.matmul(weights).view((N, H, C))
        
        alpha_l = torch.sum(h_prime*self.att_l, dim=-1)
        alpha_r = torch.sum(h_prime*self.att_r, dim=-1)

         
        out = self.propagate(edge_index=edge_index, x=(h_prime, h_prime), alpha=(alpha_l, alpha_r), size=size)
        out = out.view((-1, H*C))
        

        return out


    def message(self,index, x_j, alpha_j, alpha_i, ptr, size_i):

       
        
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