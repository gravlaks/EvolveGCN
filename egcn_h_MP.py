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

    def __init__(self, out_channels, heads = 2,
                 negative_slope = 0.2, dropout = 0., **kwargs):
        super(GAT_MP, self).__init__(node_dim=0, **kwargs)

        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout


        self.att_l = Parameter(torch.zeros((1, heads, out_channels//heads)))
        self.att_r = Parameter(torch.zeros((1, heads, out_channels//heads)))
        self.reset_parameters()


    def reset_parameters(self):

        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)

    def forward(self, x, edge_index, size = None, weights=None, edge_weights = None):
        
        H, C = self.heads, self.out_channels
        N = x.shape[0]
        
        h_prime = x.matmul(weights).view((N, H, C//H))
        
        alpha_l = torch.sum(h_prime*self.att_l, dim=-1)
        alpha_r = torch.sum(h_prime*self.att_r, dim=-1)

        # print("x", x.shape)
        # print("edge_index", edge_index.shape)
        # print("N, H, C", N, H, C)
        # print("h_prime", h_prime.shape)
        
        out = self.propagate(edge_index=edge_index, x=(h_prime, h_prime), alpha=(alpha_l, alpha_r), size=size)
        
        out = out.view((N, C))
        
        # print("out", out.shape)
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


class MP(MessagePassing):
    
    def __init__(self, in_channels, out_channels, normalize = True,
                 bias = False, **kwargs):  
        super(MP, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        self.lin_l = nn.Linear(in_channels, out_channels, bias=bias)
        self.lin_r = nn.Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x, edge_index, weights, size = None):

        out = self.propagate(edge_index, size=size, x = (x, x))
        skip = x.matmul(weights) #self.lin_r(x) 
        out = out.matmul(weights) + skip

        if self.normalize:
          out = F.normalize(out)
          
        return out

    def message(self, x_j):

        out = x_j

        return out

    def aggregate(self, inputs, index, dim_size = None):

        node_dim = self.node_dim

        out = torch_scatter.scatter(inputs.to_dense(), index.to_dense(), node_dim, reduce="mean")

        return out