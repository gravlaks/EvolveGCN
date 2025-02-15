import utils as u
import torch
from torch.nn.parameter import Parameter

import torch.nn as nn
import math
from egcn_h_MP import GAT_MP, MP
from torch.nn import functional as F

class EGCN(torch.nn.Module):
    def __init__(self, args, activation, device='cpu', skipfeats=False, gat=False):
        super().__init__()
        GRCU_args = u.Namespace({})

        feats = [args.feats_per_node,
                 args.layer_1_feats,
                 args.layer_2_feats]
        self.device = device
        self.skipfeats = skipfeats
        self.GRCU_layers = []
        self._parameters = nn.ParameterList()
        for i in range(1,len(feats)):
            GRCU_args = u.Namespace({'in_feats' : feats[i-1],
                                     'out_feats': feats[i],
                                     'activation': activation, 
                                     'device':device})
            grcu_i = GRCU(GRCU_args, gat, args.recurrent_unit)
            self.GRCU_layers.append(grcu_i.to(self.device))
            self._parameters.extend(list(self.GRCU_layers[-1].parameters()))

    def parameters(self):
        return self._parameters

    def forward(self,A_list, Nodes_list,nodes_mask_list, edge_weights=None):
        node_feats= Nodes_list[-1]

        for unit in self.GRCU_layers:
            Nodes_list = unit(A_list,Nodes_list,nodes_mask_list, edge_weights)

        out = Nodes_list[-1]
        if self.skipfeats:
            out = torch.cat((out,node_feats), dim=1)   # use node_feats.to_dense() if 2hot encoded input 
        return out





class GRCU(torch.nn.Module):
    def __init__(self,args, gat=True, recurrent_unit="gru"):
        super().__init__()
        self.args = args
        cell_args = u.Namespace({})
        cell_args.rows = args.in_feats
        cell_args.cols = args.out_feats

        hidden_size =  args.in_feats*args.out_feats
        self.recurrent_unit = recurrent_unit
        if recurrent_unit == "gru":
            self.evolve_weights = torch.nn.GRUCell(args.in_feats, hidden_size)
        elif recurrent_unit == "lstm":
            self.evolve_weights = torch.nn.LSTMCell(args.in_feats, hidden_size)
        elif recurrent_unit == "original":
            self.evolve_weights = mat_GRU_cell(cell_args)
        #self.evolve_weights = mat_GRU_cell(cell_args)

        self.GCN_init_weights = Parameter(torch.Tensor(self.args.in_feats,self.args.out_feats))
        self.cell_state_init = torch.zeros((self.GCN_init_weights.flatten().shape)).to(self.args.device)

        self.reset_param(self.GCN_init_weights)

        if gat:
            self.conv = GAT_MP(out_channels=self.args.out_feats)
        else:
            self.conv = MP(in_channels = self.args.in_feats, out_channels=self.args.out_feats)

    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)

    def forward(self,A_list,node_embs_list,mask_list, edge_weights):
        GCN_weights = self.GCN_init_weights
        cell_state = self.cell_state_init
        out_seq = []
        for t, (edge_index, edge_weight) in enumerate(zip(A_list, edge_weights)):
            node_embs = node_embs_list[t].to_dense()
            if self.recurrent_unit == "original":
                GCN_weights = self.evolve_weights(GCN_weights,node_embs,mask_list[t])
                
            else:
           
                mask = mask_list[t].flatten()
                #node_embs = node_embs_list[t].to_dense()

                input_GRU = torch.sum(torch.mul(torch.softmax(mask, dim=0), node_embs.t()), axis=1)
                hidden_GRU = GCN_weights.flatten()
                
                if self.recurrent_unit == "gru":
                    GCN_weights = self.evolve_weights(input_GRU, hidden_GRU)
                elif self.recurrent_unit == "lstm":
                    GCN_weights, cell_state = self.evolve_weights(input_GRU, (hidden_GRU, cell_state))

                GCN_weights = GCN_weights.reshape(self.GCN_init_weights.shape)
            node_embs = self.conv(node_embs, edge_index, weights=GCN_weights, edge_weights=edge_weight)
            node_embs = self.args.activation(node_embs)

            out_seq.append(node_embs)
            
        return out_seq





class mat_GRU_cell(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.update = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Sigmoid())

        self.reset = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Sigmoid())

        self.htilda = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Tanh())
        
        self.choose_topk = TopK(feats = args.rows,
                                k = args.cols)

    def forward(self,prev_Q,prev_Z,mask):
        z_topk = self.choose_topk(prev_Z,mask)

        update = self.update(z_topk,prev_Q)
        reset = self.reset(z_topk,prev_Q)

        h_cap = reset * prev_Q
        h_cap = self.htilda(z_topk, h_cap)

        new_Q = (1 - update) * prev_Q + update * h_cap

        return new_Q

        

class mat_GRU_gate(torch.nn.Module):
    def __init__(self,rows,cols,activation):
        super().__init__()
        self.activation = activation
        #the k here should be in_feats which is actually the rows
        self.W = Parameter(torch.Tensor(rows,rows))
        self.reset_param(self.W)

        self.U = Parameter(torch.Tensor(rows,rows))
        self.reset_param(self.U)

        self.bias = Parameter(torch.zeros(rows,cols))

    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)

    def forward(self,x,hidden):
        out = self.activation(self.W.matmul(x) + \
                              self.U.matmul(hidden) + \
                              self.bias)

        return out

class TopK(torch.nn.Module):
    def __init__(self,feats,k):
        super().__init__()
        self.scorer = Parameter(torch.Tensor(feats,1))
        self.reset_param(self.scorer)
        
        self.k = k

    def reset_param(self,t):
        #Initialize based on the number of rows
        stdv = 1. / math.sqrt(t.size(0))
        t.data.uniform_(-stdv,stdv)

    def forward(self,node_embs,mask):
        scores = node_embs.matmul(self.scorer) / self.scorer.norm()
        scores = scores + mask

        vals, topk_indices = scores.view(-1).topk(self.k)
        topk_indices = topk_indices[vals > -float("Inf")]

        if topk_indices.size(0) < self.k:
            topk_indices = u.pad_with_last_val(topk_indices,self.k)
            
        tanh = torch.nn.Tanh()

        if isinstance(node_embs, torch.sparse.FloatTensor) or \
           isinstance(node_embs, torch.cuda.sparse.FloatTensor):
            node_embs = node_embs.to_dense()

        out = node_embs[topk_indices] * tanh(scores[topk_indices].view(-1,1))

        #we need to transpose the output
        return out.t()
