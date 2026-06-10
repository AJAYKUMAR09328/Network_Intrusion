import torch
import torch.nn as nn
import torch.nn.functional as F

class TabularGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(TabularGNN, self).__init__()
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.fc_msg = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)
    def forward(self, x):
        h = F.relu(self.fc_in(x))
        h_norm = F.normalize(h, p=2, dim=1)     
        adj = torch.mm(h_norm, h_norm.t())
        threshold = 0.5
        adj = (adj > threshold).float()
        adj = adj + torch.eye(adj.size(0), device=adj.device)
        degree = adj.sum(dim=1)
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(degree + 1e-3))
        adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt  
        support = self.fc_msg(h) 
        h_graph = adj_norm @ support    
        h_graph = F.relu(h_graph)   
        h_graph = self.dropout(h_graph)  
        out = self.fc_out(h_graph) 
        return out


