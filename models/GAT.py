import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        # Learnable weight matrix
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # Learnable attention parameters
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        # Linear transformation
        Wh = torch.mm(h, self.W)
        
        # Attention mechanism
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)

        # Aggregation using attention coefficients
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]  # Number of nodes
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        return all_combinations_matrix.view(N, N, 2 * self.out_features)


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout

        # Multiple attention heads
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # Final layer for classification
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        # Apply dropout to input features
        x = F.dropout(x, self.dropout, training=self.training)

        # Multi-head attention mechanism
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)

        # Apply dropout after attention aggregation
        x = F.dropout(x, self.dropout, training=self.training)

        # Final layer for classification
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

# Example usage
# Define your dataset, features, and adjacency matrix
# For simplicity, let's assume a random dataset with 10 nodes and 5 features
import torch.nn.functional as F

num_nodes = 10
num_features = 5
num_classes = 2
dropout = 0.5
alpha = 0.2
num_heads = 2

# Generate random features and adjacency matrix
features = torch.randn((num_nodes, num_features))
adjacency = torch.randint(2, (num_nodes, num_nodes)).float()  # Assuming a binary adjacency matrix

# Create the GAT model
model = GAT(nfeat=num_features, nhid=8, nclass=num_classes, dropout=dropout, alpha=alpha, nheads=num_heads)

# Forward pass
output = model(features, adjacency)
print(output)