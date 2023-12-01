#if using colab, do !pip install torch_geometric before running this file
from torch_geometric.datasets import Planetoid
import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch.nn as nn
import pickle
import argparse
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GATConv #GATConv

parser = argparse.ArgumentParser()
parser.add_argument('--hc', type=int, default = 64, help = "number of hidden channels")
parser.add_argument('--lr', type=float, default = 0.01, help = "learning rate")
parser.add_argument('--decay', type=float, default = 5e-4, help = "decay rate")
parser.add_argument('--epochs', type=int, default = 1500, help = "epochs")
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


dataset = Planetoid(root='/DATA/saketh/graph_transferability/scripts/data/planetoid', name='PubMed')
source_features = dataset[0].x.numpy()
n_in = 256
pca_source = PCA(n_components=n_in)
source_features_reduced = torch.from_numpy(pca_source.fit_transform(source_features))
source_features_reduced = source_features_reduced.to(device)


# storing the graph in the data variable
data = dataset[0]  

# some statistics about the graph.
print(data)
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Is undirected: {data.is_undirected()}')


# GAT model
class GAT(torch.nn.Module):
    def __init__(self, n_in, hidden_channels):
        super(GAT, self).__init__()
        torch.manual_seed(42)

        # Initialize the layers
        self.conv1 = GATConv(n_in, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.out = Linear(hidden_channels, dataset.num_classes)
        self.hook = self.conv2.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.intermediate_output = output

    def forward(self, x, edge_index):
        # First Message Passing Layer (Transformation)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        # Second Message Passing Layer
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        # Output layer 
        x = F.softmax(self.out(x), dim=1)
        return x

# Initialize model
model = GAT(n_in, hidden_channels=args.hc)

# Use CPU
# device = torch.device("cpu")
model = model.to(device)
data = data.to(device)

# Initialize Optimizer
learning_rate = args.lr
decay = args.decay
optimizer = torch.optim.Adam(model.parameters(), 
                             lr=learning_rate, 
                             weight_decay=decay)
# Define loss function (CrossEntropyLoss for Classification Problems with 
# probability distributions)
criterion = torch.nn.CrossEntropyLoss()

def train():
      model.train()
      optimizer.zero_grad() 
      # Use all data as input, because all nodes have node features
      out = model(source_features_reduced, data.edge_index)  
      # Only use nodes with labels available for loss calculation --> mask
      loss = criterion(out[data.train_mask], data.y[data.train_mask])  
      loss.backward() 
      optimizer.step()
      return loss

def test():
      model.eval()
      # out = model(data.x, data.edge_index)
      out = model(source_features_reduced, data.edge_index)  
      # Use the class with highest probability.
      pred = out.argmax(dim=1)  
      # Check against ground-truth labels.
      test_correct = pred[data.test_mask] == data.y[data.test_mask]  
      # Derive ratio of correct predictions.
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  
      return test_acc

losses = []

best_test_acc = 0
for epoch in range(0, args.epochs+1):
    loss = train()
    losses.append(loss)
    test_acc = test()
    if (test_acc > best_test_acc):
        best_test_acc = test_acc
        print("Test accuracy Changed", test_acc)
        torch.save(model.state_dict(), "/DATA/saketh/graph_transferability/source_models/Pubmed_best_GAT.pth")
    if epoch % 100 == 0:
      print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
