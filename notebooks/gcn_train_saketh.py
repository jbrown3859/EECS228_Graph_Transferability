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


class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # Step 1: Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Multiply with weights
        x = self.lin(x)

        # Step 3: Calculate the normalization
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4: Propagate the embeddings to the next layer
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x,
                              norm=norm)

    def message(self, x_j, norm):
        # Normalize node features.
        return norm.view(-1, 1) * x_j


class Net(torch.nn.Module):
    def __init__(self, dataset, n_in):
        super(Net, self).__init__()
        self.conv1 = GCNConv(n_in, 64)
        self.conv2 = GCNConv(64, 64)
        self.out = torch.nn.Linear(64, dataset.num_classes)
        self.hook = self.conv2.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.intermediate_output = output

    def forward(self, data, source_features_reduced):
        x, edge_index = source_features_reduced, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.out(x)
        return F.log_softmax(x, dim=1)


def plot_dataset(dataset):
    edges_raw = dataset.data.edge_index.numpy()
    edges = [(x, y) for x, y in zip(edges_raw[0, :], edges_raw[1, :])]
    labels = dataset.data.y.numpy()

    G = nx.Graph()
    G.add_nodes_from(list(range(np.max(edges_raw))))
    G.add_edges_from(edges)
    plt.subplot(111)
    options = {
                'node_size': 30,
                'width': 0.2,
    }
    nx.draw(G, with_labels=False, node_color=labels.tolist(), cmap=plt.cm.tab10, font_weight='bold', **options)
    plt.show()


def test(data, source_features_reduced, train=True):
    model.eval()
    correct = 0
    pred = model(data, source_features_reduced).max(dim=1)[1]
    if train:
        correct += pred[data.train_mask].eq(data.y[data.train_mask]).sum().item()
        return correct / (len(data.y[data.train_mask]))
    else:
        correct += pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
        return correct / (len(data.y[data.test_mask]))


def train(data, source_features_reduced, epochs, plot=False):
    train_accuracies, test_accuracies = list(), list()
    best_test_acc = 0
    for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            out = model(data, source_features_reduced)
            # print(out.shape)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            train_acc = test(data, source_features_reduced)
            test_acc = test(data, source_features_reduced, train=False)
            if (test_acc > best_test_acc):
                best_test_acc = test_acc
                print("Test accuracy", test_acc)
                torch.save(model.state_dict(), "/DATA/saketh/graph_transferability/source_models/Citeseer_best_GCN.pth")

            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
            print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Test Acc: {:.5f}'.
                  format(epoch, loss, train_acc, test_acc))

    if plot:
        plt.plot(train_accuracies, label="Train accuracy")
        plt.plot(test_accuracies, label="Validation accuracy")
        plt.xlabel("# Epoch")
        plt.ylabel("Accuracy")
        plt.legend(loc='upper right')
        plt.show()


if __name__ == "__main__":
    # dataset = Planetoid(root='/tmp/Cora', name='Cora')
    dataset = Planetoid(root='/DATA/saketh/graph_transferability/scripts/data/planetoid', name='CiteSeer')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    source_features = dataset[0].x.numpy()
    n_in = 256
    pca_source = PCA(n_components=n_in)
    source_features_reduced = torch.from_numpy(pca_source.fit_transform(source_features))
    source_features_reduced = source_features_reduced.to(device)
    model = Net(dataset, n_in).to(device)
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    train(data, source_features_reduced, epochs=150, plot=False)
