import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.datasets import TUDataset
from sklearn.model_selection import train_test_split

# Load the PROTEINS dataset
dataset = TUDataset(root='/tmp/PROTEINS', name='PROTEINS')

# Define the GCNConv class
class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation.
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
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, norm=norm)

    def message(self, x_j, norm):
        # Normalize node features
        return norm.view(-1, 1) * x_j

# Define the GCN class
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_neurons, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_neurons)
        self.conv2 = GCNConv(hidden_neurons, hidden_neurons)
        self.fc = torch.nn.Linear(hidden_neurons, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        # Global Pooling
        x = global_mean_pool(x, batch)

        # Final classification layer
        x = self.fc(x)

        return F.log_softmax(x, dim=1)

# Device configuration
device ="cpu"#torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Initialize the model with the specified number of hidden neurons
model = GCN(dataset.num_node_features,16, dataset.num_classes).to(device)

# Split the dataset
train_dataset, test_dataset = train_test_split(dataset, test_size=0.1)
train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.1)

# Training and Testing Functions
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
# criterion = torch.nn.CrossEntropyLoss()
criterion = F.nll_loss
def train():
    model.train()
    total_loss = 0
    for data in train_dataset:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_dataset)

def test(dataset):
    model.eval()
    correct = 0
    total = 0
    for data in dataset:
        data = data.to(device)
        with torch.no_grad():
            logits = model(data)
            pred = logits.max(1)[1]
            correct += pred.eq(data.y).sum().item()
            total += data.y.size(0)
    return correct / total

# Training Loop
epochs = 200
train_losses = []
val_accuracies = []
test_accuracies = []

for epoch in range(1, epochs + 1):
    train_loss = train()
    val_acc = test(val_dataset)
    test_acc = test(test_dataset)

    # Record metrics
    train_losses.append(train_loss)
    val_accuracies.append(val_acc)
    test_accuracies.append(test_acc)
    if(epoch%10==0 or epoch ==(epochs)):
            print(f'Epoch: {epoch+1}, Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
            
# Plot training loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.legend()

# Plot validation and test accuracy
plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation and Test Accuracy Over Time')
plt.legend()

plt.tight_layout()
plt.show()

