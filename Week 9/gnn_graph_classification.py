# %%
import torch
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import wandb

# %% Interactive plots
plt.ion() # Enable interactive plotting
def drawnow():
    """Force draw the current plot."""
    plt.gcf().canvas.draw()
    plt.gcf().canvas.flush_events()

# %% Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# %% Load the MUTAG dataset
# Load data
dataset = TUDataset(root='./data/', name='MUTAG')
node_feature_dim = 7

# Split into training and validation
rng = torch.Generator().manual_seed(0)
train_dataset, validation_dataset, test_dataset = random_split(dataset, (100, 44, 44), generator=rng)

# Create dataloader for training and validation
train_loader = DataLoader(train_dataset, batch_size=100)
validation_loader = DataLoader(validation_dataset, batch_size=44)
test_loader = DataLoader(test_dataset, batch_size=44)

# %% Question C.1: Examine the data
data_batch = next(iter(train_loader))
print("--- Question C.1 ---")
print(f"Node features (data_batch.x) shape: {data_batch.x.shape}")
print(f"Edges (data_batch.edge_index) shape: {data_batch.edge_index.shape}")
print(f"Graph assignments (data_batch.batch) shape: {data_batch.batch.shape}")
print(f"Number of graphs in this batch: {data_batch.batch.max().item() + 1}")
print("--------------------")

# %% Define a simple GNN for graph classification
class SimpleGNN(torch.nn.Module):
    """Simple graph neural network for graph classification

    Keyword Arguments
    -----------------
        node_feature_dim : Dimension of the node features
        state_dim : Dimension of the node states
        num_message_passing_rounds : Number of message passing rounds
    """

    def __init__(self, node_feature_dim, state_dim, num_message_passing_rounds, dropout_rate=0.0):
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.state_dim = state_dim
        self.num_message_passing_rounds = num_message_passing_rounds

        # Input network
        self.input_net = torch.nn.Sequential(
            torch.nn.Linear(self.node_feature_dim, self.state_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate) 
            )

        # Message networks
        self.message_net = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(self.state_dim, self.state_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=dropout_rate) 
            ) for _ in range(num_message_passing_rounds)])
        
        # Update network - Replaced with GRU 
        self.update_net = torch.nn.ModuleList([
            torch.nn.GRUCell(self.state_dim, self.state_dim) 
            for _ in range(num_message_passing_rounds)
        ])

        # State output network
        self.output_net = torch.nn.Linear(self.state_dim, 1)

        
    def forward(self, x, edge_index, batch):
        """Evaluate neural network on a batch of graphs.

        Parameters
        ----------
        x : torch.tensor (num_nodes x num_features)
            Node features.
        edge_index : torch.tensor (2 x num_edges)
            Edges (to-node, from-node) in all graphs.
        batch : torch.tensor (num_nodes)
            Index of which graph each node belongs to.

        Returns
        -------
        out : torch tensor (num_graphs)
            Neural network output for each graph.

        """
        # Extract number of nodes and graphs
        num_graphs = batch.max()+1
        num_nodes = batch.shape[0]

        # Initialize node state from node features
        state = self.input_net(x)
        # state = x.new_zeros([num_nodes, self.state_dim]) # Uncomment to disable the use of node features

        # Loop over message passing rounds
        for r in range(self.num_message_passing_rounds):
            # Compute outgoing messages
            message = self.message_net[r](state)

            # Aggregate: Sum messages
            aggregated = x.new_zeros((num_nodes, self.state_dim))
            aggregated = aggregated.index_add(0, edge_index[1], message[edge_index[0]])

            # Update states
            state = self.update_net[r](aggregated, state)

        # Aggretate: Sum node features
        graph_state = x.new_zeros((num_graphs, self.state_dim))
        graph_state = torch.index_add(graph_state, 0, batch, state)

        # Output
        out = self.output_net(graph_state).flatten()
        return out

# # 1. Define the Sweep Configuration
# # 'bayes' is much smarter than 'grid'. It uses previous runs to guess where the lowest loss is.
# sweep_config = {
#     'method': 'bayes',
#     'metric': {
#         'name': 'val_loss',
#         'goal': 'minimize'   
#     },
#     'parameters': {
#         'state_dim': {'values': [16, 32, 64]},
#         'epochs': {'value': 1000},
#         'num_message_passing_rounds': {'values': [2, 3, 4, 5]},
#         'learning_rate': {'distribution': 'log_uniform_values', 'min': 1e-4, 'max': 5e-2},
#         'weight_decay': {'distribution': 'log_uniform_values', 'min': 1e-6, 'max': 1e-3},
#         'dropout_rate': {'values': [0.0, 0.3, 0.5, 0.7]}
#     }
# }

# # (Note: You must update your SimpleGNN __init__ to accept 'dropout_rate' 
# # and pass it to your torch.nn.Dropout layers!)

# # 2. Wrap your training loop in a single function
# def train_sweep():
#     # Initialize a new wandb run
#     wandb.init()
    
#     # WandB injects the hyperparameters for this specific run here
#     config = wandb.config
    
#     model = SimpleGNN(
#         node_feature_dim=node_feature_dim, 
#         state_dim=config.state_dim, 
#         num_message_passing_rounds=config.num_message_passing_rounds,
#         dropout_rate=config.dropout_rate # Make sure you added this to SimpleGNN!
#     ).to(device)
    
#     cross_entropy = torch.nn.BCEWithLogitsLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
#     scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    
#     epochs = config.epochs

#     for epoch in range(epochs):
#         # --- TRAINING ---
#         model.train()
#         train_accuracy, train_loss = 0., 0.
#         for data in train_loader:
#             data = data.to(device)
#             out = model(data.x, data.edge_index, batch=data.batch)
#             loss = cross_entropy(out, data.y.float())
            
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#             train_accuracy += sum((out>0) == data.y).detach().cpu() / len(train_loader.dataset)
#             train_loss += loss.detach().cpu().item() * data.batch_size / len(train_loader.dataset)
            
#         scheduler.step()
        
#         # --- VALIDATION ---
#         model.eval()
#         validation_loss, validation_accuracy = 0., 0.
#         with torch.no_grad():
#             for data in validation_loader:
#                 data = data.to(device)
#                 out = model(data.x, data.edge_index, data.batch)
#                 validation_accuracy += sum((out>0) == data.y).cpu() / len(validation_loader.dataset)
#                 validation_loss += cross_entropy(out, data.y.float()).cpu().item() * data.batch_size / len(validation_loader.dataset)
        
#         # 3. Log metrics to WandB instead of Matplotlib
#         wandb.log({
#             'epoch': epoch,
#             'train_loss': train_loss,
#             'train_acc': train_accuracy,
#             'val_loss': validation_loss,
#             'val_acc': validation_accuracy,
#             'lr': scheduler.get_last_lr()[0]
#         })

# # 4. Initialize and run the sweep
# # This will run 30 different combinations, using Bayesian optimization to find the best one
# sweep_id = wandb.sweep(sweep_config, project="gnn-mutag-sweep")
# wandb.agent(sweep_id, train_sweep, count=30)

# ==========================================
# FINAL RUN (Run this AFTER the sweep finds the best parameters)
# ==========================================

# 1. Hardcode your winning parameters from the WandB dashboard here:
best_state_dim = 64 # (Replace with your winner)
best_rounds = 5 # (Replace with your winner)
best_dropout = 0.3 # (Replace with your winner)
best_lr = 0.000185696208571603 # (Replace with your winner)
best_wd = 0.000757386170593012 # (Replace with your winner)
best_epochs = 536 # (Look at WandB: what epoch hit the lowest val_loss before it started rising?)

print("Training final model with best hyperparameters...")

# 2. Initialize the final model
final_model = SimpleGNN(node_feature_dim, best_state_dim, best_rounds, best_dropout).to(device)
cross_entropy = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(final_model.parameters(), lr=best_lr, weight_decay=best_wd)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

# 3. Train strictly on the train_loader (respecting the split constraints)
for epoch in range(best_epochs):
    final_model.train()
    for data in train_loader:
        data = data.to(device)
        out = final_model(data.x, data.edge_index, batch=data.batch)
        loss = cross_entropy(out, data.y.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(loss)
    scheduler.step()

# 4. Save final predictions (The ONLY time we look at the test set)
print("Saving final predictions...")
final_model.eval()
with torch.no_grad():
    
    test_data = next(iter(test_loader)).to(device)
    out = final_model(test_data.x, test_data.edge_index, test_data.batch).cpu()
    torch.save(out, 'test_predictions.pt')

print("Done! Ready to submit.")