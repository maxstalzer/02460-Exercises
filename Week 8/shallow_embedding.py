# Programming excercise: Shallow embedding

# Import libraries
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

# Device
device = 'cpu'

# Load graph data
# Load graph from file
A = torch.load('data.pt')

# It is standard practice to convert the PyTorch tensor to a NumPy array for plotting
A_numpy = A.numpy()

# Set up the figure size
plt.figure(figsize=(8, 8))

# Plot the heatmap. 'Greys' is a great colormap for a binary (0 and 1) matrix.
# interpolation='none' ensures the squares stay crisp and don't blur together.
plt.imshow(A_numpy, cmap='Greys', interpolation='none') 

plt.title('Adjacency Matrix')
plt.xlabel('Node Index')
plt.ylabel('Node Index')
plt.colorbar(label='Edge (1) / No Edge (0)')
plt.show()

# Get number of nodes
n_nodes = A.shape[0]

# Number of un-ordered node pairs (possible links)
n_pairs = n_nodes*(n_nodes-1)//2

# Get indices of all un-ordered node pairs excluding self-links (shape: 2 x n_pairs)
idx_all_pairs = torch.triu_indices(n_nodes,n_nodes,1)

# Collect all links/non-links in a list (shape: n_pairs)
target = A[idx_all_pairs[0],idx_all_pairs[1]]

# Create a random permutation of all pair indices
shuffled_indices = torch.randperm(n_pairs)

# Find the index that represents 80% of the data
split_point = int(0.8 * n_pairs)

# Split the indices
train_indices = shuffled_indices[:split_point]
val_indices = shuffled_indices[split_point:]

# Create the training sets
rx_train, tx_train = idx_all_pairs[0, train_indices], idx_all_pairs[1, train_indices]
target_train = target[train_indices]

# Create the validation sets
rx_val, tx_val = idx_all_pairs[0, val_indices], idx_all_pairs[1, val_indices]
target_val = target[val_indices]

# Shallow node embedding
class Shallow(torch.nn.Module):
    '''Shallow node embedding

    Args: 
        n_nodes (int): Number of nodes in the graph
        embedding_dim (int): Dimension of the embedding
    '''
    def __init__(self, n_nodes, embedding_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(n_nodes, embedding_dim=embedding_dim)
        self.bias = torch.nn.Parameter(torch.tensor([0.]))

    def forward(self, rx, tx):
        '''Returns the probability of a links between nodes in lists rx and tx'''
        return torch.sigmoid((self.embedding.weight[rx]*self.embedding.weight[tx]).sum(1) + self.bias)

# Embedding dimension
embedding_dim = 2

# Instantiate the model                
model = Shallow(n_nodes, embedding_dim)

import copy # Add this to your imports at the top!

def train_and_evaluate(emb_dim, max_steps=5000, patience=200):
    model = Shallow(n_nodes, emb_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)
    cross_entropy = torch.nn.BCELoss()
    
    val_loss_history = []
    
    # --- EARLY STOPPING TRACKERS ---
    best_val_loss = float('inf')
    patience_counter = 0
    best_step = 0
    best_model_state = None
    
    for step in range(max_steps):
        # --- TRAINING PHASE ---
        model.train() 
        link_probability = model(rx_train, tx_train)
        loss = cross_entropy(link_probability, target_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # --- VALIDATION PHASE ---
        model.eval() 
        with torch.no_grad(): 
            val_probability = model(rx_val, tx_val)
            val_loss = cross_entropy(val_probability, target_val)
            val_loss_value = val_loss.item()
            val_loss_history.append(val_loss_value)
            
            # --- EARLY STOPPING LOGIC ---
            if val_loss_value < best_val_loss:
                # We found a new minimum! Save the state.
                best_val_loss = val_loss_value
                best_step = step
                patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                # Loss didn't improve. Increase patience.
                patience_counter += 1
                
            # If we haven't seen an improvement in 'patience' steps, halt.
            if patience_counter >= patience:
                print(f"    -> Early stopping at step {step}. Best was step {best_step}.")
                break
                
    # Restore the model to its absolute best state before returning
    model.load_state_dict(best_model_state)
    
    return val_loss_history, best_val_loss, best_step, model

# Define the dimensions to test
dimensions_to_test = [4, 8, 16]

plt.figure(figsize=(10, 6))

best_overall_model = None
best_overall_loss = float('inf')

# Train and plot the *Validation* curve for each dimension
for dim in dimensions_to_test:
    print(f"Evaluating embedding dimension: {dim}...")
    # I bumped max_steps up since early stopping will catch it anyway
    val_history, best_loss, best_step, trained_model = train_and_evaluate(dim, max_steps=5000, patience=300)
    
    print(f"  -> Final Best Validation Loss: {best_loss:.4f} (Found at step {best_step})")
    plt.plot(val_history, label=f'Dim = {dim} (Stop: {len(val_history)})')
    
    # Keep track of the ultimate winner across all dimensions
    if best_loss < best_overall_loss:
        best_overall_loss = best_loss
        best_overall_model = trained_model

plt.title('Validation Loss vs. Embedding Dimension (with Early Stopping)')
plt.xlabel('Gradient Steps')
plt.ylabel('Binary Cross-Entropy Loss (Validation)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Save final estimated link probabilities using the WINNING model
best_overall_model.eval()
with torch.no_grad():
    link_probability = best_overall_model(idx_all_pairs[0], idx_all_pairs[1])
    
torch.save(link_probability, 'link_probability.pt')
print(f"Saved predictions from best model with loss {best_overall_loss:.4f}")