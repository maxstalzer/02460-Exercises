# Programming exercise: Shallow embedding (Final Competition Script)

# Import libraries
import torch
from tqdm import tqdm

# Device
device = 'cpu'

# --- 1. DATA LOADING & PREP ---
print("Loading data...")
A = torch.load('data.pt')
n_nodes = A.shape[0]
n_pairs = n_nodes * (n_nodes - 1) // 2

# Get indices of all un-ordered node pairs (100% of the data)
idx_all_pairs = torch.triu_indices(n_nodes, n_nodes, 1)

# Collect all targets
target = A[idx_all_pairs[0], idx_all_pairs[1]]

# --- 2. MODEL DEFINITION ---
class Shallow(torch.nn.Module):
    '''Shallow node embedding'''
    def __init__(self, n_nodes, embedding_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(n_nodes, embedding_dim=embedding_dim)
        self.bias = torch.nn.Parameter(torch.tensor([0.]))

    def forward(self, rx, tx):
        '''Returns the probability of a link between nodes in lists rx and tx'''
        return torch.sigmoid((self.embedding.weight[rx] * self.embedding.weight[tx]).sum(1) + self.bias)

# --- 3. HYPERPARAMETERS & SETUP ---
# EMPIRICAL WINNER: Dimension 8, peaking at exactly 601 steps.
embedding_dim = 8  
max_step = 601

print(f"Initializing model with dimension {embedding_dim} for {max_step} steps...")
model = Shallow(n_nodes, embedding_dim)

# The "Hack": weight_decay adds L2 regularization to prevent extreme probabilities
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)
cross_entropy = torch.nn.BCELoss()

# --- 4. FINAL TRAINING LOOP (100% DATA) ---
print("Training final model...")
model.train()

for i in (progress_bar := tqdm(range(max_step))):    
    link_probability = model(idx_all_pairs[0], idx_all_pairs[1])
    loss = cross_entropy(link_probability, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Display loss on progress bar
    progress_bar.set_description(f'Loss = {loss.item():.4f}')

# --- 5. COMPUTE & SAVE PREDICTIONS ---
print("Saving final predictions...")
model.eval() # Set to evaluation mode for final inference
with torch.no_grad():
    final_link_probability = model(idx_all_pairs[0], idx_all_pairs[1])

torch.save(final_link_probability, 'link_probability.pt')
print("Complete! Ready to hand in 'link_probability.pt' on DTU Learn.")