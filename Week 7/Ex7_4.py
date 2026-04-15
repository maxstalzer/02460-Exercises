import torch
import tqdm
import matplotlib.pyplot as plt

def optimize_geodesic(z_start, z_intermediate, z_end, optimizer, num_steps=1000):
    with tqdm.tqdm(range(num_steps)) as pbar:
        for step in pbar:
            try:
                optimizer.zero_grad()
                
                # Rebuild the full curve to connect the fixed endpoints with the variable inner points
                z_full_curve = torch.cat([z_start, z_intermediate, z_end], dim=0)
                
                # Compute the energy
                loss = compute_energy(z_full_curve)
                
                # Backpropagate and take an optimizer step
                loss.backward()
                optimizer.step()

                # Report progress
                if step % 100 == 0:
                    pbar.set_description(f"step={step}, energy={loss.item():.4f}")

            except KeyboardInterrupt:
                print(f"Stopping optimization at step {step}")
                break
                
    # Return the final optimized curve (detached from gradient graph)
    final_curve = torch.cat([z_start, z_intermediate, z_end], dim=0).detach()
    return final_curve

# def compute_energy(z_curve):
#     # Calculate segments (dt) dynamically so PyTorch tracks the gradients
#     delta_theta = z_curve[1:] - z_curve[:-1] 
#     dt1 = delta_theta[:, 0]
#     dt2 = delta_theta[:, 1]

#     # Calculate midpoints (t) dynamically
#     midpoints = (z_curve[1:] + z_curve[:-1]) / 2.0
#     t1 = midpoints[:, 0]
#     t2 = midpoints[:, 1]

#     # The simplified quadratic form for the metric you derived
#     segment_energies = 0.5 * (t2 * dt1 + t1 * dt2)**2
#     energy = segment_energies.sum()
    
#     return energy

def compute_energy(z_curve):
    # 1. Calculate segments (dt)
    delta_theta = z_curve[1:] - z_curve[:-1] 
    dt1 = delta_theta[:, 0]
    dt2 = delta_theta[:, 1]

    # 2. Calculate midpoints (t)
    midpoints = (z_curve[1:] + z_curve[:-1]) / 2.0
    t1 = midpoints[:, 0]
    t2 = midpoints[:, 1]

    # 3. Calculate Jacobian components A and B evaluated at the midpoints
    A = t2 * (1.0 - torch.tanh(t1)**2)
    B = torch.tanh(t1)

    # 4. Compute the quadratic form based on the tanh metric
    # (Notice there is no 0.5 factor this time!)
    segment_energies = (A * dt1 + B * dt2)**2
    energy = segment_energies.sum()
    
    return energy

# 1. Define Endpoints (must be floats and 2D for concatenation)
z_start = torch.tensor([[1.0, 1.0]])
z_end = torch.tensor([[21.0, 3.0/7.0]])

# 2. Initialize the straight line
num_t = 20
t = torch.linspace(0, 1, num_t).view(-1, 1).to("cpu")
z_curve_init = (1 - t) * z_start + t * z_end

# 3. Extract intermediate points and tell PyTorch to track their gradients
z_intermediate = z_curve_init[1:-1].clone().detach().requires_grad_(True)

# 4. Calculate initial energy for comparison
with torch.no_grad():
    init_energy = compute_energy(z_curve_init).item()

# 5. Setup optimizer for the intermediate points
curve_optimizer = torch.optim.Adam([z_intermediate], lr=5e-3)

# 6. Run the optimization
optimized_curve = optimize_geodesic(
    z_start=z_start,
    z_intermediate=z_intermediate,
    z_end=z_end,
    optimizer=curve_optimizer,
    num_steps=2000
)

# 7. Evaluate and Print Results
with torch.no_grad():
    final_energy = compute_energy(optimized_curve).item()
    
euc_dist = torch.norm(z_start - z_end).item()
reduction = (1 - final_energy / init_energy) * 100 if init_energy > 0 else 0.0

print(f"\nResults:")
print(f"Energy: {init_energy:.4f} → {final_energy:.4f} ({reduction:.1f}% reduction)")
print(f"Standard Euclidean distance: {euc_dist:.4f}")

# 8. Plot the results
init_np = z_curve_init.cpu().detach().numpy()
opt_np = optimized_curve.cpu().detach().numpy()

plt.figure(figsize=(8, 6))
# Plot straight line in gray, dashed
plt.plot(init_np[:, 0], init_np[:, 1], color='gray', linestyle='--', alpha=0.5, label="Euclidean Straight Line")
# Plot the optimized geodesic in red, solid
plt.plot(opt_np[:, 0], opt_np[:, 1], color='red', linewidth=2, alpha=0.8, label="Riemannian Geodesic")
plt.scatter([1, 21], [1, 3/7], color='black', zorder=5, label="Endpoints")

plt.xlabel("Theta 1")
plt.ylabel("Theta 2")
plt.title("Optimization of ReLU Network Geodesic")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()