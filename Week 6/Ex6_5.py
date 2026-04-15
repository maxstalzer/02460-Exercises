import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import math

# ==========================================
# 1. Curve Models (Identical to Ex 6.4)
# ==========================================
class PiecewiseCurve(nn.Module):
    def __init__(self, start_pt, end_pt, num_segments=30):
        super().__init__()
        self.start_pt = start_pt
        self.end_pt = end_pt
        self.num_segments = num_segments
        
        t = torch.linspace(0, 1, num_segments + 1).view(-1, 1)
        init_points = start_pt + t * (end_pt - start_pt)
        self.inner_points = nn.Parameter(init_points[1:-1])

    def get_curve_and_velocity(self):
        c = torch.cat([self.start_pt.view(1, 2), self.inner_points, self.end_pt.view(1, 2)], dim=0)
        dt = 1.0 / self.num_segments
        c_dot = (c[1:] - c[:-1]) / dt
        c_mid = (c[1:] + c[:-1]) / 2.0
        return c_mid, c_dot, dt

class PolynomialCurve(nn.Module):
    def __init__(self, start_pt, end_pt, num_points=50):
        super().__init__()
        self.start_pt = start_pt
        self.end_pt = end_pt
        
        self.t = torch.linspace(0, 1, num_points).view(-1, 1)
        self.dt = 1.0 / (num_points - 1)
        
        self.w2 = nn.Parameter(torch.zeros(2))
        self.w3 = nn.Parameter(torch.zeros(2))

    def get_curve_and_velocity(self):
        w0 = self.start_pt
        w1 = self.end_pt - self.start_pt - self.w2 - self.w3
        
        c = w0 + w1*self.t + self.w2*(self.t**2) + self.w3*(self.t**3)
        c_dot = w1 + 2*self.w2*self.t + 3*self.w3*(self.t**2)
        
        return c, c_dot, self.dt

# ==========================================
# 2. The New Data-Driven Energy Function
# ==========================================
def compute_density_energy(c, c_dot, dt, X, sigma=0.1, epsilon=1e-4):
    """
    Computes Energy using the KDE metric: G_x = 1 / (p(x) + eps) * I
    """
    # 1. Calculate pairwise squared distances between curve points (c) and data points (X)
    # c shape: (num_points, 1, 2) | X shape: (1, N, 2) -> diff shape: (num_points, N, 2)
    diff = c.unsqueeze(1) - X.unsqueeze(0)
    sq_dist = torch.sum(diff**2, dim=2) # Shape: (num_points, N)
    
    # 2. Evaluate the Gaussian KDE
    norm_const = 1.0 / (2 * math.pi * sigma**2)
    gaussian_vals = norm_const * torch.exp(-sq_dist / (2 * sigma**2))
    p_x = torch.mean(gaussian_vals, dim=1) # Shape: (num_points,)
    
    # 3. Calculate Energy: E = sum( ||c_dot||^2 / (p(c) + eps) ) * dt
    norm_cdot_sq = torch.sum(c_dot**2, dim=1)
    energy = torch.sum(norm_cdot_sq / (p_x + epsilon)) * dt
    
    return energy

def train_geodesic(model, X, epochs=1500, lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        c, c_dot, dt = model.get_curve_and_velocity()
        
        loss = compute_density_energy(c, c_dot, dt, X)
        loss.backward()
        optimizer.step()
        
    return model

# ==========================================
# 3. Execution and Plotting
# ==========================================
if __name__ == "__main__":
    # Load the data
    print("Loading data...")
    X_np = np.load('data/toybanana.npy')
    X = torch.tensor(X_np, dtype=torch.float32)

    # Define start and end points across the "gap" of the banana
    # This forces the curve to choose between the short empty path, or the long dense path
    start_point = torch.tensor([-1.5, -0.5])
    end_point = torch.tensor([1.5, -0.5])

    # Train Piecewise model
    print("Training Piecewise model...")
    piecewise_model = PiecewiseCurve(start_point, end_point, num_segments=40)
    train_geodesic(piecewise_model, X, epochs=1500, lr=0.01)

    # Train Polynomial model
    print("Training Polynomial model...")
    poly_model = PolynomialCurve(start_point, end_point, num_points=50)
    train_geodesic(poly_model, X, epochs=1500, lr=0.01)

    # Extract final optimized paths
    with torch.no_grad():
        c_piece, _, _ = piecewise_model.get_curve_and_velocity()
        c_poly, _, _ = poly_model.get_curve_and_velocity()
        c_piece = torch.cat([start_point.view(1,2), c_piece, end_point.view(1,2)])

    # Plot
    plt.figure(figsize=(10, 6))
    
    # Plot the data manifold
    plt.scatter(X_np[:, 0], X_np[:, 1], color='lightgray', s=10, label='Data Manifold p(x)')
    
    # Plot straight line for reference
    plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 
             'k--', alpha=0.5, label='Straight Euclidean Line')
    
    # Plot our optimized geodesics
    plt.plot(c_piece[:, 0].numpy(), c_piece[:, 1].numpy(), 
             'r-o', markersize=3, label='Piecewise Geodesic')
    plt.plot(c_poly[:, 0].numpy(), c_poly[:, 1].numpy(), 
             'b-', linewidth=2, label='Polynomial Geodesic')
    
    plt.scatter([start_point[0], end_point[0]], [start_point[1], end_point[1]], 
                color='black', s=100, zorder=5)
    
    plt.title("Geodesics on KDE Data Manifold (Toy Banana)")
    plt.legend()
    plt.axis('equal')
    plt.show()