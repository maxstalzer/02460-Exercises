import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ==========================================
# Part 1: Piecewise Straight Lines
# ==========================================
class PiecewiseCurve(nn.Module):
    def __init__(self, start_pt, end_pt, num_segments=30):
        super().__init__()
        self.start_pt = start_pt
        self.end_pt = end_pt
        self.num_segments = num_segments
        
        # Initialize as a straight Euclidean line
        t = torch.linspace(0, 1, num_segments + 1).view(-1, 1)
        init_points = start_pt + t * (end_pt - start_pt)
        
        # We only optimize the points in the middle! The endpoints must stay fixed.
        self.inner_points = nn.Parameter(init_points[1:-1])

    def get_curve_and_velocity(self):
        # Reconstruct the full path
        c = torch.cat([self.start_pt.view(1, 2), self.inner_points, self.end_pt.view(1, 2)], dim=0)
        
        # Calculate velocity using finite differences: v = dx/dt
        dt = 1.0 / self.num_segments
        c_dot = (c[1:] - c[:-1]) / dt
        
        # Evaluate the position at the midpoints of each segment
        c_mid = (c[1:] + c[:-1]) / 2.0
        
        return c_mid, c_dot, dt

# ==========================================
# Part 2: Third-Order Polynomial
# ==========================================
class PolynomialCurve(nn.Module):
    def __init__(self, start_pt, end_pt, num_points=50):
        super().__init__()
        self.start_pt = start_pt
        self.end_pt = end_pt
        
        # We use a dense grid of points to numerically integrate the polynomial
        self.t = torch.linspace(0, 1, num_points).view(-1, 1)
        self.dt = 1.0 / (num_points - 1)
        
        # Polynomial: c(t) = w0 + w1*t + w2*t^2 + w3*t^3
        # To enforce endpoints:
        # c(0) = start_pt  ==>  w0 = start_pt
        # c(1) = end_pt    ==>  w1 = end_pt - start_pt - w2 - w3
        self.w2 = nn.Parameter(torch.zeros(2))
        self.w3 = nn.Parameter(torch.zeros(2))

    def get_curve_and_velocity(self):
        w0 = self.start_pt
        w1 = self.end_pt - self.start_pt - self.w2 - self.w3
        
        # Calculate position c(t)
        c = w0 + w1*self.t + self.w2*(self.t**2) + self.w3*(self.t**3)
        
        # Calculate analytic velocity c_dot(t)
        c_dot = w1 + 2*self.w2*self.t + 3*self.w3*(self.t**2)
        
        return c, c_dot, self.dt

# ==========================================
# Optimization Loop
# ==========================================
def compute_energy(c, c_dot, dt):
    """
    Computes the Energy integral: Sum of (1 + ||c||^2) * ||c_dot||^2 * dt
    """
    norm_c_sq = torch.sum(c**2, dim=1)         # ||c(t)||^2
    norm_cdot_sq = torch.sum(c_dot**2, dim=1)  # ||c_dot(t)||^2
    
    # E = (1 + ||c||^2) * ||c_dot||^2
    energy = torch.sum((1.0 + norm_c_sq) * norm_cdot_sq) * dt
    return energy

def train_geodesic(model, epochs=1000, lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        c, c_dot, dt = model.get_curve_and_velocity()
        
        loss = compute_energy(c, c_dot, dt)
        loss.backward()
        optimizer.step()
        
    return model

# ==========================================
# Execution and Plotting
# ==========================================
if __name__ == "__main__":
    # Define our starting and ending points
    start_point = torch.tensor([-2.0, 2.0])
    end_point = torch.tensor([2.0, 2.0])

    # 1. Train Piecewise model
    piecewise_model = PiecewiseCurve(start_point, end_point)
    train_geodesic(piecewise_model, epochs=1500, lr=0.01)

    # 2. Train Polynomial model
    poly_model = PolynomialCurve(start_point, end_point)
    train_geodesic(poly_model, epochs=1500, lr=0.01)

    # Extract final optimized paths for plotting
    with torch.no_grad():
        c_piece, _, _ = piecewise_model.get_curve_and_velocity()
        c_poly, _, _ = poly_model.get_curve_and_velocity()
        
        # Add the endpoints back into the piecewise array for plotting
        c_piece = torch.cat([start_point.view(1,2), c_piece, end_point.view(1,2)])

    # Plot the results
    plt.figure(figsize=(8, 6))
    
    # Plot standard straight Euclidean line for reference
    plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 
             'k--', label='Straight Euclidean Line')
    
    # Plot our optimized geodesics
    plt.plot(c_piece[:, 0].numpy(), c_piece[:, 1].numpy(), 
             'r-o', markersize=3, label='Piecewise Geodesic')
    plt.plot(c_poly[:, 0].numpy(), c_poly[:, 1].numpy(), 
             'b-', linewidth=2, label='Polynomial Geodesic')
    
    plt.scatter([start_point[0], end_point[0]], [start_point[1], end_point[1]], 
                color='black', s=100, zorder=5)
    
    plt.title("Energy Minimization Geodesics under Quadratic Metric")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.xlim(-3, 3)
    plt.ylim(0, 3)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()