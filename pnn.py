import torch
import torch.nn as nn
import torch.optim as optim

# Define the PINN model
class PINN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

# Physics-informed loss function (Advection-Diffusion PDE)
def physics_loss(model, x, t, wind_speed, diffusion_coeff):
    x.requires_grad = True
    t.requires_grad = True
    
    c = model(torch.cat((x, t), dim=1))  # Predicted concentration
    c_x = torch.autograd.grad(c, x, grad_outputs=torch.ones_like(c), create_graph=True)[0]
    c_t = torch.autograd.grad(c, t, grad_outputs=torch.ones_like(c), create_graph=True)[0]
    c_xx = torch.autograd.grad(c_x, x, grad_outputs=torch.ones_like(c_x), create_graph=True)[0]
    
    pde_residual = c_t + wind_speed * c_x - diffusion_coeff * c_xx
    return torch.mean(pde_residual**2)  # Minimize PDE residual

# Generate training data (synthetic)
torch.manual_seed(42)
n_samples = 1000
x = torch.rand(n_samples, 1) * 10  # Spatial coordinate

t = torch.rand(n_samples, 1) * 24  # Time (hours)
wind_speed = 0.5  # Advection coefficient
diffusion_coeff = 0.1  # Diffusion coefficient

# Define model and optimizer
pinn = PINN(input_dim=2, hidden_dim=20, output_dim=1)
optimizer = optim.Adam(pinn.parameters(), lr=0.001)

# Training loop
epochs = 5000
for epoch in range(epochs):
    optimizer.zero_grad()
    loss = physics_loss(pinn, x, t, wind_speed, diffusion_coeff)
    loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

print("Training complete!")
