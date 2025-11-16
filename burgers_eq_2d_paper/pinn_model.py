import torch
import torch.nn as nn
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


class ResidualBlock(nn.Module):

    # Define a block with two layers and a residual connection
    def __init__(self, _size: int, _activation=nn.Tanh()):
        super().__init__()
        self.Linear = nn.Linear(_size, _size)
        self.activation = _activation

    def forward(self, X):
        return X + self.activation(self.Linear(self.activation(X)))


class MLP(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=5):
        super().__init__()
        # modules = []
        # modules.append(nn.Linear(3, hidden_dim))
        # for _ in range(num_layers - 2):
        #     modules.append(ResidualBlock(hidden_dim))
        # modules.append(nn.Linear(hidden_dim, 2))
        # self.model = nn.Sequential(*modules)
        # Add number of MLP input and outputs to the layers list
        layers = [3,128,128,128,128,128,2]
        
        # Built the MLP
        modules = []
        for _in, _out in list(zip(layers, layers[1:])):
            modules.append(nn.Linear(_in, _out))
            modules.append(ResidualBlock(_out))
        
        # Remove last block
        modules.pop()

        self.model = nn.Sequential(*modules)
        
    def forward(self, X):
        Y_n = self.model(X)
        Y_p = self.boundary_solution(X)
        D = self.boundary_distance(X)

        return D * Y_n + (1 - D) * Y_p

    def boundary_solution(self, X):
        x = X[:, 1].reshape(-1, 1)
        y = X[:, 2].reshape(-1, 1)

        u = torch.sin(2 * np.pi * x) * torch.sin(2 * np.pi * y)
        v = torch.sin(np.pi * x) * torch.sin(np.pi * y)

        return torch.hstack((u, v))

    def boundary_distance(self, X):
        alpha = 26.4  # Reaches 0.99 at t = 0.1
        # alpha = 10.56 # Reaches 0.99 at t = 0.25

        t = X[:, 0].reshape(-1, 1)
        x = X[:, 1].reshape(-1, 1)
        y = X[:, 2].reshape(-1, 1)

        dt = torch.tanh(t * alpha)
        dx = 4 * x * (1 - x)
        dy = 4 * y * (1 - y)

        return torch.hstack((dt * dx * dy, dt * dx * dy))

class PINN:
    def __init__(self):
        self.network = MLP()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)
        self.loss_f = nn.MSELoss().to(self.device)
        self.loss_s = nn.L1Loss().to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)

        # parameters
        self.nf = 1000
        self.ns = 100
        self.nu = 0.01 / np.pi
        self.x_low = 0.0
        self.x_high = 1.0
        self.y_low = 0.0
        self.y_high = 1.0
        self.t_low = 0.0
        self.t_high = 1.0

        # sample points
        self.X_f = self.residual_sample_points()

        # solution points
        [X_s, Y_s] = self.solution_sample_points("reference_solution.mat")
        self.X_s = torch.tensor(X_s, dtype=torch.float, requires_grad=True).to(
            self.device
        )
        self.Y_s = torch.tensor(Y_s, dtype=torch.float, requires_grad=False).to(
            self.device
        )

    def pde_residual(self, X: torch.Tensor):
        # Forward pass
        t = X[:, 0].reshape(-1, 1)
        x = X[:, 1].reshape(-1, 1)
        y = X[:, 2].reshape(-1, 1)
        Y = self.network(torch.hstack((t, x, y)))

        u = Y[:, 0].reshape(-1, 1)
        v = Y[:, 1].reshape(-1, 1)

        # Get derivatives
        u_t = torch.autograd.grad(
            u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
        )[0]
        v_t = torch.autograd.grad(
            v, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
        )[0]
        u_x = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
        )[0]
        u_y = torch.autograd.grad(
            u, y, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
        )[0]
        v_x = torch.autograd.grad(
            v, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
        )[0]
        v_y = torch.autograd.grad(
            v, y, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x,
            x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True,
        )[0]
        u_yy = torch.autograd.grad(
            u_y,
            y,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True,
        )[0]
        v_xx = torch.autograd.grad(
            v_x,
            x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True,
        )[0]
        v_yy = torch.autograd.grad(
            v_y,
            y,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True,
        )[0]

        # Compute residuals
        R1 = u_t + u * u_x + v * u_y - self.nu * (u_xx + u_yy)
        R2 = v_t + u * v_x + v * v_y - self.nu * (v_xx + v_yy)

        return self.loss_f(R1, torch.zeros_like(R1)) + self.loss_f(
            R2, torch.zeros_like(R2)
        )

    def residual_sample_points(self):
        t = np.random.uniform(low=self.t_low, high=self.t_high, size=(self.nf, 1))
        x = np.random.uniform(low=self.x_low, high=self.x_high, size=(self.nf, 1))
        y = np.random.uniform(low=self.y_low, high=self.y_high, size=(self.nf, 1))
        X_f = np.hstack((t, x, y))
        X_f = torch.tensor(
            X_f, dtype=torch.float32, requires_grad=True, device=self.device
        )
        return X_f

    def solution_sample_points(self, file_path: str):
        data = loadmat(file_path)

        [x, t, y] = np.meshgrid(data["x"], data["t"], data["y"])

        t = t.flatten().reshape(-1, 1)
        x = x.flatten().reshape(-1, 1)
        y = y.flatten().reshape(-1, 1)
        u = data["uref"].transpose((2, 1, 0)).flatten().reshape(-1, 1)
        v = data["vref"].transpose((2, 1, 0)).flatten().reshape(-1, 1)

        ind = np.random.choice(t.shape[0], size=self.ns)

        X = np.hstack((t[ind], x[ind], y[ind]))
        Y = np.hstack((u[ind], v[ind]))

        return X, Y

    def total_loss(self):
        loss_residual = self.pde_residual(self.X_f)
        loss_solution_points = self.loss_s(self.network(self.X_s), self.Y_s)
        return loss_residual + loss_solution_points

    def train(self, epochs=100):
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            loss = self.total_loss()
            loss.backward()
            self.optimizer.step()
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")

    def predict(self, X: torch.Tensor):
        self.network.eval()
        with torch.no_grad():
            return self.network(X)

    def save_model(self, path="./saved_models/pinn_model.pth"):
        torch.save(self.network.state_dict(), path)

    def load_model(self, path="./saved_models/pinn_model.pth"):
        self.network.load_state_dict(torch.load(path))

    def plot_prediction(self, save_path="./saved_plots/prediction.png"):
        # Prediction grid
        N_x = 256
        x = np.linspace(self.x_low, self.x_high, N_x)
        y = np.linspace(self.y_low, self.y_high, N_x)

        # Create meshgrid
        xx, yy = np.meshgrid(x, y)
        xy_flat = np.stack([xx.flatten(), yy.flatten()], axis=1)

        # Select time slices to plot
        time_slices = [0.0, 0.25, 0.5, 0.75, 1.0]

        # Adjusted layout for side-by-side plots with a full-height colorbar
        fig, ax = plt.subplots(2, len(time_slices), figsize=(15, 6))

        for j, ts in enumerate(time_slices):
            Y_pred = self.predict(torch.tensor(np.hstack((xy_flat, np.full((xy_flat.shape[0], 1), ts))), dtype=torch.float32).to(self.device)).cpu().numpy()
            u = Y_pred[:, 0].reshape(N_x, N_x)
            v = Y_pred[:, 1].reshape(N_x, N_x)

            # Plot U
            im_u = ax[0, j].contourf(xx, yy, u, levels=50, cmap="viridis")
            ax[0, j].set_title(f"U at t = {ts}")
            ax[0, j].set_xlabel("x")
            ax[0, j].set_ylabel("y")

            # Plot V
            im_v = ax[1, j].contourf(xx, yy, v, levels=50, cmap="viridis")
            ax[1, j].set_title(f"V at t = {ts}")
            ax[1, j].set_xlabel("x")
            ax[1, j].set_ylabel("y")

        # Add a single colorbar on the right-hand side, spanning the full height
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(im_u, cax=cbar_ax)
        cbar.set_label("Colorbar Label")

        plt.suptitle("Predicted U(x,y,t) and V(x,y,t) at Different Time Slices")
        plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to make space for colorbar
        plt.savefig(save_path)
        plt.show()
        plt.close(fig)
