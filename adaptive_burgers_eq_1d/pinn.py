import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import matplotlib.gridspec as gridspec


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.Tanh()  # naturally scales network output to [-1, 1]
        layers = [2, 50, 50, 50, 50, 1]
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
        self.layers = nn.ModuleList(modules)

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation(layer(x))
        return self.layers[-1](x)


class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        # set up neural network parameters
        self.network = MLP()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)
        self.num_epochs = 5000
        self.train_time = 0

        # save old model for adaptive collocation point control
        self.old_network = MLP()
        self.old_network.load_state_dict(self.network.state_dict())
        self.old_network.to(self.device)

        # adaptive collocation point control parameters
        self.is_adaptive = True
        self.adaptive_interval = 100
        self.add_point_p = 0.3
        self.remove_point_p = 0.1
        self.point_std = 0.1
        self.error_tol = 1e-2  # make lower
        self.adaptive_error_tol = True
        
        self.N_f_0 = 1000  # initial number of collocation points

        # set up experiment parameters
        torch.set_default_dtype(torch.float32)
        # spatial and temporal domain boundaries
        self.x_min, self.x_max = -1.0, 1.0
        self.t_min, self.t_max = 0.0, 1.0

        # viscosity coefficient
        self.nu = 0.01 / np.pi

        # collocation points
        self.N_f = self.N_f_0
        self.N_0 = 200
        self.N_b = 200

        # Initial and boundary conditions
        X_f = np.random.rand(self.N_f, 2)
        X_f[:, 0] = X_f[:, 0] * (self.x_max - self.x_min) + self.x_min  # x in [-1, 1]
        X_f[:, 1] = X_f[:, 1] * (self.t_max - self.t_min) + self.t_min  # t in [0, 1]

        # Initial condition on velocity: u(x, 0) = -sin(pi * x)
        x0 = np.linspace(self.x_min, self.x_max, self.N_0)[:, None]
        t0 = np.zeros_like(x0)
        u0 = -np.sin(np.pi * x0)

        # Boundary conditions: u(-1, t) = 0, u(1, t) = 0
        tb = np.linspace(self.t_min, self.t_max, self.N_b)[:, None]
        xb_left = np.ones_like(tb) * self.x_min
        xb_right = np.ones_like(tb) * self.x_max
        ub_left = np.zeros_like(tb)
        ub_right = np.zeros_like(tb)

        # Convert to PyTorch tensors
        self.X_f = torch.tensor(
            X_f, dtype=torch.float32, requires_grad=True
        )  # enable gradients for collocation points
        self.x0 = torch.tensor(x0, dtype=torch.float32)
        self.t0 = torch.tensor(t0, dtype=torch.float32)
        self.u0 = torch.tensor(u0, dtype=torch.float32)
        self.tb = torch.tensor(tb, dtype=torch.float32)
        self.xb_left = torch.tensor(xb_left, dtype=torch.float32)
        self.xb_right = torch.tensor(xb_right, dtype=torch.float32)
        self.ub_left = torch.tensor(ub_left, dtype=torch.float32)
        self.ub_right = torch.tensor(ub_right, dtype=torch.float32)

        # load ground truth data
        data = scipy.io.loadmat("./data/burgers_shock.mat")

        self.t = data["t"].flatten()[:, None]
        self.x = data["x"].flatten()[:, None]
        self.Exact = np.real(data["usol"]).T

        X, T = np.meshgrid(self.x, self.t)

        X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
        u_star = self.Exact.flatten()[:, None]

        # convert to pytorch tensors
        self.X_star = torch.tensor(X_star, dtype=torch.float32).to(self.device)
        self.u_star = torch.tensor(u_star, dtype=torch.float32).to(self.device)

    def pde_residual(self, X):
        x = X[:, 0:1]
        t = X[:, 1:2]
        u = self.network(torch.cat([x, t], dim=1))  # network output u(x,t)

        u_x = autograd.grad(
            u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True
        )[0]
        u_t = autograd.grad(
            u, t, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True
        )[0]
        u_xx = autograd.grad(
            u_x,
            x,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
            retain_graph=True,
        )[0]

        f = u_t + u * u_x - self.nu * u_xx  # Burgers' equation residual
        return f

    def loss_func(self):

        # PDE residual loss
        f_pred = self.pde_residual(self.X_f)
        loss_f = torch.mean(f_pred**2)

        # Initial condition loss
        u0_pred = self.network(torch.cat([self.x0, self.t0], dim=1))
        loss_0 = torch.mean((u0_pred - self.u0) ** 2)

        # Boundary condition loss
        u_left_pred = self.network(torch.cat([self.xb_left, self.tb], dim=1))
        u_right_pred = self.network(torch.cat([self.xb_right, self.tb], dim=1))
        loss_b = torch.mean(u_left_pred**2) + torch.mean(u_right_pred**2)

        loss = loss_f + loss_0 + loss_b
        return loss

    def update_collocation_points_old_model(self):
        """
        Adaptively update the collocation points based on the current model and old model.
        """
        add_point_p = 0.3
        remove_point_p = 0.3
        point_std = 0.01

        with torch.no_grad():
            u_pred_new = self.network(self.X_f)
            u_pred_old = self.old_network(self.X_f)
            error = torch.abs(u_pred_new - u_pred_old).detach().cpu().numpy().flatten()

        error_tol = 1e-2
        high_error_indices = np.where(error > error_tol)[0]
        low_error_indices = np.where(error <= error_tol)[0]
        print(f"high_error_indices: {high_error_indices.shape[0]}/{self.N_f}")
        print(f"low_error_indices: {low_error_indices.shape[0]}/{self.N_f}")

        # add new points
        new_points = []
        for i in high_error_indices:
            if np.random.rand() < add_point_p:  # do not always add a point
                base_point = self.X_f[i].detach().cpu().numpy()
                new_noisy_point = base_point + np.random.normal(
                    0, point_std, size=base_point.shape
                )  # fixe std deviation
                if (
                    new_noisy_point[0] > self.x_min
                    and new_noisy_point[0] < self.x_max
                    and new_noisy_point[1] > self.t_min
                    and new_noisy_point[1] < self.t_max
                ):
                    new_points.append(new_noisy_point)

        # remove points
        if low_error_indices.size > 0:
            to_drop_mask = (
                np.random.rand(low_error_indices.size) < remove_point_p
            )  # drop them with fixed probability
            drop_indices = low_error_indices[to_drop_mask]

            if drop_indices.size > 0:
                # build a boolean mask over all points: True => keep
                keep_mask = np.ones(self.N_f, dtype=bool)
                keep_mask[drop_indices] = False

                # apply mask to self.X_f
                X_f_np = self.X_f.detach().cpu().numpy()
                X_f_np = X_f_np[keep_mask]

                # rebuild self.X_f as a tensor with requires_grad=True
                self.X_f = torch.tensor(
                    X_f_np, dtype=torch.float32, requires_grad=True
                ).to(self.device)

                # update N_f
                self.N_f = self.X_f.shape[0]

        self.N_f += len(new_points)
        if len(new_points) > 0:
            new_points = np.array(new_points)
            new_points_tensor = torch.tensor(
                new_points, dtype=torch.float32, requires_grad=True
            ).to(self.device)
            self.X_f = torch.cat([self.X_f, new_points_tensor], dim=0)

        print(f"new N_f: {self.N_f}")

        # save current model as old model for next iteration
        self.old_network.load_state_dict(self.network.state_dict())

    def update_collocation_points_residual(self):
        """
        Adaptively update the collocation points based on the PDE residual.
        """
        f_pred = self.pde_residual(self.X_f)
        error = torch.abs(f_pred).detach().cpu().numpy().flatten()

        if self.adaptive_error_tol:
            self.error_tol = np.median(error)

        # print(f"Max residual error: {np.max(error):.6e}")
        # print(f"Min residual error: {np.min(error):.6e}")
        # print(f"Mean residual error: {np.mean(error):.6e}")
        # print(f"Median residual error: {np.median(error):.6e}")

        high_error_indices = np.where(error > self.error_tol)[0]
        low_error_indices = np.where(error <= self.error_tol)[0]
        # print(f"high_error_indices: {high_error_indices.shape[0]}/{self.N_f}")
        # print(f"low_error_indices: {low_error_indices.shape[0]}/{self.N_f}")

        # add new points
        new_points = []
        for i in high_error_indices:
            if np.random.rand() < self.add_point_p:  # do not always add a point
                base_point = self.X_f[i].detach().cpu().numpy()
                new_noisy_point = base_point + np.random.multivariate_normal(
                    np.zeros(base_point.shape[0]),
                    self.point_std * np.eye(base_point.shape[0]),
                )  # fixe std deviation
                if (
                    new_noisy_point[0] > self.x_min
                    and new_noisy_point[0] < self.x_max
                    and new_noisy_point[1] > self.t_min
                    and new_noisy_point[1] < self.t_max
                ):
                    new_points.append(new_noisy_point)

        # remove points
        if low_error_indices.size > 0:
            to_drop_mask = (
                np.random.rand(low_error_indices.size) < self.remove_point_p
            )  # drop them with fixed probability
            drop_indices = low_error_indices[to_drop_mask]

            if drop_indices.size > 0:
                # build a boolean mask over all points: True => keep
                keep_mask = np.ones(self.N_f, dtype=bool)
                keep_mask[drop_indices] = False

                # apply mask to self.X_f
                X_f_np = self.X_f.detach().cpu().numpy()
                X_f_np = X_f_np[keep_mask]

                # rebuild self.X_f as a tensor with requires_grad=True
                self.X_f = torch.tensor(
                    X_f_np, dtype=torch.float32, requires_grad=True
                ).to(self.device)

                # update N_f
                self.N_f = self.X_f.shape[0]

        self.N_f += len(new_points)
        if len(new_points) > 0:
            new_points = np.array(new_points)
            new_points_tensor = torch.tensor(
                new_points, dtype=torch.float32, requires_grad=True
            ).to(self.device)
            self.X_f = torch.cat([self.X_f, new_points_tensor], dim=0)

        print(f"new N_f: {self.N_f}")

    def train(self):
        print("Starting training...")
        start_time = time.perf_counter()

        for epoch in range(self.num_epochs):
            self.optimizer.zero_grad()
            loss = self.loss_func()
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {loss.item():.5e}")

            if self.is_adaptive and (epoch + 1) % self.adaptive_interval == 0:
                self.update_collocation_points_residual()

        total_time = time.perf_counter() - start_time
        self.train_time = f"{total_time:.2f} seconds"
        print(f"Training complete! Total time: {total_time:.2f} seconds")

    def predict(self, X: torch.Tensor):
        self.network.eval()
        with torch.no_grad():
            u_pred = self.network(X)
        return u_pred

    def save_model(self, root="./saved_models", name="pinn_model.pth"):
        os.makedirs(root, exist_ok=True)
        path = os.path.join(root, name)
        checkpoint = {
            "model_state": self.network.state_dict(),
            "N_f": self.N_f,
            "X_f": self.X_f.detach().cpu().numpy(),
        }
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")

    def load_model(self, path="./saved_models/pinn_model.pth"):
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint["model_state"])
        self.N_f = checkpoint["N_f"]
        self.X_f = torch.tensor(
            checkpoint["X_f"], dtype=torch.float32, requires_grad=True
        ).to(self.device)
        print(f"Model loaded from {path}")

    def compute_l2_error(self):
        u_pred = self.predict(self.X_star)
        error_u = np.linalg.norm(self.u_star - u_pred, 2) / np.linalg.norm(
            self.u_star, 2
        )
        return error_u

    def create_experiment_folder(self):
        base_folder = "./saved_plots"
        experiment_name = "experiment"

        # Start with the base experiment folder
        experiment_folder = os.path.join(base_folder, f"{experiment_name}_1")
        counter = 1

        # Increment the folder name until a non-existing folder is found
        while os.path.exists(experiment_folder):
            counter += 1
            experiment_folder = os.path.join(
                base_folder, f"{experiment_name}_{counter}"
            )

        # Create the new folder
        os.makedirs(experiment_folder)
        return experiment_folder

    def save_parameters(self, path):
        """
        Save the parameters of the PINN model to a text file with aligned formatting.

        Args:
            path (str): The directory where the parameters file will be saved.
        """
        # Ensure the directory exists
        os.makedirs(path, exist_ok=True)

        # Define the file path
        parameter_file = os.path.join(path, "parameters.txt")

        # Prepare parameters with aligned formatting
        parameters = {
            "N_f": self.N_f,
            "N_0": self.N_0,
            "N_b": self.N_b,
            "is_adaptive": self.is_adaptive,
            "num_epochs": self.num_epochs,
            "adaptive_interval": self.adaptive_interval,
            "adaptive_error_tol": self.adaptive_error_tol,
            "error_tol": self.error_tol,
            "add_point_p": self.add_point_p,
            "remove_point_p": self.remove_point_p,
            "point_std": self.point_std,
            "N_f_0": self.N_f_0,
            "train_time": self.train_time,
            "l2_error" : self.compute_l2_error(),
        }
        if hasattr(self, "add_point_p"):
            parameters["add_point_p"] = self.add_point_p
        if hasattr(self, "remove_point_p"):
            parameters["remove_point_p"] = self.remove_point_p

        # Calculate the maximum length of parameter names for alignment
        max_key_length = max(len(key) for key in parameters.keys())

        # Write parameters to the file
        with open(parameter_file, "w") as f:
            f.write("PINN Parameters:\n")
            for key, value in parameters.items():
                f.write(f"{key:<{max_key_length}} : {value}\n")

        print(f"Parameters saved to {parameter_file}")

    def plot_solution(
        self, path, overlay_collocation_points=True, name="prediction.png"
    ):
        N_x, N_t = 256, 100
        x = np.linspace(self.x_min, self.x_max, N_x)
        t = np.linspace(self.t_min, self.t_max, N_t)
        X, T = np.meshgrid(x, t)
        XT = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
        XT_tensor = torch.tensor(XT, dtype=torch.float32).to(self.device)

        u_pred = self.predict(XT_tensor).cpu().numpy().reshape(N_t, N_x)

        time_slices = [0.0, 0.25, 0.5, 0.75]

        # Create figure and GridSpec
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(
            3,  # 3 rows
            len(time_slices),  # same number of columns as time_slices
            height_ratios=[2, 0.1, 1],  # top contour, small spacer, bottom plots
            hspace=0.4,  # vertical spacing
            wspace=0.3,  # horizontal spacing
        )

        # Top: contour spanning all columns
        ax_contour = fig.add_subplot(gs[0, :])
        contour = ax_contour.contourf(T.T, X.T, u_pred.T, levels=100, cmap="coolwarm")
        fig.colorbar(contour, ax=ax_contour, label="u(x,t)")
        ax_contour.set_xlabel("t")
        ax_contour.set_ylabel("x")
        ax_contour.set_title("Predicted solution u(x,t) via PINN")

        # overlay collocation points
        if overlay_collocation_points:
            Xf_np = self.X_f.detach().cpu().numpy()  # shape (N_f, 2): [x, t]
            x_f = Xf_np[:, 0]
            t_f = Xf_np[:, 1]
            ax_contour.scatter(
                t_f,
                x_f,  # note: t on x-axis, x on y-axis
                marker="x",
                s=10,
                c="k",
                linewidths=0.7,
                alpha=0.7,
                label="Collocation points",
            )
            ax_contour.legend(loc="upper right")

        # Bottom: one axis per time slice, sharing the same columns
        for i, t_slice in enumerate(time_slices):
            ax_1d = fig.add_subplot(gs[2, i])

            t_idx = int(t_slice * (N_t - 1))
            u_pred_slice = u_pred[t_idx, :]

            ax_1d.plot(self.x, self.Exact[t_idx, :], "r--", label="Exact", linewidth=2)
            ax_1d.plot(x, u_pred_slice, "b-", label="Predicted", linewidth=2)
            ax_1d.set_xlabel("x")
            ax_1d.set_ylabel("u")
            ax_1d.set_title(f"t = {t_slice}")
            ax_1d.grid(True, alpha=0.3)
            ax_1d.legend()

        fig.savefig(os.path.join(path, name), dpi=300, bbox_inches="tight")
        print(f"Plot saved to {os.path.join(path, name)}")
        plt.show()
        plt.close(fig)

    def save_plot_parameters(self):
        path = self.create_experiment_folder()
        self.plot_solution(path)
        self.save_parameters(path)
