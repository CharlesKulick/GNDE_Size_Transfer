import torch
from torch.utils.data import Dataset
import numpy as np
import torchdiffeq
import logging

def generate_random_fourier_polynomial(sample_points: np.array, degree: int, seed: int = None) -> np.array:
    """
    Generate a random Fourier polynomial evaluated at the given sample points.
    
    Args:
        sample_points (np.array): Points in [0, 1] where the Fourier polynomial is evaluated
        degree (int): Degree of the Fourier polynomial
        seed (int): Seed for reproducibility of random coefficients
    
    Returns:
        np.array: Array of size [num_points] containing the Fourier polynomial evaluated at each sample point.
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    # Sample coefficients a_k and b_k uniformly from [-1, 1]
    a_k = rng.uniform(-1, 1, degree)
    b_k = rng.uniform(-1, 1, degree)

    # Generate the Fourier polynomial values
    values = np.zeros(len(sample_points))
    for k in range(1, degree + 1):
        values += a_k[k - 1] * np.cos(2 * np.pi * k * sample_points) + b_k[k - 1] * np.sin(2 * np.pi * k * sample_points)
    
    return values

class PairOnlyDataset(Dataset):
    """Dataset for (initial, final) state pairs from multiple trajectories.
    
    This dataset is specifically designed for training Neural ODEs on full time interval
    predictions rather than step-by-step evolution. Each trajectory contributes exactly
    one training pair: (initial_condition, final_state).

    Args:
        initial_states (list[torch.Tensor]): List of initial condition tensors
        final_states (list[torch.Tensor]): List of corresponding final state tensors
    """
    def __init__(self, initial_states: list[torch.Tensor], final_states: list[torch.Tensor]):
        assert len(initial_states) == len(final_states), "Initial and final state lists must have same length"
        self.initial_states = initial_states
        self.final_states = final_states

    def __len__(self) -> int:
        """Return number of (initial, final) pairs."""
        return len(self.initial_states)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the (initial, final) pair for the given index.

        Args:
            idx (int): Index of the trajectory pair

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (initial_condition, final_state) pair
        """
        return self.initial_states[idx], self.final_states[idx]
    
    def get_device(self):
        """Get the device of the first tensor (assumes all on same device)"""
        return self.initial_states[0].device if self.initial_states else None

class ODEDataGeneratorPairs:
    """Generate (initial, final) training pairs for Neural ODE training.

    Args:
        n_nodes (int): Number of nodes in the graph
        device (torch.device): Device to use for computations
        end_time (float): End time for ODE integration
        num_timesteps (int): Number of timesteps for ODE integration (for accuracy)
        ode_func (callable): The ODE function to integrate
        num_train_trajectories (int): Number of training trajectories to generate
        fourier_degree (int): Degree of Fourier polynomials for initial conditions
    """
    def __init__(self, n_nodes: int, device: torch.device, end_time: float, num_timesteps: int, ode_func: callable, num_train_trajectories: int = 50, fourier_degree: int = 10):
        self.n_nodes = n_nodes
        self.device = device
        self.end_time = end_time
        self.num_timesteps = num_timesteps
        self.ode_func = ode_func
        self.num_train_trajectories = num_train_trajectories
        self.fourier_degree = fourier_degree
        self.positions = torch.linspace(0, 1, n_nodes).to(device)
        self.integration_time = torch.linspace(0, end_time, num_timesteps).to(device)

    def generate_random_initial_condition(self, seed: int = None) -> np.array:
        """Generate a random Fourier polynomial initial condition.

        Args:
            seed (int): Seed for reproducible random generation

        Returns:
            np.array: Initial condition values at node positions
        """
        positions_np = self.positions.cpu().numpy()
        return generate_random_fourier_polynomial(positions_np, self.fourier_degree, seed=seed)

    def _generate_trajectory_pair(self, initial_condition_values: np.array) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate a (initial, final) pair from the specified initial condition.

        Args:
            initial_condition_values (np.array): Initial condition values at each node

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (initial_state, final_state) pair
        """
        initial_u = torch.tensor(initial_condition_values).float().to(self.device)
        
        # Integrate the full trajectory
        solution = torchdiffeq.odeint(self.ode_func, initial_u, self.integration_time)
        
        # Return only initial and final states
        initial_state = solution[0]
        final_state = solution[-1]
        
        return initial_state, final_state

    def generate_all_datasets(self, base_seed: int = 42) -> dict[str, Dataset]:
        """Generate train, validation, and test datasets with (initial, final) pairs.

        Args:
            base_seed (int): Base seed for reproducible random generation

        Returns:
            dict[str, Dataset]: Dictionary containing 'train', 'val', and 'test' datasets
        """
        # Multiple (initial, final) pairs from different random initial conditions
        train_initial_states = []
        train_final_states = []
        
        logging.info(f"Generating {self.num_train_trajectories} training trajectory pairs.")
        for i in range(self.num_train_trajectories):
            ic_values = self.generate_random_initial_condition()
            initial_state, final_state = self._generate_trajectory_pair(ic_values)
            train_initial_states.append(initial_state)
            train_final_states.append(final_state)
            logging.info(f"  Training pair {i+1}/{self.num_train_trajectories} complete")
        
        # Validation: single (initial, final) pair
        logging.info("Generating validation trajectory pair.")
        val_seed = base_seed + 1000
        val_ic = self.generate_random_initial_condition(val_seed)
        val_initial, val_final = self._generate_trajectory_pair(val_ic)
        logging.info(f"  Validation pair complete")
        
        # Test: single (initial, final) pair
        logging.info("Generating test trajectory pair.")
        test_seed = base_seed + 2000
        test_ic = self.generate_random_initial_condition(test_seed)
        test_initial, test_final = self._generate_trajectory_pair(test_ic)
        logging.info(f"  Test pair complete")
        
        return {
            'train': PairOnlyDataset(train_initial_states, train_final_states),
            'val': PairOnlyDataset([val_initial], [val_final]),
            'test': PairOnlyDataset([test_initial], [test_final])
        }

    def generate_test_dataset(self, seed: int = 42) -> PairOnlyDataset:
        """Generate only test dataset, for convenient use in transfer settings.

        Args:
            seed (int): Seed for reproducible random generation

        Returns:
            PairOnlyDataset: Test dataset with single (initial, final) pair
        """
        test_ic = self.generate_random_initial_condition(seed)
        test_initial, test_final = self._generate_trajectory_pair(test_ic)
        return PairOnlyDataset([test_initial], [test_final])

    def generate_multiple_test_datasets(self, num_test_conditions: int) -> list[PairOnlyDataset]:
        """Generate multiple test datasets for evaluation averaging.

        Args:
            num_test_conditions (int): Number of test datasets to generate

        Returns:
            list[PairOnlyDataset]: List of test datasets, each with one (initial, final) pair
        """
        test_datasets = []
        logging.info(f"Generating {num_test_conditions} test trajectory pairs.")
        
        for i in range(num_test_conditions):
            test_ic = self.generate_random_initial_condition()
            test_initial, test_final = self._generate_trajectory_pair(test_ic)
            test_datasets.append(PairOnlyDataset([test_initial], [test_final]))
            logging.info(f"  Test pair {i+1}/{num_test_conditions} complete")
        
        return test_datasets