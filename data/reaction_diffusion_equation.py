import torch
import dgl

class ReactionDiffusionSystem(torch.nn.Module):
    """Computes the derivative of each node in a graph using reaction-diffusion dynamics.
    
    Supports types:
    - 'heat_sin': Nonlinear heat equation with sin() nonlinearity
    - 'heat_linear': Linear heat equation
    - 'allen_cahn': Allen-Cahn dynamics
    - 'consensus': Nonlinear consensus

    Args:
        graph (dgl.DGLGraph): The graph structure on which the equation is solved.
        device (torch.device): The device (CPU/GPU) where computations are performed.
        dynamics_type (str): Type of dynamics.
        lambda_value (float, optional): Scaling factor. Defaults to 1.0.
    """
    
    
    def __init__(self, graph: dgl.DGLGraph, device: torch.device, 
                 dynamics_type: str = 'heat_sin', lambda_value: float = 1.0):

        super().__init__()
        self.graph = graph
        self.device = device
        self.dynamics_type = dynamics_type
        self.lambda_value = lambda_value
        self._setup_dynamics()
    
    def _setup_dynamics(self):
        """Set up diffusion and reaction functions based on dynamics type."""
        
        if self.dynamics_type == 'heat_sin':
            self.diffusion_nonlinearity = torch.sin
            self.reaction_function = None
            
        elif self.dynamics_type == 'heat_linear':
            self.diffusion_nonlinearity = lambda x: x
            self.reaction_function = None
            
        elif self.dynamics_type == 'allen_cahn':
            # Epsilon is a tunable scaling factor
            epsilon = 0.5
            self.diffusion_nonlinearity = lambda x: -(epsilon**2) * x
            self.reaction_function = lambda u: u - u**3
            
        elif self.dynamics_type == 'consensus':
            self.diffusion_nonlinearity = lambda x: x / (1 + x**2)
            self.reaction_function = None

    def forward(self, t: float, u: torch.Tensor) -> torch.Tensor:
        """Compute the time derivative du/dt for each node.

        Args:
            t (float): Current time (unused in autonomous system).
            u (torch.Tensor): Current node values.

        Returns:
            torch.Tensor: Time derivatives du/dt for each node.
        """
        self.graph.ndata['u'] = u
        du_dt = torch.zeros_like(u)
        n = u.size(0)
        
        # Check if graph has edge weights
        has_weights = 'weight' in self.graph.edata
        
        # Diffusion term
        for i in range(n):
            neighbors = self.graph.successors(i)
            
            if len(neighbors) > 0:
                u_differences = u[neighbors] - u[i]
                nonlin_u_differences = self.diffusion_nonlinearity(u_differences)
                
                if has_weights:
                    # Get edge IDs for edges from i to its neighbors
                    edge_ids = self.graph.edge_ids(i, neighbors)
                    weights = self.graph.edata['weight'][edge_ids]
                    # Weigh each term
                    neighbor_sum = (weights * nonlin_u_differences).sum()
                else:
                    # Unweighted case
                    neighbor_sum = nonlin_u_differences.sum()
                
                du_dt[i] = (1/n) * neighbor_sum
        
        # Reaction term added to diffusion
        if self.reaction_function is not None:
            du_dt += self.reaction_function(u)
                
        return du_dt