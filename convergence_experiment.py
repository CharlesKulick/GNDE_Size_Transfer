# Import external libraries
import argparse
import logging
import yaml
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Import torch libraries
import torch
import torch.nn as nn

# Import local project files
from models import model_defs
from utils import graphon_utils
from utils import data_utils

# Torch options for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Set available device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using: {device}")

# Instantiate argument parser
parser = argparse.ArgumentParser(description='Pass parameters for experiment runs.')
parser.add_argument('--graphon', type=str, required=True, default=None, help='Graphon to use for computations.')
parser.add_argument('--graphon_parameter', type=int, required=True, default=None, help='Graphon parameter to use in general graphon construction method parameter fields.')
parser.add_argument('--num_random_inits', type=int, required=False, default=10, help='Number of random weight initialization and feature pairs to test for each graph sequence.')
args = parser.parse_args()

# Access args
graphon_name = args.graphon
graphon_parameter = args.graphon_parameter
num_random_inits = args.num_random_inits

if args.graphon not in ["tent", "hsbm", "hexaflake"]:
    raise ValueError('The graphon type entered is not currently supported.')

# If weighted graph, we change our graph generation methods slightly to match our theoretical formulation
weighted_graph = args.graphon == "tent"

# Make folders for saving results
results_path = f'outputs/graphon_results'
if not os.path.exists(results_path):
    os.makedirs(results_path)
    logging.info(f"Created directory: {results_path}")

# Set seed, for full deterministic reproducibility
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# Utility to convert graph features to the needed form for our methods
def piecewise_constant(test_points: np.array, graph_features: np.array) -> np.array:
    """
    Function for converting graph features to a piecewise constant function on [0,1] of the desired resolution.
    
    Args:
        test_points (np.array): Points in [0, 1] to evaluate the piecewise constant function
        graph_features (np.array): numpy list of the graph features of interest
    
    Returns:
        np.array: Array of size test_points containing the graph_features function evaluated at each sample point.
    """
    n = len(graph_features)
    interval_length = 1.0 / n
    out = np.zeros_like(test_points)

    # Evaluate at each test point
    for index, u in enumerate(test_points):
        found = False
        for i in range(n):
            if u >= i * interval_length and u < (i + 1) * interval_length:
                out[index] = graph_features[i]
                found = True
                
        # For u == 1, return the last element of the feature
        if not found:
            out[index] = graph_features[-1]

    return out

# Method to generate all needed graphs
def make_list_of_all_graphs(graphon_parameter: int, config: dict) -> list:
    """
    Given a sequence of graphs, construct a random feature and a randomly initialized model, and compute the corresponding errors.
    
    Args:
        graphon_parameter (int): Parameter for graphon construction
        config (dict): Config file containing parameters
    
    Returns:
        list: List of all constructed graphs for the experiment
    """

    # Safely access training hyperparameters from the config
    try:
        graphon_resolution: int = config['general']['graphon_resolution']
        graph_sizes_min: int = config['general']['graph_sizes_min']
        graph_sizes_max: int = config['general']['graph_sizes_max']
        graph_sizes_total: int = config['general']['graph_sizes_total']
    except KeyError as e:
        raise KeyError(f"Missing required config parameter: {e}")

    # Set graph sizes and construct corresponding graphs
    graph_sizes = np.linspace(graph_sizes_min, graph_sizes_max, graph_sizes_total, endpoint=True)
    graph_sizes = np.append(graph_sizes, graphon_resolution)
    graphs = []

    # If weighted, use weighted methods
    if weighted_graph:
        for graph_size in graph_sizes:
            g = graphon_utils.create_weighted_graph(int(graph_size), graphon_parameter=0)
            g = g.to(device)
            graphs.append(g)
    else:
        W = graphon_method(graphon_resolution, graphon_parameter)
        for graph_size in graph_sizes:
            g = graphon_utils.create_specific_graph(W, int(graph_size))
            g = g.to(device)
            graphs.append(g)

    return graphs

# Main convergence calculation method
def get_convergence_errors(graphs: list, seed: int, config: dict) -> tuple[list[float], list[float]]:
    """
    Given a sequence of graphs, construct a random feature and a randomly initialized model, and compute the corresponding errors.
    
    Args:
        graphs (list): List of graphs to use for models
        seed (int): Seed for reproducibility
        config (dict): Config file containing parameters
    
    Returns:
        tuple[list[float], list[float]]: Two lists, one with absolute errors and one with relative errors
    """

    set_seed(seed)

    # Safely access training hyperparameters from the config
    try:
        model_dims: tuple[int, int, int] = config['model']['model_dims']
        input_dim, hidden_dim, output_dim = model_dims
        K: int = config['model']['K']
        method: str = config['model']['method']
        atol: float = config['model']['atol']
        rtol: float = config['model']['rtol']
        adjoint: bool = config['model']['adjoint']
        graphon_resolution: int = config['general']['graphon_resolution']
        graph_sizes_min: int = config['general']['graph_sizes_min']
        graph_sizes_max: int = config['general']['graph_sizes_max']
        graph_sizes_total: int = config['general']['graph_sizes_total']
        fourier_length: int = config['general']['fourier_length']
        integration_end_time: int = config['general']['integration_end_time']
        test_points_for_error: int = config['general']['test_points_for_error']
        time_points: int = config['general']['time_points']
    except KeyError as e:
        raise KeyError(f"Missing required config parameter: {e}")

    # Set graph sizes and construct corresponding graphs
    graph_sizes = np.linspace(graph_sizes_min, graph_sizes_max, graph_sizes_total, endpoint=True)
    graph_sizes = np.append(graph_sizes, graphon_resolution)

    # Generate features for all graphs
    for i, graph in enumerate(graphs):

        N_local = graph.num_nodes()
        sample_points = np.linspace(0, 1, N_local)

        # Generate feature tensor
        feature_vector = torch.tensor(data_utils.generate_random_fourier_polynomial(sample_points, fourier_length, seed=seed), dtype=torch.float32).unsqueeze(1).to(device)
        graph.ndata['feat'] = feature_vector

    # Construct a list of models with identical weights, on each graph
    models = []
    for i, graph in enumerate(graphs):

        # Initialize the vector field
        gnn_vector_field = model_defs.TwoLayerGCN(graph, input_dim, hidden_dim, output_dim, activation=nn.Softplus(), K=K, dropout=0).to(device)

        # Initialize the corresponding GNDE model
        model = model_defs.ODEBlock(odefunc=gnn_vector_field, method=method, atol=atol, rtol=rtol, adjoint=adjoint).to(device)
            
        # If not the first model, copy weights
        if i > 0:

            # Extract parameters from the original model
            original_parameters = {name: param.clone() for name, param in models[0].odefunc.named_parameters()}

            # Apply the extracted parameters to the new model
            for name, param in original_parameters.items():
                if name in gnn_vector_field.state_dict():
                    gnn_vector_field.state_dict()[name].copy_(param.data)

            # Verify the parameters are copied correctly
            for (name1, param1), (_, param2) in zip(models[0].odefunc.named_parameters(), gnn_vector_field.named_parameters()):
                assert torch.equal(param1, param2), f"Mismatch in parameter {name1}"
        models.append(model)

    # Lists to store calculated maximum errors across time points
    max_absolute_errors = [0.0] * len(graph_sizes)
    max_relative_errors = [0.0] * len(graph_sizes)
    highest_resolution_outputs = None

    # Iterate, starting from the largest (reference) graph
    for i in range(len(graph_sizes)-1, -1, -1):

        # Evaluate with current model on current graph
        graph = graphs[i]
        model = models[i]
        input_feature = graph.ndata['feat'].to(device)
        
        # Get full trajectory across time points
        trajectory = model(input_feature, integration_end_time, time_points=time_points, return_trajectory=True)
        trajectory = trajectory.cpu().detach().numpy()
        torch.cuda.empty_cache()
        
        # If this is the highest resolution model, save its outputs as reference
        if i == len(graph_sizes) - 1:
            highest_resolution_outputs = []
            
            # For each time point, compute the reference function values
            for t in range(time_points):
                time_output = trajectory[t].squeeze(-1)
                test_values = np.linspace(0, 1, test_points_for_error)
                f_values = piecewise_constant(test_values, time_output)
                highest_resolution_outputs.append(f_values)
                
            # No errors to compute for highest resolution
            max_absolute_errors[i] = 0.0
            max_relative_errors[i] = 0.0
            
        else:
            current_max_abs_error = 0.0
            current_max_rel_error = 0.0
            
            # For each time point, compute the error against highest resolution
            for t in range(time_points):
                time_output = trajectory[t].squeeze(-1)
                
                # Generate test points for error calculation
                test_values = np.linspace(0, 1, test_points_for_error)
                f_values = piecewise_constant(test_values, time_output)
                
                # Calculate errors at this time point
                abs_error = np.linalg.norm(highest_resolution_outputs[t] - f_values)
                rel_error = abs_error / np.linalg.norm(highest_resolution_outputs[t])
                current_max_abs_error = max(current_max_abs_error, abs_error)
                current_max_rel_error = max(current_max_rel_error, rel_error)
            
            # Store maximum errors for this graph size
            max_absolute_errors[i] = current_max_abs_error
            max_relative_errors[i] = current_max_rel_error
            
            logging.info(f"Graph size {graph.num_nodes()} maximum relative error: {current_max_rel_error}")

    return max_absolute_errors, max_relative_errors

# Main code
graphon_method = None
if graphon_name == "hsbm":
    graphon_method = graphon_utils.generate_deterministic_hsbm_graphon
elif graphon_name == "hexaflake":
    graphon_method = graphon_utils.generate_hexaflake_graphon

with open(f'configs/convergence_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

try:
    graph_sizes_min: int = config['general']['graph_sizes_min']
    graph_sizes_max: int = config['general']['graph_sizes_max']
    graph_sizes_total: int = config['general']['graph_sizes_total']
    graphon_resolution: int = config['general']['graphon_resolution']
except KeyError as e:
    raise KeyError(f"Missing required config parameter: {e}")

# Run the computation to determine the errors and get the error plot.
logging.info(f"Starting error computations, Graphon {graphon_name}, Parameter {graphon_parameter}:")

level_errors = []
level_relative_errors = []

list_of_all_graphs = make_list_of_all_graphs(graphon_parameter, config)

for local_seed in range(num_random_inits):
    errors, relative_errors = get_convergence_errors(list_of_all_graphs, local_seed, config)
    level_errors.append(errors)
    level_relative_errors.append(relative_errors)

# Convert lists to numpy arrays and store in dictionary
error_data = np.array(level_errors)
relative_error_data = np.array(level_relative_errors)

# Save results in numpy files for further analysis
np.save(f'{results_path}/graphon_{graphon_name}_parameter_{graphon_parameter}_inits_{num_random_inits}_experiment_abs_error_data', error_data)
np.save(f'{results_path}/graphon_{graphon_name}_parameter_{graphon_parameter}_inits_{num_random_inits}_experiment_rel_error_data', relative_error_data)

logging.info("Computation completed. Making log-log convergence plot.")

# Define graph sizes for the plotting
graph_sizes = np.linspace(graph_sizes_min, graph_sizes_max, graph_sizes_total, endpoint=True)
graph_sizes = np.append(graph_sizes, graphon_resolution)

# Plot
plt.figure(figsize=(10, 6))

# Compute mean and std across seeds
mean_errors = np.mean(relative_error_data, axis=0)
std_errors = np.std(relative_error_data, axis=0)

# Exclude the first error (because it is 0)
plotting_errors = mean_errors[:-1]
plotting_std = std_errors[:-1]
plotting_graph_sizes = graph_sizes[:-1]

# Take logs
log_N = np.log2(plotting_graph_sizes)
log_errors = np.log2(plotting_errors)

# Linear regression for slope (legend)
slope, _, _, _, _ = linregress(log_N, log_errors)

plt.errorbar(
    plotting_graph_sizes, 
    plotting_errors, 
    yerr=plotting_std, 
    fmt='s',
    linestyle='-', 
    label=f'Graphon: {graphon_name}, Parameter: {graphon_parameter}, Slope: {slope:.2f}',
    capsize=5
)

plt.xscale('log', base=2)
plt.yscale('log', base=2)
plt.xlabel('Number of Nodes ($\log_2$ scale)', fontsize=18)
plt.ylabel('Relative Error ($\log_2$ scale)', fontsize=18)
plt.tick_params(axis='both', labelsize=18)
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
plt.legend(fontsize=18, loc='best')
plt.tight_layout()
plt.savefig(f'{graphon_name}_{graphon_parameter}_basic_convergence_plot.png', bbox_inches='tight', dpi=600)

logging.info("Basic visualization saved. Computation complete.")