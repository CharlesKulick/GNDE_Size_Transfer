# Import external libraries
import argparse
import logging
import yaml
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import linregress

# Import torch libraries
import torch
import torch.nn as nn

# Import local project files
import models.model_defs as model_defs
from utils import data_utils
from utils import graphon_utils
from data import reaction_diffusion_equation

# Torch options for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Set available device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using: {device}")

# Instantiate argument parser 
parser = argparse.ArgumentParser(description='Evaluate transfer performance of pair-trained models.')
parser.add_argument('--dynamics', type=str, required=True, help='Type of dynamics to use for data generation (heat_sin, heat_linear, allen_cahn, consensus)')
parser.add_argument('--nvals', nargs='+', type=int, help='List of N values used for original training')
parser.add_argument('--seeds', nargs='+', type=int, help='List of seeds')
parser.add_argument('--N', type=int, required=True, help='Size of large graph N for transfer testing')
parser.add_argument('--use_weighted', type=bool, required=False, default="", help='Use weighted graphs instead of unweighted graphs, empty string for unweighted, nonempty for weighted.')
parser.add_argument('--graphon', type=str, required=True, help='Graphon type for unweighted (hsbm, hexaflake, checkerboard, sierpinski, koch, knn)')
parser.add_argument('--graphon_parameter', type=int, required=True, help='Graphon parameter: for unweighted (general parameter), for weighted (function index 0-5)')
parser.add_argument('--num_train_trajectories', type=int, default=50, help='Number of training pairs (must match training)')
parser.add_argument('--fourier_degree', type=int, default=10, help='Fourier degree (must match training)')
parser.add_argument('--num_test_conditions', type=int, default=10, help='Number of different initial conditions to test on')
parser.add_argument('--dropout', type=float, required=False, default=0.0, help='Dropout value used during training (must match training)')

args = parser.parse_args()

# Get parser values for later reference
large_graph_n = int(args.N)
n_values = args.nvals
seeds = args.seeds
dynamics_type = args.dynamics
use_weighted = args.use_weighted
graphon_name = args.graphon
graphon_parameter = args.graphon_parameter
num_train_trajectories = args.num_train_trajectories
fourier_degree = args.fourier_degree
num_test_conditions = args.num_test_conditions
dropout_val = args.dropout

if use_weighted:
    if graphon_parameter not in [0, 1, 2, 3, 4, 5]:
        raise ValueError('For weighted mode, graphon_parameter must be an integer from 0 to 5.')
else:
    if graphon_name not in ["hsbm", "hexaflake", "checkerboard", "sierpinski", "koch", "knn"]:
        raise ValueError('For unweighted mode, graphon must be one of: hsbm, hexaflake, checkerboard, sierpinski, koch, knn.')

if use_weighted:
    graphon_display_name = graphon_utils.get_weighted_graphon_name(graphon_parameter)
    mode_str = "Weighted"
else:
    graphon_display_name = graphon_name
    mode_str = "Unweighted"

# Log experiment parameters
logging.info(f"{mode_str} transfer experiment parameters:")
logging.info(f"Large graph size (N): {large_graph_n}")
logging.info(f"Training sizes: {n_values}")
logging.info(f"seeds: {seeds}")
if use_weighted:
    logging.info(f"Weighted graphon function {graphon_parameter}: W(x,y) = {graphon_display_name}")
else:
    logging.info(f"Graphon: {graphon_name} (parameter: {graphon_parameter})")
logging.info(f"Training pairs: {num_train_trajectories}")
logging.info(f"Fourier degree: {fourier_degree}")
logging.info(f"Dropout: {dropout_val}")
logging.info(f"Test conditions: {num_test_conditions}")

if use_weighted:
    figure_path = f'outputs/figures_weighted/'
else:
    figure_path = f'outputs/figures/'

if not os.path.exists(figure_path):
    os.makedirs(figure_path)
    logging.info(f"Created directory: {figure_path}")
else:
    logging.info(f"Directory already exists: {figure_path}")

with open('configs/dynamics_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Construct graphs
if use_weighted:
    # Create large weighted test graph
    graph = graphon_utils.create_weighted_graph(large_graph_n, graphon_parameter, device=device)
    logging.info(f"Created large weighted test graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    if graph.number_of_edges() > 0:
        weights = graph.edata['weight']

else:
    # Set up graphon method for unweighted graphs
    graphon_method = None
    
    if graphon_name == "hsbm":
        graphon_method = graphon_utils.generate_deterministic_hsbm_graphon
    elif graphon_name == "hexaflake":
        graphon_method = graphon_utils.generate_hexaflake_graphon
    elif graphon_name == "checkerboard":
        graphon_method = graphon_utils.generate_deterministic_checkerboard_graphon
    elif graphon_name == "sierpinski":
        graphon_method = graphon_utils.generate_sierpinski_carpet_graphon
    elif graphon_name == "koch":
        graphon_method = graphon_utils.generate_koch_snowflake_graphon
    elif graphon_name == "knn":
        graphon_method = graphon_utils.generate_knn_graphon
    
    # Create large test graph from desired graphon
    graphon_resolution: int = config['data']['graphon_resolution']
    W_large = graphon_method(graphon_resolution, graphon_parameter)
    graph = graphon_utils.create_specific_graph(W_large, large_graph_n)
    graph = graph.to(device)
    
    logging.info(f"Created large unweighted test graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

# Access ODE hyperparameters from the config
num_timesteps = config['data']['num_timesteps']
end_time = config['data']['end_time']
lambda_value = 1.0 / large_graph_n

# Create the reaction-diffusion system
ode_func = reaction_diffusion_equation.ReactionDiffusionSystem(graph, device, dynamics_type=dynamics_type, lambda_value=lambda_value)

# Instantiate data generator for the ODE
data_generator = data_utils.ODEDataGeneratorPairs( n_nodes=large_graph_n, device=device, end_time=end_time, num_timesteps=num_timesteps, ode_func=ode_func, num_train_trajectories=1, fourier_degree=fourier_degree)

# Generate multiple test datasets with different initial conditions
logging.info(f"Generating {num_test_conditions} test pairs on large {mode_str.lower()} graph.")
test_datasets = data_generator.generate_multiple_test_datasets(num_test_conditions=num_test_conditions)

logging.info(f"All test datasets generated.")

def evaluate_model_on_multiple_conditions(model, test_datasets, input_dim):
    """
    Evaluate a pair-trained model on multiple test conditions and return averaged error.
    
    Args:
        model: The neural ODE model to evaluate
        test_datasets: List of test datasets with different initial conditions
        input_dim: Input dimension for reshaping
        
    Returns:
        float: avg_relative_error
    """
    model.eval()
    
    relative_errors = []
    
    with torch.no_grad():
        for _, test_data in enumerate(test_datasets):
            
            # Get the single (initial, final) pair from this test dataset
            initial_state, true_final_state = test_data[0]
            initial_state = initial_state.view(large_graph_n, input_dim)
            true_final_state = true_final_state.view(large_graph_n, input_dim)
            
            # Predict final state using full time integration
            predicted_final_state = model(initial_state, T=end_time, time_points=num_timesteps)
            
            # Calculate error metrics
            true_final_np = true_final_state.cpu().numpy()
            pred_final_np = predicted_final_state.cpu().numpy()
            
            # Relative L2 error
            relative_error = np.linalg.norm(true_final_np - pred_final_np) / np.linalg.norm(true_final_np)
            relative_errors.append(relative_error)
    
    # Return averages
    return np.mean(relative_errors)

# Store results across all seeds and sizes
relative_error_list = []

# Main evaluation loop
for seed_of_interest in seeds:
    
    logging.info(f"\nEvaluating models for seed {seed_of_interest}...")
    
    relative_errors_for_seed = []
    
    for n_of_interest in n_values:

        if use_weighted:
            # Weighted checkpoint path (matches main)
            checkpoint_path = f'checkpoints_weighted/Size_{n_of_interest}/dynamics_{dynamics_type}_weighted_nval{n_of_interest}_func{graphon_parameter}_npairs{num_train_trajectories}_fourier{fourier_degree}_seed{seed_of_interest}_best.pth'
        else:
            # Unweighted checkpoint path (matches main)
            checkpoint_path = f'checkpoints/Size_{n_of_interest}/dynamics_{dynamics_type}_nval{n_of_interest}_npairs{num_train_trajectories}_fourier{fourier_degree}_seed{seed_of_interest}_best.pth'
        
        if not os.path.exists(checkpoint_path):
            logging.error(f"Checkpoint not found: {checkpoint_path}")
            logging.error(f"Make sure you've run {'weighted' if use_weighted else 'unweighted'} training with the same parameters!")
            continue

        # Access same training hyperparameters from the config (matches main)
        input_dim, hidden_dim, output_dim = config['model']['model_dims']
        K = config['model']['K']
        activation_class = getattr(nn, config['model']['activation'])
        activation = activation_class()

        # Build the same model infrastructure
        gnn_vector_field = model_defs.ThreeLayerGCN(graph, input_dim, hidden_dim, output_dim, activation, K=K, dropout=dropout_val).to(device)
        model = model_defs.ODEBlock(odefunc=gnn_vector_field, method=config['model']['method'], atol=config['model']['atol'], rtol=config['model']['rtol'], adjoint=config['model']['adjoint']).to(device)

        # Load the weights from the checkpoint
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            logging.info(f"Loaded {mode_str.lower()} pair-trained model (size {n_of_interest}) for testing on size {large_graph_n}")
        except Exception as e:
            logging.error(f"Error loading checkpoint {checkpoint_path}: {e}")
            continue

        # Evaluate model on all test conditions and get averaged errors
        avg_relative_error = evaluate_model_on_multiple_conditions(model, test_datasets, input_dim)
        
        relative_errors_for_seed.append(avg_relative_error)
        
        logging.info(f"  n={n_of_interest}: Relative error={avg_relative_error:.4f}")

    relative_error_list.append(relative_errors_for_seed)

def plot_error_figure(error_list: list, plot_title: str, error_type: str):
    """Convenience method to plot error metric figures in log-log scale with line of best fit.

    Args:
        error_list (list): list of lists, each sublist containing all n_value errors for a seed
        plot_title (str): string to prepend to start of plot title
        error_type (str): type of error for filename
    """
    # Convert to np list
    error_np = np.array(error_list)

    # Calculate the mean and standard deviation across seeds
    mean_errors = np.mean(error_np, axis=0)
    std_errors = np.std(error_np, axis=0)

    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot error bars
    plt.errorbar(n_values, mean_errors, yerr=std_errors, fmt='o-', capsize=5, capthick=2, ecolor='red', markeredgecolor='black', markerfacecolor='blue', linewidth=2, markersize=8, label='Mean $\pm$ Std')

    # Calculate line of best fit in log-log space
    log_n_values = np.log10(n_values)
    log_mean_errors = np.log10(mean_errors)
    
    # Linear regression in log space: log(error) = slope * log(n) + intercept
    slope, intercept, r_value, p_value, std_err = linregress(log_n_values, log_mean_errors)
    
    # Generate fitted line
    log_n_fit = np.linspace(log_n_values[0], log_n_values[-1], 100)
    log_error_fit = slope * log_n_fit + intercept
    n_fit = 10**log_n_fit
    error_fit = 10**log_error_fit
    
    # Plot line of best fit
    plt.plot(n_fit, error_fit, 'r--', linewidth=2, alpha=0.8, label=f'Best fit: n^{slope:.2f} (R²={r_value**2:.3f})')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Training Graph Size (n)')
    plt.ylabel(f'{error_type} Error')
    
    # Create title based on mode
    if use_weighted:
        title = f'{plot_title} Error - {mode_str} Transfer for {dynamics_type} (Averaged over {num_test_conditions} conditions)\n(Function {graphon_parameter}: W(x,y) = {graphon_display_name}, Testing: {large_graph_n} nodes, Time interval: [0, {end_time}])\nConvergence Rate: O(n^{slope:.2f})'
    else:
        title = f'{plot_title} Error - {mode_str} Transfer for {dynamics_type} (Averaged over {num_test_conditions} conditions)\n(Training: {graphon_name}, Testing: {large_graph_n} nodes, Time interval: [0, {end_time}])\nConvergence Rate: O(n^{slope:.2f})'
    
    plt.title(title)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    
    # Save with descriptive filename based on mode
    if use_weighted:
        filename = f'weighted_dynamics_{dynamics_type}_{error_type.lower()}_transfer_func{graphon_parameter}_to_N{large_graph_n}_averaged_{num_test_conditions}_conditions_report.png'
    else:
        filename = f'unweighted_dynamics_{dynamics_type}_{error_type.lower()}_transfer_{graphon_name}_to_N{large_graph_n}_averaged_{num_test_conditions}_conditions_report.png'
    
    plt.savefig(figure_path + filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Log the convergence rate
    logging.info(f"{plot_title} {error_type} convergence rate: O(n^{slope:.3f}) with R^2={r_value**2:.3f}")
    
    return slope, r_value**2  # Return slope and R-squared for summary

relative_slope, relative_r2 = plot_error_figure(relative_error_list, "Relative L2", "Relative")

# Print convergence rates
logging.info(f"Relative L2 error:  O(n^{relative_slope:.3f})  (R²={relative_r2:.3f})")

# Calculate overall statistics
relative_np = np.array(relative_error_list)
logging.info(f"\nRelative L2 error summary (averaged over {num_test_conditions} conditions):")

for i, n in enumerate(n_values):
    mean_err = np.mean(relative_np[:, i])
    std_err = np.std(relative_np[:, i])
    logging.info(f"n={n}: {mean_err:.4f} \pm {std_err:.4f}")

logging.info("Transfer experiment completed.")