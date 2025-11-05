# Import external libraries
import argparse
import logging
import yaml
import numpy as np
import os
import random

# Import torch libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
parser = argparse.ArgumentParser(description='Train Neural ODE with (initial, final) pair training on given dynamics and graph setup')
parser.add_argument('--dynamics', type=str, required=True, help='Type of dynamics to use for data generation (heat_sin, heat_linear, allen_cahn, consensus)')
parser.add_argument('--nval', type=int, required=True, help='N val for training graph size')
parser.add_argument('--use_weighted', type=bool, required=False, default="", help='Use weighted graphs instead of unweighted graphs. Empty string for unweighted, any nonempty string will result in weighted.')
parser.add_argument('--graphon', type=str, required=True, help='Graphon to use for unweighted graphs (hsbm, hexaflake, checkerboard, sierpinski, koch, knn)')
parser.add_argument('--graphon_parameter', type=int, required=True, help='Graphon parameter: for unweighted (general parameter), for weighted (function index 0-5)')
parser.add_argument('--seed', type=int, required=False, default=0, help='Seed for reproducibility (optional)')
parser.add_argument('--num_train_trajectories', type=int, required=False, default=50, help='Number of (initial, final) pairs for training')
parser.add_argument('--fourier_degree', type=int, required=False, default=10, help='Degree of Fourier polynomials for initial conditions')
parser.add_argument('--dropout', type=float, required=False, default=0.0, help='Dropout value for GCN layers')

args = parser.parse_args()

# Reference args
training_graph_size = int(args.nval)
dynamics_type = args.dynamics
use_weighted = args.use_weighted
graphon_name = args.graphon
graphon_parameter = args.graphon_parameter
seed = args.seed
num_train_trajectories = args.num_train_trajectories
fourier_degree = args.fourier_degree
dropout_val = args.dropout

# Validate graphon parameters based on mode
if use_weighted:
    if graphon_parameter not in [0, 1, 2, 3, 4, 5]:
        raise ValueError('For weighted mode, graphon_parameter must be an integer from 0 to 5.')
else:
    if graphon_name not in ["hsbm", "hexaflake", "checkerboard", "sierpinski", "koch", "knn"]:
        raise ValueError('For unweighted mode, graphon must be one of: hsbm, hexaflake, checkerboard, sierpinski, koch, knn.')

# Get descriptive graphon name for output
if use_weighted:
    graphon_display_name = graphon_utils.get_weighted_graphon_name(graphon_parameter)
    mode_str = "Weighted"
else:
    graphon_display_name = graphon_name
    mode_str = "Unweighted"

# Log experiment parameters
logging.info(f"{mode_str} Pair Training Experiment Parameters:")
logging.info(f"Graph size (N): {training_graph_size}")
if use_weighted:
    logging.info(f"Weighted graphon function {graphon_parameter}: W(x,y) = {graphon_display_name}")
else:
    logging.info(f"Graphon: {graphon_name} (parameter: {graphon_parameter})")
logging.info(f"Seed: {seed}")
logging.info(f"Training pairs: {num_train_trajectories}")
logging.info(f"Fourier degree: {fourier_degree}")
logging.info(f"Dropout: {dropout_val}")

# To save checkpoints, first verify checkpoint folders exist or make new ones
if use_weighted:
    dir_path = f'checkpoints_weighted/Size_{training_graph_size}'
    checkpoint_filename = dir_path + f'/dynamics_{dynamics_type}_weighted_nval{training_graph_size}_func{graphon_parameter}_npairs{num_train_trajectories}_fourier{fourier_degree}_seed{seed}'
else:
    dir_path = f'checkpoints/Size_{training_graph_size}'
    checkpoint_filename = dir_path + f'/dynamics_{dynamics_type}_nval{training_graph_size}_npairs{num_train_trajectories}_fourier{fourier_degree}_seed{seed}'

if not os.path.exists(dir_path):
    os.makedirs(dir_path)
    logging.info(f"Created directory: {dir_path}")
else:
    logging.info(f"Directory already exists: {dir_path}")

# Set seed for reproducibility
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(seed)

# Load config file
with open('configs/dynamics_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Conditional graph construction
if use_weighted:
    
    g = graphon_utils.create_weighted_graph(training_graph_size, graphon_parameter, device=device)
    logging.info(f"Created weighted graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
    
    if g.number_of_edges() > 0:
        weights = g.edata['weight']

else:

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
    
    # Generate graphon and create unweighted graph
    graphon_resolution: int = config['data']['graphon_resolution']
    W = graphon_method(graphon_resolution, graphon_parameter)
    
    g = graphon_utils.create_specific_graph(W, training_graph_size)
    g = g.to(device)
    logging.info(f"Created unweighted graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")

# Access ODE hyperparameters from the config
num_timesteps = config['data']['num_timesteps']
end_time = config['data']['end_time']
lambda_value = 1.0 / training_graph_size

# Create the reaction-diffusion system
ode_func = reaction_diffusion_equation.ReactionDiffusionSystem(g, device, dynamics_type=dynamics_type, lambda_value=lambda_value)

# Instantiate data generator for pair training
data_generator = data_utils.ODEDataGeneratorPairs(n_nodes=training_graph_size, device=device, end_time=end_time, num_timesteps=num_timesteps, ode_func=ode_func, num_train_trajectories=num_train_trajectories, fourier_degree=fourier_degree)

# Generate and store datasets
logging.info(f"Generating (initial, final) pair datasets.")
datasets = data_generator.generate_all_datasets(base_seed=seed)
train_data = datasets['train']
val_data = datasets['val']
test_data = datasets['test']

# Create DataLoaders
train_data_loader = DataLoader(train_data, shuffle=True)
val_data_loader = DataLoader(val_data)
test_data_loader = DataLoader(test_data)

# Access training hyperparameters from the config
num_epochs = config['training']['num_epochs']
input_dim, hidden_dim, output_dim = config['model']['model_dims']
K = config['model']['K']
activation_class = getattr(nn, config['model']['activation'])
activation = activation_class()

# Initialize a three layer GCN serving as our vector field
gnn_vector_field = model_defs.ThreeLayerGCN(g, input_dim, hidden_dim, output_dim, activation, K=K, dropout=dropout_val).to(device)

# Initialize the corresponding Neural ODE model
model = model_defs.ODEBlock(odefunc=gnn_vector_field, method=config['model']['method'], atol=config['model']['atol'], rtol=config['model']['rtol'], adjoint=config['model']['adjoint']).to(device)

# Define the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
criterion = nn.MSELoss()

# Hyperparameters for output/checkpointing frequency
epochs_btw_checkpoints = config['training']['epochs_btw_checkpoints']
epochs_btw_val_runs = config['training']['epochs_btw_val_runs']

# Set up early stopping infrastructure
best_val_loss = float('inf')
patience = config['training']['early_stopping_patience']
patience_counter = 0

# Learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config['training']['scheduler_factor'], patience=config['training']['scheduler_patience'])

# Convenience method for easy validation and test evals
def dataset_eval(model: nn.Module, data_loader: DataLoader) -> float:
    """Evaluates model on the given DataLoader for (initial, final) pair performance.

    Args:
        model (nn.Module): Full Neural ODE model used in training
        data_loader (DataLoader): Validation or Test DataLoader

    Returns:
        float: Total DataLoader loss 
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for initial_state, final_state in data_loader:
            initial_state = initial_state.view(training_graph_size, input_dim)
            final_state = final_state.view(training_graph_size, input_dim)
            
            # Integrate over full time interval for training pairs
            predicted_final_state = model(initial_state, T=end_time, time_points=num_timesteps)
            
            loss = criterion(predicted_final_state, final_state)
            total_loss += loss.item()
    return total_loss / len(data_loader)

# Start main training loop
logging.info(f"Starting model training loop:")

for epoch in range(num_epochs):

    model.train()
    epoch_loss = 0.0

    for initial_state, final_state in train_data_loader:

        optimizer.zero_grad()
        initial_state = initial_state.view(training_graph_size, input_dim)
        final_state = final_state.view(training_graph_size, input_dim)
        predicted_final_state = model(initial_state, T=end_time, time_points=num_timesteps)
        loss = criterion(predicted_final_state, final_state)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_train_loss = epoch_loss / len(train_data_loader)
    
    # If time, perform a validation run
    if (epoch + 1) % epochs_btw_val_runs == 0:
       
        avg_val_loss = dataset_eval(model, val_data_loader)
        logging.info(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')

        # Step learning rate based on validation loss
        scheduler.step(avg_val_loss)

        # Test for early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save the best model yet
            torch.save(model.state_dict(), f"{checkpoint_filename}_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    else:
        logging.info(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}')

    # If time, perform mandatory checkpointing
    if (epoch + 1) % epochs_btw_checkpoints == 0:
        torch.save(model.state_dict(), f"{checkpoint_filename}_{epoch+1}.pth")
    
logging.info(f"{mode_str} pair training completed.")

# Test final model on test set
logging.info(f'Avg Test Loss (Final Model): {dataset_eval(model, test_data_loader):.6f}')

# Test best validation loss model on test set to compare
model.load_state_dict(torch.load(f'{checkpoint_filename}_best.pth'))
logging.info(f'Avg Test Loss (Best Validation Model): {dataset_eval(model, test_data_loader):.6f}')

logging.info(f"Experiment completed, checkpoints saved with prefix: {checkpoint_filename}")