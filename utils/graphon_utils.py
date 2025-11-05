import numpy as np
import torch
import math
import dgl
from scipy import ndimage
import scipy.sparse as sp

# General graphon point evaluation method needed for unweighted graphs
def graphon_point_evaluation(W: np.array, x: np.array, y: np.array) -> np.array:
    """
    Evaluate the graphon value at points (x, y) for discretization W taken as a function on the unit square.

    Args:
        W (np.array): Precomputed graphon, size n by n
        x (np.array): numpy array of x-coordinates in [0, 1]
        y (np.array): numpy array of y-coordinates in [0, 1]

    Returns:
        np.array: Values of the graphon evaluated at the given points (x, y)
    """
    n = W.shape[0]

    # Map x, y to indices in [0, n-1]
    x_idx = (np.clip(x, 0, 1) * (n - 1)).astype(int)
    y_idx = (np.clip(y, 0, 1) * (n - 1)).astype(int)
    return W[x_idx, y_idx]

# Unweighted graph generation utility method
def cube_intersects_support(W: np.array, x_min: float, x_max: float, y_min: float, y_max: float, num_samples: int = 5) -> bool:
    """
    Check if the cube defined by [x_min, x_max] x [y_min, y_max] intersects with the support of the graphon W.

    Args:
        W (np.array): Precomputed graphon, size n by n
        x_min (float): Min x-coordinate of square in [0, 1]
        x_max (float): Max x-coordinate of square in [0, 1]
        y_min (float): Min y-coordinate of square in [0, 1]
        y_max (float): Max y-coordinate of square in [0, 1]
        num_samples (int): Number of points in each coordinate to test for support intersection

    Returns:
        bool: Returns True when a sampled point in the square intersects graphon support 
    """
    x_samples = np.linspace(x_min, x_max, num_samples)
    y_samples = np.linspace(y_min, y_max, num_samples)
    return any(graphon_point_evaluation(W, x_samples, y_samples))

def generate_hexaflake_graphon(n: int, depth: int, scale: float = 1.65, rotate: bool = True) -> np.array:
    """
    Generate a hexaflake graphon discretized on an n x n grid.
   
    Args:
        n (int): Grid size for discretization
        depth (int): Number of iterations for hexaflake construction
        scale (float): Scale factor for the hexaflake size (default: 1.65)
        rotate (bool): Whether to rotate the hexagon for x=y symmetry (default: True)
       
    Returns:
        np.array: n x n matrix representing the graphon at the given discretization
    """
    # Initialize graphon matrix
    W = np.zeros((n, n))
    
    # Center coordinates and initial radius
    center_x, center_y = 0.5, 0.5
    radius = 0.3 * scale
    
    # Generate the vertices of the unit hexagon
    unit_hexagon = []
    rotation_angle = math.pi/4 if rotate else 0
    
    for i in range(6):
        angle = i * math.pi / 3 + rotation_angle
        unit_hexagon.append((math.cos(angle), math.sin(angle)))
    
    # Method to test for interior points of hexagon
    def is_point_in_hexagon(px, py, center_x, center_y, radius):
        
        vertices = []
        for vx, vy in unit_hexagon:
            vertices.append((center_x + radius * vx, center_y + radius * vy))
        
        # Use ray casting algorithm for point in polygon test
        inside = False
        j = len(vertices) - 1
        
        for i in range(len(vertices)):
            xi, yi = vertices[i]
            xj, yj = vertices[j]
            
            if ((yi > py) != (yj > py)) and (px < xi + (xj - xi) * (py - yi) / (yj - yi)):
                inside = not inside
            j = i
                
        return inside
    
    def recursive_hexaflake(x, y, r, level):
        if level == 0:
            # Calculate a slightly expanded bounding box to ensure edge points are included
            buffer = 0.5 / n
            min_x = max(0, int((x - r - buffer) * n))
            max_x = min(n, int((x + r + buffer) * n) + 1)
            min_y = max(0, int((y - r - buffer) * n))
            max_y = min(n, int((y + r + buffer) * n) + 1)
            
            # Test all points in the bounding box
            for i in range(min_x, max_x):
                for j in range(min_y, max_y):
                    px = (i + 0.5) / n
                    py = (j + 0.5) / n
                    
                    if is_point_in_hexagon(px, py, x, y, r):
                        W[i, j] = 1.0
            return
        
        # Add central hexagon
        recursive_hexaflake(x, y, r/3, level-1)
        
        # Add 6 surrounding hexagons
        for i in range(6):
            angle = i * math.pi / 3 + rotation_angle
            new_x = x + (2*r/3) * math.cos(angle)
            new_y = y + (2*r/3) * math.sin(angle)
            recursive_hexaflake(new_x, new_y, r/3, level-1)
    
    # Start recursion
    recursive_hexaflake(center_x, center_y, radius, depth)
    
    return W

def generate_deterministic_hsbm_graphon(n: int, num_levels: int) -> np.array:
    """
    Generate a deterministic {0, 1} valued piecewise constant graphon for a hierarchical stochastic block model
    (HSBM) with multiscale community structure.

    Args:
        n (int): Grid size for discretization
        num_levels (int): Levels of hierarchical communities, controls complexity

    Returns:
        np.array: n x n matrix representing the HSBM graphon at the given discretization level
    """
    W = np.zeros((n, n))

    def assign_blocks(start_x, end_x, start_y, end_y, level, value):
        # Recursively assign blocks with alternating {0, 1} values
        if level == 0:
            W[start_x:end_x, start_y:end_y] = value
        else:
            mid_x = (start_x + end_x) // 2
            mid_y = (start_y + end_y) // 2

            # Build sub-blocks
            assign_blocks(start_x, mid_x, start_y, mid_y, level - 1, value)
            assign_blocks(mid_x, end_x, start_y, mid_y, level - 1, 1 - value)
            assign_blocks(start_x, mid_x, mid_y, end_y, level - 1, 1 - value)
            assign_blocks(mid_x, end_x, mid_y, end_y, level - 1, value)

    assign_blocks(0, n, 0, n, num_levels, 0)
    return W

def generate_deterministic_checkerboard_graphon(n: int, num_levels: int) -> np.array:
    """
    Generate a deterministic {0, 1} valued piecewise constant graphon in a checkerboard pattern.
    
    Args:
        n (int): Grid size for discretization
        num_levels (int): Number of squares in each dimension of the checkerboard
    
    Returns:
        np.array: n x n matrix representing the checkerboard graphon
    """
    W = np.zeros((n, n))
    
    if num_levels == 1:
        # Special case: entire square is 0
        return W
    
    # Calculate the size of each square in the checkerboard
    square_size = n // num_levels
    
    for i in range(num_levels):
        for j in range(num_levels):
            # Calculate boundaries of current square
            start_x = i * square_size
            end_x = (i + 1) * square_size if i < num_levels - 1 else n
            start_y = j * square_size
            end_y = (j + 1) * square_size if j < num_levels - 1 else n
            
            # Checkerboard pattern: value is 1 if (i + j) is odd, 0 if even
            # This starts with 0 at origin (0,0) since 0+0=0 (even)
            value = (i + j) % 2
            
            W[start_x:end_x, start_y:end_y] = value
    
    return W

def generate_sierpinski_carpet_graphon(n: int, depth: int) -> np.array:
    """
    Generate a Sierpinski carpet graphon discretized on an nxn grid.
    
    Args:
        n (int): Grid size for discretization
        depth (int): Number of iterations for Sierpinski carpet construction
        
    Returns:
        np.array: n x n matrix representing the graphon at the given discretization
    """

    def is_in_sierpinski_carpet(x: float, y: float, depth: int) -> bool:
        """
        Check if point (x,y) in [0,1]x[0,1] is in the Sierpinski carpet.
        
        Args:
            x, y (float): Coordinates in [0,1]x[0,1]
            depth (int): Number of iterations/precision of the Sierpinski carpet
        
        Returns:
            bool: True if point is in the carpet, False otherwise
        """
        # For each level of recursion, check if both x and y are in middle third
        x_val, y_val = x, y
        for _ in range(depth):
            x_digit = int(x_val * 3) % 3
            y_digit = int(y_val * 3) % 3
            
            # If both coordinates have digit '1' (middle third), point is not in carpet
            if x_digit == 1 and y_digit == 1:
                return False
            
            # Prepare for next iteration
            x_val = x_val * 3 - int(x_val * 3)
            y_val = y_val * 3 - int(y_val * 3)
        
        return True

    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            # Convert grid coordinates to [0,1] interval
            x = i / n
            y = j / n
            
            # Set W[i,j] = 1 if (x,y) is in the Sierpinski carpet
            if is_in_sierpinski_carpet(x, y, depth):
                W[i, j] = 1.0
    
    return W

def koch_snowflake_boundary_iterative(depth: int, center_x: float=0.5, center_y: float=0.5, size: float=0.25):
    """
    Generate the Koch curve boundary to the given iterative depth.
    
    Args:
        depth (int): Number of iterations for Koch curve construction
        center_x, center_y (floats): Coordinates of koch curve center
        size (float): Scaling factor for curve construction
        
    Returns:
        np.array: list of vertices for the boundary curve
    """
    # Calculate the height of the equilateral triangle
    height = size * np.sqrt(3) / 2
    
    # Initial triangle vertices (centered in the unit square)
    vertices = np.array([
        [center_x, center_y + 2*height/3],
        [center_x - size/2, center_y - height/3],
        [center_x + size/2, center_y - height/3],
        [center_x, center_y + 2*height/3]
    ])
    
    # Each iteration replaces each line segment with 4 smaller segments
    for _ in range(depth):
        new_vertices = []
        
        # Process each edge of the current shape
        for i in range(len(vertices) - 1):
            p0 = vertices[i]
            p4 = vertices[i+1]
            v = p4 - p0
            
            # Calculate the four points that will replace this segment
            p1 = p0 + v / 3
            p3 = p0 + 2 * v / 3
            
            # The new point p2 is found by rotating the vector (p3-p1) by -60 degrees (outward)
            rot = np.array([
                [np.cos(-np.pi/3), -np.sin(-np.pi/3)],
                [np.sin(-np.pi/3), np.cos(-np.pi/3)]
            ])
            
            p2 = p1 + np.dot(rot, p3 - p1)
            
            # Add these points to the new vertices
            new_vertices.append(p0)
            new_vertices.append(p1)
            new_vertices.append(p2)
            new_vertices.append(p3)
        
        # Add the final vertex to close the loop
        new_vertices.append(vertices[-1])
        
        # Update vertices for the next iteration
        vertices = np.array(new_vertices)
    
    return vertices


def generate_koch_snowflake_graphon(n: int, depth: int) -> np.array:
    """
    Generate an n by n binary grid with the Koch snowflake boundary and filled interior.
    
    Args:
        n (int): Grid size
        depth (int): Fractal depth
    
    Returns:
        n by n binary matrix representing the filled snowflake
    """
   
    print(f"Generating filled Koch snowflake with n={n}, depth={depth}")
    
    # Get the snowflake boundary points
    points = koch_snowflake_boundary_iterative(depth)
    
    # Initialize empty grid
    grid = np.zeros((n, n), dtype=bool)
    
    # Scale points to grid coordinates
    grid_points = np.floor(points * (n-1)).astype(int)
    
    # Draw lines between consecutive points to ensure a closed boundary
    for i in range(len(grid_points) - 1):
        p1 = grid_points[i]
        p2 = grid_points[i+1]
        
        # Use Bresenham's line algorithm to draw a line between points
        x1, y1 = p1
        x2, y2 = p2
        
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        while True:
            if 0 <= x1 < n and 0 <= y1 < n:
                grid[y1, x1] = True
                
            if x1 == x2 and y1 == y2:
                break
                
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy
    
    # Fill the interior of the snowflake
    filled_grid = ndimage.binary_fill_holes(grid)
    
    return filled_grid


def is_point_in_triangle(x: float, y: float, triangle: list) -> bool:
    """
    Check if a point is inside a triangle using barycentric coordinates.
    
    Args:
        x, y (float): Coordinates of the point
        triangle (list): List of 3 (x, y) coordinates forming a triangle
        
    Returns:
        bool: True if point is inside the triangle, False otherwise
    """
    def sign(p1x, p1y, p2x, p2y, p3x, p3y):
        return (p1x - p3x) * (p2y - p3y) - (p2x - p3x) * (p1y - p3y)
    
    x1, y1 = triangle[0]
    x2, y2 = triangle[1]
    x3, y3 = triangle[2]
    
    d1 = sign(x, y, x1, y1, x2, y2)
    d2 = sign(x, y, x2, y2, x3, y3)
    d3 = sign(x, y, x3, y3, x1, y1)
    
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    
    # Also check if point is on the boundary (any d value is close to 0)
    epsilon = 1e-9
    on_edge = (abs(d1) < epsilon) or (abs(d2) < epsilon) or (abs(d3) < epsilon)
    
    return on_edge or not (has_neg and has_pos)

def generate_koch_snowflake_vertices(base_vertices: list, depth: int) -> list:
    """
    Check if a point is inside a triangle using barycentric coordinates.
    
    Args:
        base_vertices (list): List of base generated vertices
        depth (int): Recursive depth of construction for the fractal
        
    Returns:
        np.array: Vertex list for construction
    """
    vertices = base_vertices.copy()
    
    for _ in range(depth):
        new_vertices = []
        for i in range(len(vertices)):
            p0 = vertices[i]
            p1 = vertices[(i + 1) % len(vertices)]
            
            new_vertices.append(p0)
            
            # Calculate new points for the Koch transformation
            p2 = (p0[0] + (p1[0] - p0[0])/3, p0[1] + (p1[1] - p0[1])/3)
            p4 = (p0[0] + 2*(p1[0] - p0[0])/3, p0[1] + 2*(p1[1] - p0[1])/3)
            
            # Calculate the equilateral triangle point
            dx = p4[0] - p2[0]
            dy = p4[1] - p2[1]
            length = math.sqrt(dx*dx + dy*dy)
            angle = math.atan2(dy, dx) + math.pi/3
            p3 = (p2[0] + length * math.cos(angle), p2[1] + length * math.sin(angle))
            
            new_vertices.append(p2)
            new_vertices.append(p3)
            new_vertices.append(p4)
            
        vertices = new_vertices
    
    return vertices

def point_in_polygon(x: float, y: float, vertices: list) -> bool:
    """
    Determine if a point is inside or on a polygon.
    
    Args:
        x (float): x-coordinate to check
        y (float): y-coordinate to check
        vertices (list): List of polygon vertices
        
    Returns:
        bool: True iff point is inside or on polygon
    """
    # First check if point is on any edge
    for i in range(len(vertices)):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % len(vertices)]
        
        if is_point_on_line_segment(x, y, x1, y1, x2, y2):
            return True
    
    # Ray casting algorithm
    inside = False
    j = len(vertices) - 1
    for i in range(len(vertices)):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        
        if ((yi > y) != (yj > y)) and (x < xi + (xj - xi) * (y - yi) / (yj - yi)):
            inside = not inside
        j = i
            
    return inside

def is_point_on_line_segment(x: float, y: float, x1: float, y1: float, x2: float, y2: float, epsilon: float = 1e-9) -> bool:
    """
    Determine if a point is directly on a line segment (polygon boundary).
    
    Args:
        x, y (floats): coordinates of the point to check
        x1, y1 (floats): coordinates of first line segment endpoint
        x2, y2 (floats): coordinates of second line segment endpoint
        epsilon (float): Small numerical error bound
        
    Returns:
        bool: True iff point is on line segment
    """
    cross_product = abs((y - y1) * (x2 - x1) - (x - x1) * (y2 - y1))
    scaled_epsilon = epsilon * max(1.0, abs(x2 - x1), abs(y2 - y1))
    
    if cross_product > scaled_epsilon:
        return False
    
    # Check if point is within the bounding box
    if x < min(x1, x2) - epsilon or x > max(x1, x2) + epsilon:
        return False
    if y < min(y1, y2) - epsilon or y > max(y1, y2) + epsilon:
        return False
    
    return True

def generate_knn_graphon(n: int, k: int) -> np.array:
    """
    Generate a k-nearest neighbors graphon as a continuous function on [0,1]x[0,1].
    
    Args:
        n (int): Grid size for discretization 
        k (int): Number of nearest neighbors (determines band width)
        
    Returns:
        np.array: n x n matrix representing the k-NN graphon
    """
    W = np.zeros((n, n))
    
    # Calculate band width for k neighbors on each side means band extends k/(2*n) in each direction
    band_width = (k + 1) / (2 * n)

    for i in range(n):
        for j in range(n):
            # Convert grid indices to [0,1] coordinates
            x = (i + 0.5) / n
            y = (j + 0.5) / n
            
            # Check if point is in the diagonal band |x - y| <= band_width
            if abs(x - y) <= band_width:
                W[i, j] = 1.0
            
            # Check corner regions for wraparound connections
            elif abs(x - y + 1) <= band_width:
                W[i, j] = 1.0
            
            elif abs(x - y - 1) <= band_width: 
                W[i, j] = 1.0
    
    return W

def weighted_function_graphon(x: float, y: float, graphon_parameter: int) -> float:
    """
    Weighted graphon function that supports multiple W(x,y) functions based on parameter.
    
    Args:
        x (float): x-coordinate in [0,1]
        y (float): y-coordinate in [0,1]
        graphon_parameter (int): Selects which W(x,y) function to use (0-5)
    
    Returns:
        float: Computed point value of the selected graphon function
    """
    if graphon_parameter == 0:
        return max(0, 1 - abs(x - y))
    elif graphon_parameter == 1:
        return math.exp(-(x**0.7 + y**0.7))
    elif graphon_parameter == 2:
        return 0.25 * (x**2 + y**2 + math.sqrt(x) + math.sqrt(y))
    elif graphon_parameter == 3:
        return 0.5 * (x + y)
    elif graphon_parameter == 4:
        return 1.0 / (1.0 + math.exp(-2.0 * (x**2 + y**2)))
    elif graphon_parameter == 5:
        return max(0, 1 - (abs(x - y) ** 0.5))
    else:
        raise ValueError(f"graphon_parameter {graphon_parameter} not supported. Must be 0-5.")


def create_weighted_graph(graph_size: int, graphon_parameter: int, device=None) -> dgl.DGLGraph:
    """
    Create a connected DGL graph with weighted edges by evaluating a weighted graphon function.
    
    Args:
        graph_size (int): Number of nodes in the graph
        graphon_parameter (int): Which weighted graphon function to use (0-4)
        device: PyTorch device to place the graph on (optional)
        
    Returns:
        dgl.DGLGraph: Weighted graph with edge weights stored in g.edata['weight']
    """
    src_nodes = []
    dst_nodes = []
    weights = []
    
    # For each pair of nodes (i,j) where i != j
    for i in range(graph_size):
        for j in range(graph_size):
            if i != j:
                # Map node indices to [0,1] interval
                x = (i + 0.5) / graph_size
                y = (j + 0.5) / graph_size
                
                # Evaluate the weighted graphon function
                weight = weighted_function_graphon(x, y, graphon_parameter)
                
                # Add edge if weight is non-zero
                if weight > 0:
                    src_nodes.append(i)
                    dst_nodes.append(j)
                    weights.append(weight)
    
    # Create the DGL graph
    g = dgl.graph((torch.tensor(src_nodes), torch.tensor(dst_nodes)))
    
    # Add edge weights as edge data
    g.edata['weight'] = torch.tensor(weights, dtype=torch.float32)
    
    # Add node normalization (standard GCN normalization)
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    g.ndata['norm'] = norm.unsqueeze(1)
    
    # Move to device if specified
    if device is not None:
        g = g.to(device)
    
    return g


def get_weighted_graphon_name(graphon_parameter: int) -> str:
    """
    Get a human-readable name/formula for a weighted graphon function.
    
    Args:
        graphon_parameter (int): The graphon function identifier (0-5)
        
    Returns:
        str: Mathematical formula as a string
    """
    names = {
        0: "xy",
        1: "exp(-(x^0.7 + y^0.7))",
        2: "0.25*(x^2 + y^2 + sqrt(x) + sqrt(y))",
        3: "0.5*(x + y)",
        4: "1/(1 + exp(-2*(x^2 + y^2)))",
        5: "Holder Tent"
    }
    
    if graphon_parameter not in names:
        raise ValueError(f"graphon_parameter {graphon_parameter} not supported. Must be 0-5.")
    
    return names[graphon_parameter]

def create_specific_graph(W: np.array, graph_size: int) -> dgl.DGLGraph:
    """
    Create an unweighted DGL graph using the given graphon construction method to create the adjacency matrix.
    
    Args:
        W (np.array): Precomputed matrix representation of the graphon
        graph_size (int): Size of adjacency matrix to be generated
        
    Returns:
        dgl.DGLGraph: Adjacency matrix for the graph of the desired size
    """

    partition_size = 1.0 / graph_size
    adj_matrix = np.zeros((graph_size, graph_size))

    # Loop through each cell in the grid and determine intersection with the graphon
    for i in range(graph_size):
        for j in range(i):
            x_min = i * partition_size
            x_max = (i + 1) * partition_size
            y_min = j * partition_size
            y_max = (j + 1) * partition_size
           
            # Check if the cell intersects with the graphon support
            if cube_intersects_support(W, x_min, x_max, y_min, y_max):
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1
                
    # Add norm information
    g = dgl.from_scipy(sp.coo_matrix(adj_matrix))
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    g.ndata['norm'] = norm.unsqueeze(1)
    return g