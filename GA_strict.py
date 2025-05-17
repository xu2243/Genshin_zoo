import random
import numpy as np
from deap import base, creator, tools, algorithms
import copy
import math
import time
import itertools
from tqdm import tqdm
import os
import datetime
import concurrent.futures # Import for parallelization
from typing import List, Tuple, Dict, Any, Set, Optional

# --- 1. 定义常量和拼图形状 ---

GRID_HEIGHT: int = 6
GRID_WIDTH: int = 7

# 基础形状定义 (相对于 (0,0) 的坐标偏移)
# 使用 List[Tuple[int, int]] 作为类型提示
SPECIAL_SHAPES_BASE: List[List[Tuple[int, int]]] = [
    [(0, 0), (0, 1), (0, 2)],  # 1x3
    [(0, 0), (0, 1), (0, 2), (0, 3)], # 1x4
    [(0, 0), (0, 1), (1, 0)],  # L shape
    [(0, 0), (0, 1), (1, 1), (1, 2)],  # Z shape
]
SPECIAL_SIZES: List[int] = [3, 4, 3, 4]

LAND_SHAPES_BASE: List[List[Tuple[int, int]]] = [
    [(0, 0), (0, 1), (0, 2)],  # 1x3
    [(0, 0), (0, 1), (1, 0)],  # L shape
    [(0, 0), (0, 1), (1, 0), (1, 1)],  # Square
    [(0, 0), (0, 1), (0, 2), (1, 1)],  # T shape
]
LAND_SIZES: List[int] = [3, 3, 4, 4]

LARGE_SHAPES_BASE: List[List[Tuple[int, int]]] = [
    [(0, 0), (0, 1), (0, 2), (1, 2)],
    [(0, 0), (0, 1), (0, 2), (1, 1)],
    [(0, 1), (0, 2), (1, 0), (1, 1)],
    [(0, 2), (1, 0), (1, 1), (1, 2)],
]
LARGE_SIZES: List[int] = [4, 4, 4, 4]

MAGIC_SHAPES_BASE: List[List[Tuple[int, int]]] = [
    [(0, 0)], # Magic 1 (1x1)
    [(0, 0)], # Magic 2 (1x1)
]
MAGIC_SIZES: List[int] = [1, 1]

# 拼图库存 (类型, 形状索引, 拷贝索引)
# 类型: 0=Special, 1=Land, 2=Large, 3=Magic1, 4=Magic2
# 使用 Tuple[int, int, int] 作为 piece_id 的类型提示
PieceId = Tuple[int, int, int]
PIECE_INVENTORY: List[PieceId] = []
for i in range(4): # 4 shapes per type
    for j in range(4): # 4 copies per shape
        PIECE_INVENTORY.append((0, i, j)) # Special
        PIECE_INVENTORY.append((1, i, j)) # Land
        PIECE_INVENTORY.append((2, i, j)) # Large
for j in range(2): # 2 copies per magic type
    PIECE_INVENTORY.append((3, 0, j)) # Magic 1
    PIECE_INVENTORY.append((4, 0, j)) # Magic 2

TOTAL_PIECES_COUNT: int = len(PIECE_INVENTORY) # 52

PIECE_TYPE_NAMES: Dict[int, str] = {
    0: "Special",
    1: "Land",
    2: "Large",
    3: "Magic1",
    4: "Magic2"
}

PIECE_SHAPES: List[List[List[Tuple[int, int]]]] = [SPECIAL_SHAPES_BASE, LAND_SHAPES_BASE, LARGE_SHAPES_BASE, MAGIC_SHAPES_BASE, MAGIC_SHAPES_BASE]
PIECE_SIZES: List[List[int]] = [SPECIAL_SIZES, LAND_SIZES, LARGE_SIZES, MAGIC_SIZES, MAGIC_SIZES]

def rotate_shape(shape: List[Tuple[int, int]], rotation: int) -> List[Tuple[int, int]]:
    rotated: List[Tuple[int, int]] = []
    for r, c in shape:
        if rotation == 0: rotated.append((r, c))
        elif rotation == 90: rotated.append((c, -r))
        elif rotation == 180: rotated.append((-r, -c))
        elif rotation == 270: rotated.append((-c, r))
        else: raise ValueError("Invalid rotation angle")
    return rotated

def normalize_shape(shape: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not shape: return []
    min_r: int = min(r for r, c in shape)
    min_c: int = min(c for r, c in shape)
    return sorted([(r - min_r, c - min_c) for r, c in shape])

# Store rotations as normalized lists of tuples
ALL_ROTATED_SHAPES: Dict[Tuple[int, int], List[List[Tuple[int, int]]]] = {}
for type_idx in range(len(PIECE_SHAPES)):
    for shape_idx, base_shape in enumerate(PIECE_SHAPES[type_idx]):
        unique_rotations: List[List[Tuple[int, int]]] = []
        seen_normalized: Set[Tuple[Tuple[int, int], ...]] = set()
        for rot_angle in [0, 90, 180, 270]:
            rotated = rotate_shape(base_shape, rot_angle)
            normalized = tuple(normalize_shape(rotated)) # Use tuple for set hashing
            if normalized not in seen_normalized:
                seen_normalized.add(normalized)
                unique_rotations.append(list(normalized)) # Store as list
        ALL_ROTATED_SHAPES[(type_idx, shape_idx)] = unique_rotations

# Helper to get shape dimensions after rotation and normalization
def get_shape_dimensions(shape_coords: List[Tuple[int, int]]) -> Tuple[int, int]:
    if not shape_coords: return 0, 0
    min_r: int = min(r for r, c in shape_coords)
    max_r: int = max(r for r, c in shape_coords)
    min_c: int = min(c for r, c in shape_coords)
    max_c: int = max(c for r, c in shape_coords)
    return max_r - min_r + 1, max_c - min_c + 1

# Pre-calculate max dimensions for position generation range
MAX_PIECE_HEIGHT: int = 0
MAX_PIECE_WIDTH: int = 0
for shapes in PIECE_SHAPES:
    for shape in shapes:
        for rot_angle in [0, 90, 180, 270]:
            rotated_shape = rotate_shape(shape, rot_angle)
            h, w = get_shape_dimensions(rotated_shape)
            MAX_PIECE_HEIGHT = max(MAX_PIECE_HEIGHT, h)
            MAX_PIECE_WIDTH = max(MAX_PIECE_WIDTH, w)

# --- 2. 定义染色体表示 (包含位置信息) ---

# Gene: (piece_inventory_id, rotation_index, row, col)
# Chromosome: [gene1, gene2, ...]
# 使用 Tuple[PieceId, int, int, int] 作为 Gene 的类型提示
Gene = Tuple[PieceId, int, int, int]
# Individual is a list of Genes

# DEAP setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# --- Helper function to calculate score for a *given* grid state ---
# This function remains the same as it operates on the final grid/placed_info
def calculate_total_score_from_placement(grid: List[List[Optional[PieceId]]], placed_pieces_info: Dict[PieceId, Dict[str, Any]]) -> int:
    """Calculates the total score based on a completed grid and placed pieces info."""
    total_score: int = 0
    placed_counts: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0} # Count placed pieces by type

    # Base Score
    for piece_id, info in placed_pieces_info.items():
        placed_counts[info['type']] += 1
        if info['type'] in [0, 1]: # Special or Land
            total_score += 6 if info['size'] == 3 else 12
        elif info['type'] == 2: # Large
            total_score += 4
        # Magic pieces have base score 0

    # Category Count Bonus
    if placed_counts[0] >= 3: total_score += 10 # Special
    if placed_counts[1] >= 3: total_score += 10 # Land
    if placed_counts[2] >= 3: total_score += 20 # Large

    # Large Creature Adjacency Bonus (+2 per distinct adjacent piece)
    for piece_id, info in placed_pieces_info.items():
        if info['is_large']:
            adjacent_piece_ids: Set[PieceId] = set()
            for r, c in info['cells']:
                # Check 4 neighbors (up, down, left, right)
                neighbors: List[Tuple[int, int]] = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
                for nr, nc in neighbors:
                    if 0 <= nr < GRID_HEIGHT and 0 <= nc < GRID_WIDTH:
                        neighbor_piece_id: Optional[PieceId] = grid[nr][nc]
                        # Check if neighbor cell is occupied by a *different* piece that was successfully placed
                        if neighbor_piece_id is not None and neighbor_piece_id != piece_id and neighbor_piece_id in placed_pieces_info:
                            adjacent_piece_ids.add(neighbor_piece_id)
            total_score += len(adjacent_piece_ids) * 2

    # Magic Creature Bonuses
    for piece_id, info in placed_pieces_info.items():
        if info['is_magic1']:
            affected_piece_ids: Set[PieceId] = set()
            for r, c in info['cells']: # Magic 1 is 1x1, so only one cell
                 # Check 4 neighbors (up, down, left, right)
                neighbors = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
                for nr, nc in neighbors:
                    if 0 <= nr < GRID_HEIGHT and 0 <= nc < GRID_WIDTH:
                        neighbor_piece_id = grid[nr][nc]
                         # Check if neighbor cell is occupied by a *different* piece that was successfully placed
                        if neighbor_piece_id is not None and neighbor_piece_id != piece_id and neighbor_piece_id in placed_pieces_info:
                            affected_piece_ids.add(neighbor_piece_id)
            total_score += len(affected_piece_ids) * 3 # +3 per affected piece

        elif info['is_magic2']:
            affected_piece_ids: Set[PieceId] = set()
            for r, c in info['cells']: # Magic 2 is 1x1, so only one cell
                # Check 8 neighbors (including diagonals)
                neighbors = [(r-1, c-1), (r-1, c), (r-1, c+1),
                             (r, c-1),             (r, c+1),
                             (r+1, c-1), (r+1, c), (r+1, c+1)]
                for nr, nc in neighbors:
                    if 0 <= nr < GRID_HEIGHT and 0 <= nc < GRID_WIDTH:
                        neighbor_piece_id = grid[nr][nc]
                         # Check if neighbor cell is occupied by a *different* piece that was successfully placed
                        if neighbor_piece_id is not None and neighbor_piece_id != piece_id and neighbor_piece_id in placed_pieces_info:
                            affected_piece_ids.add(neighbor_piece_id)
            total_score += len(affected_piece_ids) * 2 # +2 per affected piece

    return total_score

# --- Helper function to get placement details from an individual (new representation) ---
# This implements the STRICT placement logic.
def get_grid_and_info_from_individual(individual: List[Gene]) -> Tuple[List[List[Optional[PieceId]]], Dict[PieceId, Dict[str, Any]]]:
    """
    Processes an individual (list of proposed placements) and returns the resulting
    grid and placed_pieces_info after resolving conflicts based on list order (STRICT placement).
    """
    grid: List[List[Optional[PieceId]]] = [[None for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
    placed_pieces_info: Dict[PieceId, Dict[str, Any]] = {}
    used_inventory_ids: Set[PieceId] = set() # Keep track of which piece instances are successfully placed

    for gene in individual:
        # Gene: (piece_inventory_id, rotation_index, row, col)
        piece_id, rot_idx, start_r, start_c = gene
        p_type, p_shape_idx, p_copy_idx = piece_id

        # Skip if this piece instance is already placed (shouldn't happen if individual is built correctly, but safety)
        if piece_id in used_inventory_ids:
            continue

        try:
            # Get the shape coordinates for the specified rotation
            shape_coords_normalized: List[Tuple[int, int]] = ALL_ROTATED_SHAPES[(p_type, p_shape_idx)][rot_idx]
        except (IndexError, KeyError):
             # Invalid piece_id or rot_idx in the gene, skip this placement attempt
             continue

        # Calculate the absolute grid cells this piece would occupy
        current_piece_cells: List[Tuple[int, int]] = []
        is_valid_spot: bool = True
        for dr, dc in shape_coords_normalized:
            grid_row: int = start_r + dr
            grid_col: int = start_c + dc

            # Check bounds
            if not (0 <= grid_row < GRID_HEIGHT and 0 <= grid_col < GRID_WIDTH):
                is_valid_spot = False
                break
            # Check overlap with already placed pieces in the current grid being built
            if grid[grid_row][grid_col] is not None:
                is_valid_spot = False
                break
            current_piece_cells.append((grid_row, grid_col))

        # If the spot is valid, place the piece
        if is_valid_spot:
            # Place the piece at the chosen spot
            for cell_r, cell_c in current_piece_cells:
                grid[cell_r][cell_c] = piece_id # Mark grid with piece ID

            # Add details for the newly placed piece
            placed_pieces_info[piece_id] = {
                'type': p_type,
                'shape_idx': p_shape_idx,
                'copy_idx': p_copy_idx,
                'cells': current_piece_cells,
                'is_large': p_type == 2,
                'is_magic1': p_type == 3,
                'is_magic2': p_type == 4,
                'size': PIECE_SIZES[p_type][p_shape_idx] if p_type < 3 else 1,
                'rotation_idx': rot_idx, # Store for visualization/LS
                'start_pos': (start_r, start_c) # Store the proposed start pos from the gene
            }
            used_inventory_ids.add(piece_id)

    return grid, placed_pieces_info

# --- Helper function to find the first valid placement position for a piece ---
def find_first_valid_position(grid: List[List[Optional[PieceId]]], piece_id: PieceId, rot_idx: int) -> Optional[Tuple[int, int]]:
    """
    Searches for the first valid position (top-left corner of bounding box) for a piece
    in the given grid, iterating through all possible start positions within the
    extended grid bounds in row-major order.
    Returns (r, c) of the first valid spot found, or None if no spot is found.
    """
    p_type, p_shape_idx, p_copy_idx = piece_id
    try:
        shape_coords_normalized: List[Tuple[int, int]] = ALL_ROTATED_SHAPES[(p_type, p_shape_idx)][rot_idx]
    except (IndexError, KeyError):
         return None # Invalid piece or rotation

    # Iterate through all possible top-left corners (including those slightly outside for sliding in)
    # Range: [-max_h + 1, GRID_HEIGHT - 1] for row, [-max_w + 1, GRID_WIDTH - 1] for col
    # Iterate in row-major order
    for r in range(-MAX_PIECE_HEIGHT + 1, GRID_HEIGHT):
        for c in range(-MAX_PIECE_WIDTH + 1, GRID_WIDTH):
            is_valid_spot: bool = True
            # Check if placing the piece with its top-left at (r, c) is valid
            for dr, dc in shape_coords_normalized:
                grid_row: int = r + dr
                grid_col: int = c + dc

                # Check bounds (all cells of the piece must land within the grid)
                if not (0 <= grid_row < GRID_HEIGHT and 0 <= grid_col < GRID_WIDTH):
                    is_valid_spot = False
                    break
                # Check overlap with existing grid content
                if grid[grid_row][grid_col] is not None:
                    is_valid_spot = False
                    break

            if is_valid_spot:
                return (r, c) # Found the first valid spot

    return None # No valid spot found

# --- Helper function to find multiple valid placement positions for a piece ---
def find_multiple_valid_positions(grid: List[List[Optional[PieceId]]], piece_id: PieceId, rot_idx: int, max_results: int = 5) -> List[Tuple[int, int]]:
    """
    Searches for multiple valid positions (top-left corner of bounding box) for a piece
    in the given grid.
    Returns a list of up to `max_results` valid spots found.
    """
    p_type, p_shape_idx, p_copy_idx = piece_id
    try:
        shape_coords_normalized: List[Tuple[int, int]] = ALL_ROTATED_SHAPES[(p_type, p_shape_idx)][rot_idx]
    except (IndexError, KeyError):
         return [] # Invalid piece or rotation

    valid_positions: List[Tuple[int, int]] = []

    # Iterate through all possible top-left corners
    for r in range(-MAX_PIECE_HEIGHT + 1, GRID_HEIGHT):
        for c in range(-MAX_PIECE_WIDTH + 1, GRID_WIDTH):
            is_valid_spot: bool = True
            for dr, dc in shape_coords_normalized:
                grid_row: int = r + dr
                grid_col: int = c + dc
                if not (0 <= grid_row < GRID_HEIGHT and 0 <= grid_col < GRID_WIDTH) or grid[grid_row][grid_col] is not None:
                    is_valid_spot = False
                    break

            if is_valid_spot:
                valid_positions.append((r, c))
                if len(valid_positions) >= max_results:
                    return valid_positions # Return early if enough found

    return valid_positions # Return all found if less than max_results

# --- 3. 实现适应度函数 (使用新的放置逻辑) ---

def calculate_score_from_individual(individual: List[Gene]) -> Tuple[float,]:
    """
    Calculates the score for an individual (list of proposed placements)
    by attempting to place them using STRICT placement and scoring the resulting grid.
    """
    final_grid, final_placed_info = get_grid_and_info_from_individual(individual)

    # Calculate Total Score based on the final grid state
    total_score: int = calculate_total_score_from_placement(final_grid, final_placed_info)

    return float(total_score), # DEAP fitness functions must return a tuple of floats

# --- 4. 配置 DEAP 工具箱 (更新所有操作) ---

toolbox = base.Toolbox()

# --- Parameter Management Dictionaries ---
GA_PARAMS: Dict[str, Any] = {
    "POPULATION_SIZE": 800,
    "CXPB": 0.7,
    "MUTPB": 0.3,
    "MIN_INITIAL_PIECES": 25,
    "MAX_INITIAL_PIECES": TOTAL_PIECES_COUNT,
    # GA Mutation Probabilities (adjusted for strict placement - increased swap order)
    "PROB_GENE_MUTATE_POS": 0.05, # Probability to mutate position per gene (lower, rely on LS)
    "PROB_GENE_MUTATE_ROT": 0.1, # Probability to mutate rotation per gene
    "PROB_ADD_PIECE": 0.03,    # Probability to add a new piece placement
    "PROB_REMOVE_PIECE": 0.03, # Probability to remove a piece placement
    "PROB_SWAP_ORDER": 0.3,    # Probability to swap two piece placements (increased)
    "PROB_REPLACE_PIECE": 0.03, # Probability to replace a piece placement
    "POS_MUTATION_RANGE": 3,   # Max delta for position mutation (simple perturbation)
}

LS_PARAMS: Dict[str, Any] = {
    # Local Search parameters for periodic runs
    "LS_FREQ_START": 200, # Run LS every LS_FREQ generations (Increased frequency)
    "LS_FREQ_END": 50, # Frequency can increase over time
    "LS_FREQ_DECAY_GENS": 2000, # Generations over which frequency decays

    "NUM_ELITES_FOR_LS": 10, # Number of top individuals to apply LS to (Increased number)

    "LS_ITER_START": 500, # Number of iterations for each LS call (Adjusted)
    "LS_ITER_END": 2000, # Iterations can increase over time
    "LS_ITER_GROWTH_GENS": 2000, # Generations over which iterations grow

    "LS_INITIAL_TEMP": 100.0,
    "LS_COOLING_RATE": 0.998,

    # LS Neighbor Probabilities (Sum should be <= 1.0) - Adjusted for strict placement
    # Added 'find_random'
    "LS_NEIGHBOR_PROBS": {
        'perturb': 0.3,         # Simple position/rotation/swap perturbation
        'find_first': 0.2,      # Find first valid position for an existing gene
        'find_random': 0.2,     # Find multiple valid positions and pick one randomly
        'add_unplaced': 0.2,    # Add an unplaced piece at its first valid position
        'remove_placed': 0.1,   # Remove a successfully placed piece
    },
    # Parameters for the 'perturb' neighbor generator
    "LS_PERTURB_PARAMS": {
        'prob_perturb_pos': 0.4, # Probability *within* perturb to do pos perturb
        'prob_change_rot': 0.3,  # Probability *within* perturb to do rot change
        'prob_swap_order': 0.3,  # Probability *within* perturb to do swap order
        'pos_perturb_range': 2   # Max delta for position perturbation
    },
    "LS_FIND_RANDOM_MAX_RESULTS": 10 # Max valid positions to find for 'find_random' neighbor
}


# Helper to generate a random gene (piece_id, rot_idx, r, c)
# This is for GA initialization and simple mutations, uses random pos/rot
def generate_random_gene_with_pos_simple(piece_id: PieceId) -> Gene:
     p_type, p_shape_idx, p_copy_idx = piece_id
     possible_rotations: List[List[Tuple[int, int]]] = ALL_ROTATED_SHAPES[(p_type, p_shape_idx)]
     rot_idx: int = random.randrange(len(possible_rotations))
     # Generate a random (r, c) within the extended range
     # Range: [-max_h + 1, GRID_HEIGHT - 1] for row, [-max_w + 1, GRID_WIDTH - 1] for col
     rand_r: int = random.randint(-MAX_PIECE_HEIGHT + 1, GRID_HEIGHT - 1)
     rand_c: int = random.randint(-MAX_PIECE_WIDTH + 1, GRID_WIDTH - 1)
     return (piece_id, rot_idx, rand_r, rand_c)


# Individual initializer: Creates a list of genes (piece_id, rot_idx, r, c)
# MODIFIED: Use find_first_valid_position on an empty grid for initial placement
def init_individual_placement(ind_cls: type, piece_inventory_list: List[PieceId], min_pieces: int, max_pieces: int) -> List[Gene]:
    """Initializes an individual with a random number of unique pieces, random rotations, and valid initial positions."""
    num_pieces: int = random.randint(min_pieces, max_pieces)
    available_pieces: List[PieceId] = list(piece_inventory_list) # Copy the list
    random.shuffle(available_pieces) # Shuffle to pick random unique pieces

    individual: List[Gene] = ind_cls()
    used_piece_ids: Set[PieceId] = set()

    # Create an empty grid to find initial valid positions
    empty_grid: List[List[Optional[PieceId]]] = [[None for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]

    for piece_id in available_pieces:
        if len(individual) >= num_pieces:
            break
        if piece_id in used_piece_ids: # Should not happen with available_pieces list, but safety
             continue

        p_type, p_shape_idx, p_copy_idx = piece_id
        possible_rotations: List[List[Tuple[int, int]]] = ALL_ROTATED_SHAPES[(p_type, p_shape_idx)]
        if not possible_rotations: continue # Skip if no valid rotations

        # Try finding a valid position for a random rotation in the empty grid
        # Shuffle rotations to find a random valid one first
        shuffled_rot_indices = list(range(len(possible_rotations)))
        random.shuffle(shuffled_rot_indices)

        found_valid_pos: Optional[Tuple[int, int]] = None
        chosen_rot_idx: int = -1

        for rot_idx in shuffled_rot_indices:
             pos = find_first_valid_position(empty_grid, piece_id, rot_idx)
             if pos is not None:
                  found_valid_pos = pos
                  chosen_rot_idx = rot_idx
                  break # Found a valid position for this rotation

        if found_valid_pos is not None:
            # Create the gene with the found valid position
            gene: Gene = (piece_id, chosen_rot_idx, found_valid_pos[0], found_valid_pos[1])
            individual.append(gene)
            used_piece_ids.add(piece_id)
        # else:
            # print(f"Warning: Could not find initial valid position for piece {piece_id} in empty grid.")


    return individual


# Register the individual and population creation
toolbox.register("individual", init_individual_placement, creator.Individual, PIECE_INVENTORY, GA_PARAMS["MIN_INITIAL_PIECES"], GA_PARAMS["MAX_INITIAL_PIECES"])
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register the evaluation function (using the new placement logic)
toolbox.register("evaluate", calculate_score_from_individual)

# Register the genetic operators (crossover adapted)
def custom_crossover_placement(ind1: List[Gene], ind2: List[Gene]) -> Tuple[List[Gene], List[Gene]]:
    """Applies crossover to individuals with placement info."""
    size1: int = len(ind1)
    size2: int = len(ind2)
    if size1 < 1 or size2 < 1:
        return ind1, ind2

    # Choose crossover points
    cxpoint1: int = random.randint(0, size1)
    cxpoint2: int = random.randint(0, size2)

    # Create child gene lists by slicing
    child1_genes: List[Gene] = ind1[:cxpoint1] + ind2[cxpoint2:]
    child2_genes: List[Gene] = ind2[:cxpoint2] + ind1[cxpoint1:]

    # Helper to remove duplicate piece_ids, keeping the first occurrence
    def remove_duplicate_piece_ids(gene_list: List[Gene]) -> List[Gene]:
        seen_ids: Set[PieceId] = set()
        new_gene_list: List[Gene] = []
        for gene in gene_list:
            piece_id: PieceId = gene[0] # piece_id is the first element of the gene tuple
            if piece_id not in seen_ids:
                new_gene_list.append(gene)
                seen_ids.add(piece_id)
        return new_gene_list

    # Create new individuals from the combined and cleaned gene lists
    child1: List[Gene] = creator.Individual(remove_duplicate_piece_ids(child1_genes))
    child2: List[Gene] = creator.Individual(remove_duplicate_piece_ids(child2_genes))

    return child1, child2

toolbox.register("mate", custom_crossover_placement)

# Mutation: Custom mutation operators (for new representation)
# Keep simple random mutations for GA, rely on LS for intelligent placement adjustments
# MODIFIED: Adjusted probabilities, simplified add/replace to use simple random pos
def custom_mutation_placement(individual: List[Gene], ga_params: Dict[str, Any]) -> Tuple[List[Gene],]:
    """Applies mutation to an individual (list of placement genes)."""

    prob_gene_mutate_pos = ga_params["PROB_GENE_MUTATE_POS"]
    prob_gene_mutate_rot = ga_params["PROB_GENE_MUTATE_ROT"]
    prob_add = ga_params["PROB_ADD_PIECE"]
    prob_remove = ga_params["PROB_REMOVE_PIECE"]
    prob_swap_order = ga_params["PROB_SWAP_ORDER"]
    prob_replace_piece = ga_params["PROB_REPLACE_PIECE"]
    pos_mutation_range = ga_params["POS_MUTATION_RANGE"]
    min_pieces = ga_params["MIN_INITIAL_PIECES"] # Use min_pieces from params

    # Mutation 1: Change position (r, c) of existing genes (simple random perturbation)
    for i in range(len(individual)):
        if random.random() < prob_gene_mutate_pos: # Probability per gene
            piece_id, rot_idx, r, c = individual[i]
            # Mutate position by a small random offset
            new_r: int = r + random.randint(-pos_mutation_range, pos_mutation_range)
            new_c: int = c + random.randint(-pos_mutation_range, pos_mutation_range)
            individual[i] = (piece_id, rot_idx, new_r, new_c)

    # Mutation 2: Change rotation of existing genes
    for i in range(len(individual)):
        if random.random() < prob_gene_mutate_rot: # Probability per gene
            piece_id, old_rot_idx, r, c = individual[i]
            p_type, p_shape_idx, p_copy_idx = piece_id
            possible_rotations: List[List[Tuple[int, int]]] = ALL_ROTATED_SHAPES[(p_type, p_shape_idx)]
            if len(possible_rotations) > 1: # Only mutate if there's more than one rotation
                new_rot_idx: int = random.randrange(len(possible_rotations))
                individual[i] = (piece_id, new_rot_idx, r, c)

    # Mutation 3: Add a new piece placement (random pos/rot)
    if random.random() < prob_add:
        placed_ids: Set[PieceId] = {gene[0] for gene in individual}
        available_to_add: List[PieceId] = [pid for pid in PIECE_INVENTORY if pid not in placed_ids]
        if available_to_add:
            piece_to_add: PieceId = random.choice(available_to_add)
            # Use the simple random gene generation for GA mutation
            new_gene: Gene = generate_random_gene_with_pos_simple(piece_to_add)

            insert_index: int = random.randint(0, len(individual)) # Allow inserting at end
            individual.insert(insert_index, new_gene)

    # Mutation 4: Remove a piece placement
    if random.random() < prob_remove and len(individual) > min_pieces: # Don't remove if already at min size
        remove_index: int = random.randrange(len(individual))
        individual.pop(remove_index)

    # Mutation 5: Swap order of two genes (changes placement priority)
    if len(individual) > 1 and random.random() < prob_swap_order:
         swap_idx1, swap_idx2 = random.sample(range(len(individual)), 2)
         individual[swap_idx1], individual[swap_idx2] = individual[swap_idx2], individual[swap_idx1]

    # Mutation 6: Replace a piece placement with an unused one (random pos/rot)
    if len(individual) > 0 and random.random() < prob_replace_piece:
        placed_ids: Set[PieceId] = {gene[0] for gene in individual}
        available_to_add: List[PieceId] = [pid for pid in PIECE_INVENTORY if pid not in placed_ids]
        if available_to_add:
            replace_index: int = random.randrange(len(individual))
            piece_to_add_id: PieceId = random.choice(available_to_add)
            # Use the simple random gene generation for GA mutation
            new_gene: Gene = generate_random_gene_with_pos_simple(piece_to_add_id)

            individual[replace_index] = new_gene # Replace the gene

    return individual,

# Register mutation with GA_PARAMS
toolbox.register("mutate", custom_mutation_placement, ga_params=GA_PARAMS)

toolbox.register("select", tools.selTournament, tournsize=3)


# --- 5. 实现局部搜索 (操作个体列表 - 针对严格放置) ---

# Helper to build grid from a prefix of genes
def build_grid_from_gene_prefix(individual: List[Gene], prefix_length: int) -> List[List[Optional[PieceId]]]:
    """Builds the grid state after placing the first `prefix_length` genes strictly."""
    grid: List[List[Optional[PieceId]]] = [[None for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
    used_inventory_ids: Set[PieceId] = set()

    for i in range(min(prefix_length, len(individual))):
        gene = individual[i]
        piece_id, rot_idx, start_r, start_c = gene

        if piece_id in used_inventory_ids:
            continue

        try:
            shape_coords_normalized: List[Tuple[int, int]] = ALL_ROTATED_SHAPES[piece_id[0], piece_id[1]][rot_idx]
        except (IndexError, KeyError):
             continue

        is_valid_spot: bool = True
        for dr, dc in shape_coords_normalized:
            grid_row: int = start_r + dr
            grid_col: int = start_c + dc
            if not (0 <= grid_row < GRID_HEIGHT and 0 <= grid_col < GRID_WIDTH) or grid[grid_row][grid_col] is not None:
                is_valid_spot = False
                break

        if is_valid_spot:
            for dr, dc in shape_coords_normalized: # Iterate over shape coords again to mark grid
                 grid[start_r + dr][start_c + dc] = piece_id # Mark grid
            used_inventory_ids.add(piece_id)

    return grid

# LS Neighbor Generation 1: Perturb Position, Rotation, or Swap Order (Strict-aware)
def generate_ls_perturb_neighbor_strict(individual: List[Gene], perturb_params: Dict[str, Any]) -> List[Gene]:
    """Generates a neighbor by perturbing positions, rotations, or swapping order."""
    neighbor: List[Gene] = creator.Individual(copy.deepcopy(individual)) # Start with a copy

    if not neighbor: return neighbor # Handle empty individual

    prob_perturb_pos = perturb_params['prob_perturb_pos']
    prob_change_rot = perturb_params['prob_change_rot']
    prob_swap_order = perturb_params['prob_swap_order']
    pos_perturb_range = perturb_params['pos_perturb_range']

    # Decide which type of perturbation to apply to *one* randomly chosen gene or pair
    # Normalize probabilities if they sum to > 1
    total_prob: float = prob_perturb_pos + prob_change_rot + prob_swap_order
    weights: List[float] = [prob_perturb_pos, prob_change_rot, prob_swap_order, max(0.0, 1.0 - total_prob)] # Ensure weights sum to 1

    perturb_type: str = random.choices(
        ['pos', 'rot', 'swap', 'none'],
        weights=weights,
        k=1
    )[0]

    if perturb_type == 'pos':
        # Perturb position of one random gene
        idx: int = random.randrange(len(neighbor))
        piece_id, rot_idx, r, c = neighbor[idx]
        new_r: int = r + random.randint(-pos_perturb_range, pos_perturb_range)
        new_c: int = c + random.randint(-pos_perturb_range, pos_perturb_range)
        neighbor[idx] = (piece_id, rot_idx, new_r, new_c)

    elif perturb_type == 'rot':
        # Change rotation of one random gene
        idx: int = random.randrange(len(neighbor))
        piece_id, old_rot_idx, r, c = neighbor[idx]
        p_type, p_shape_idx, p_copy_idx = piece_id
        possible_rotations: List[List[Tuple[int, int]]] = ALL_ROTATED_SHAPES[(p_type, p_shape_idx)]
        if len(possible_rotations) > 1: # Only mutate if there's more than one rotation
            new_rot_idx: int = random.randrange(len(possible_rotations))
            neighbor[idx] = (piece_id, new_rot_idx, r, c)

    elif perturb_type == 'swap':
        # Swap order of two random genes
        if len(neighbor) > 1:
             swap_idx1, swap_idx2 = random.sample(range(len(neighbor)), 2)
             neighbor[swap_idx1], neighbor[swap_idx2] = neighbor[swap_idx2], neighbor[swap_idx1]

    # 'none' type does nothing, returns the copy

    return neighbor

# LS Neighbor Generation 2: Find First Valid Position for a Random Gene
def generate_ls_find_first_valid_pos_neighbor(individual: List[Gene]) -> List[Gene]:
    """
    Generates a neighbor by selecting a random gene and updating its (r, c)
    to the first valid position found when placed in the grid state built
    by the genes *before* it in the list.
    """
    if not individual: return creator.Individual([]) # Handle empty individual

    neighbor: List[Gene] = creator.Individual(copy.deepcopy(individual))

    # Select a random index to attempt to fix its position
    idx_to_fix: int = random.randrange(len(neighbor))

    # Get the gene to fix and the genes before it
    gene_to_fix: Gene = neighbor[idx_to_fix]
    piece_id, rot_idx, old_r, old_c = gene_to_fix
    genes_before: List[Gene] = neighbor[:idx_to_fix]

    # Build the grid state based on the genes before the selected one
    grid_before: List[List[Optional[PieceId]]] = build_grid_from_gene_prefix(genes_before, len(genes_before))

    # Find the first valid position for the selected piece in this grid state
    new_pos: Optional[Tuple[int, int]] = find_first_valid_position(grid_before, piece_id, rot_idx)

    if new_pos is not None:
        # Update the gene with the found valid position
        neighbor[idx_to_fix] = (piece_id, rot_idx, new_pos[0], new_pos[1])
        # print(f"  LS: Found new valid pos {new_pos} for piece {piece_id} at index {idx_to_fix}")
    # else:
        # print(f"  LS: Could not find a valid pos for piece {piece_id} at index {idx_to_fix}. Gene remains unchanged.")

    return neighbor

# LS Neighbor Generation 2.5 (New): Find Random Valid Position for a Random Gene
def generate_ls_find_random_valid_pos_neighbor(individual: List[Gene], max_valid_pos_to_find: int = 10) -> List[Gene]:
    """
    Generates a neighbor by selecting a random gene, finding multiple valid positions
    for it in the grid state built by genes before it, and picking one randomly.
    """
    if not individual: return creator.Individual([]) # Handle empty individual

    neighbor: List[Gene] = creator.Individual(copy.deepcopy(individual))

    # Select a random index to attempt to fix its position
    idx_to_fix: int = random.randrange(len(neighbor))

    # Get the gene to fix and the genes before it
    gene_to_fix: Gene = neighbor[idx_to_fix]
    piece_id, rot_idx, old_r, old_c = gene_to_fix
    genes_before: List[Gene] = neighbor[:idx_to_fix]

    # Build the grid state based on the genes before the selected one
    grid_before: List[List[Optional[PieceId]]] = build_grid_from_gene_prefix(genes_before, len(genes_before))

    # Find multiple valid positions for the selected piece in this grid state
    valid_positions: List[Tuple[int, int]] = find_multiple_valid_positions(grid_before, piece_id, rot_idx, max_results=max_valid_pos_to_find)

    if valid_positions:
        # Pick one valid position randomly
        new_pos: Tuple[int, int] = random.choice(valid_positions)
        # Update the gene with the found valid position
        neighbor[idx_to_fix] = (piece_id, rot_idx, new_pos[0], new_pos[1])
        # print(f"  LS: Found {len(valid_positions)} valid pos for piece {piece_id} at index {idx_to_fix}, chose {new_pos}")
    # else:
        # print(f"  LS: Could not find any valid pos for piece {piece_id} at index {idx_to_fix}. Gene remains unchanged.")

    return neighbor


# LS Neighbor Generation 3: Add an Unplaced Piece (Find First Valid Position)
# This function already finds the first valid position and inserts at a random index.
def generate_ls_add_unplaced_neighbor(individual: List[Gene]) -> List[Gene]:
    """
    Generates a neighbor by selecting an unplaced piece, finding its first
    valid position in the current individual's grid state, and adding it
    to the individual at a random position.
    """
    neighbor: List[Gene] = creator.Individual(copy.deepcopy(individual))

    # Find which pieces are already in the individual
    placed_ids: Set[PieceId] = {gene[0] for gene in neighbor}
    available_to_add: List[PieceId] = [pid for pid in PIECE_INVENTORY if pid not in placed_ids]

    if not available_to_add:
        return neighbor # No pieces to add

    # Select a random piece to add
    piece_to_add_id: PieceId = random.choice(available_to_add)
    p_type, p_shape_idx, p_copy_idx = piece_to_add_id

    # Select a random rotation for the piece
    possible_rotations: List[List[Tuple[int, int]]] = ALL_ROTATED_SHAPES[(p_type, p_shape_idx)]
    if not possible_rotations: return neighbor # Should not happen with defined shapes
    rot_idx: int = random.randrange(len(possible_rotations))

    # Build the grid state based on the current individual's strict placement
    current_grid, _ = get_grid_and_info_from_individual(neighbor)

    # Find the first valid position for the piece to add in the current grid state
    new_pos: Optional[Tuple[int, int]] = find_first_valid_position(current_grid, piece_to_add_id, rot_idx)

    if new_pos is not None:
        # Create the new gene
        new_gene: Gene = (piece_to_add_id, rot_idx, new_pos[0], new_pos[1])
        # Insert the new gene at a random position in the neighbor individual
        insert_index: int = random.randint(0, len(neighbor))
        neighbor.insert(insert_index, new_gene)
        # print(f"  LS: Added piece {piece_to_add_id} at pos {new_pos} (index {insert_index})")
    # else:
        # print(f"  LS: Could not find a valid pos to add piece {piece_to_add_id}.")

    return neighbor

# LS Neighbor Generation 4: Remove a Placed Piece
# This function randomly removes a piece that was successfully placed.
def generate_ls_remove_placed_neighbor(individual: List[Gene], min_pieces: int) -> List[Gene]:
    """
    Generates a neighbor by removing a piece that was successfully placed
    in the current individual's evaluation. Avoids removing if at min size.
    """
    if not individual: return creator.Individual([]) # Handle empty individual

    neighbor: List[Gene] = creator.Individual(copy.deepcopy(individual))

    # Evaluate the current individual to find which pieces were placed
    _, placed_info = get_grid_and_info_from_individual(neighbor)
    placed_ids: Set[PieceId] = set(placed_info.keys())

    # Find indices in the individual corresponding to placed pieces
    # We need the index in the *original* individual list
    placed_indices: List[int] = [i for i, gene in enumerate(individual) if gene[0] in placed_ids]


    if not placed_indices or len(neighbor) <= min_pieces:
        return neighbor # No placed pieces to remove or already at min size

    # Select a random index of a placed piece to remove from the neighbor copy
    # Note: We need to remove from the *neighbor* copy, using the index found in the original
    remove_idx_in_original: int = random.choice(placed_indices)
    removed_piece_id: PieceId = neighbor[remove_idx_in_original][0] # Get piece_id before removing

    # Remove the gene from the neighbor copy
    neighbor.pop(remove_idx_in_original)
    # print(f"  LS: Removed placed piece {removed_piece_id} at index {remove_idx_in_original}")

    return neighbor


# MODIFIED: Added ls_params dictionary
def run_local_search_placement_strict(initial_individual: List[Gene], ls_params: Dict[str, Any]) -> Tuple[List[Gene], float]:
    """
    Applies local search with simulated annealing directly on the individual (placement list).
    Explores neighbors using strict-placement-aware strategies.
    Returns the best individual and score found *within this LS run*.
    """
    # print("\nStarting Local Search on Individual (Placement List) + Simulated Annealing (Strict Placement)...")

    current_individual: List[Gene] = creator.Individual(copy.deepcopy(initial_individual))
    current_score, = toolbox.evaluate(current_individual) # Evaluate the starting individual

    best_individual_ls_run: List[Gene] = creator.Individual(copy.deepcopy(current_individual))
    best_score_ls_run: float = current_score

    # print(f"LS Starting Score: {current_score}")

    max_ls_iterations = ls_params["LS_ITER_CURRENT"] # Use current iterations from params
    initial_temp = ls_params["LS_INITIAL_TEMP"]
    cooling_rate = ls_params["LS_COOLING_RATE"]
    neighbor_probs = ls_params["LS_NEIGHBOR_PROBS"]
    perturb_params = ls_params["LS_PERTURB_PARAMS"]
    max_valid_pos_to_find = ls_params["LS_FIND_RANDOM_MAX_RESULTS"]
    min_pieces = GA_PARAMS["MIN_INITIAL_PIECES"] # Need min_pieces for remove neighbor

    temp: float = initial_temp

    # Ensure neighbor probabilities sum to 1 (already handled in main, but safety)
    prob_keys: List[str] = list(neighbor_probs.keys())
    total_prob: float = sum(neighbor_probs.values())
    if total_prob > 1.0:
        normalized_probs: Dict[str, float] = {k: v / total_prob for k, v in neighbor_probs.items()}
        # print(f"  LS Neighbor Probs Normalized: {normalized_probs}")
    else:
        normalized_probs = neighbor_probs
        # Add 'none' type if probabilities don't sum to 1
        if total_prob < 1.0:
             normalized_probs['none'] = 1.0 - total_prob
             prob_keys.append('none')


    for i in range(max_ls_iterations):
        # Decide which type of neighbor generation to use
        rand_prob: float = random.random()
        neighbor_individual: Optional[List[Gene]] = None

        cumulative_prob: float = 0
        chosen_type: Optional[str] = None

        # Iterate through defined neighbor types
        for n_type in prob_keys:
             if n_type == 'none': continue # Handle 'none' separately if needed, or just let it fall through
             prob = normalized_probs.get(n_type, 0)
             if rand_prob < (cumulative_prob + prob):
                  chosen_type = n_type
                  break
             cumulative_prob += prob

        # Generate neighbor based on chosen type
        if chosen_type == 'perturb':
            neighbor_individual = generate_ls_perturb_neighbor_strict(current_individual, perturb_params)
        elif chosen_type == 'find_first':
            neighbor_individual = generate_ls_find_first_valid_pos_neighbor(current_individual)
        elif chosen_type == 'find_random':
             neighbor_individual = generate_ls_find_random_valid_pos_neighbor(current_individual, max_valid_pos_to_find)
        elif chosen_type == 'add_unplaced':
             neighbor_individual = generate_ls_add_unplaced_neighbor(current_individual)
        elif chosen_type == 'remove_placed':
             neighbor_individual = generate_ls_remove_placed_neighbor(current_individual, min_pieces)
        # If chosen_type is 'none' or no type was chosen due to float precision, neighbor_individual remains None

        # If neighbor generation resulted in an empty individual or failed
        if neighbor_individual is None or len(neighbor_individual) == 0:
             # If the original individual was not empty, this is an issue, but let's handle empty gracefully
             if len(current_individual) > 0:
                  # print("Warning: LS neighbor generation resulted in empty individual or no change.")
                  pass # Continue with current_individual (no change)
             else:
                  continue # Skip if both are empty


        # Evaluate the neighbor state
        next_score, = toolbox.evaluate(neighbor_individual)

        # --- Acceptance Criteria (Simulated Annealing) ---
        # Accept better solutions
        if next_score > current_score:
            current_individual = neighbor_individual
            current_score = next_score
            # Update best found *in this LS run*
            if current_score > best_score_ls_run:
                best_score_ls_run = current_score
                best_individual_ls_run = creator.Individual(copy.deepcopy(current_individual))
                # print(f"LS Iter {i+1}: New best score in run {best_score_ls_run}")
        # Accept worse solutions with a probability
        elif temp > 0:
            delta: float = next_score - current_score # delta is negative
            acceptance_probability: float = math.exp(delta / (temp + 1e-9)) # Add epsilon to temp
            if random.random() < acceptance_probability:
                current_individual = neighbor_individual
                current_score = next_score
                # Optional: print accepted worse moves
                # print(f"LS Iter {i+1}: Accepted worse score {current_score} (delta={delta:.2f}, temp={temp:.4f})")


        # --- Cooling ---
        temp *= cooling_rate

        # Optional: Print progress periodically within LS (less verbose for periodic runs)
        # if (i + 1) % 500 == 0:
        #      print(f"LS Iter {i+1}/{max_ls_iterations}, Current Score: {current_score}, Best Score in run: {best_score_ls_run}, Temp: {temp:.4f}")


    # print(f"Local Search finished after {i+1} iterations.")
    # print(f"Best score found in this LS run: {best_score_ls_run}")

    # Return the best individual and score found *within this specific LS run*
    return best_individual_ls_run, best_score_ls_run

# --- Helper function for parallel LS execution ---
# This function is defined at the top level so it can be pickled
# MODIFIED: Pass ls_params dictionary
def _run_ls_for_map(args: Tuple[List[Gene], Dict[str, Any]]) -> Tuple[List[Gene], float]:
    """Helper to unpack args and run LS for parallel map."""
    individual, ls_params = args
    return run_local_search_placement_strict(individual, ls_params)


# --- Global variables to track the overall best solution found ---
overall_best_score: float = -float('inf') # Initialize with a very low score
overall_best_grid: Optional[List[List[Optional[PieceId]]]] = None
overall_best_placed_info: Optional[Dict[PieceId, Dict[str, Any]]] = None
overall_best_individual: Optional[List[Gene]] = None # Store the individual that produced the best score

# --- Log file configuration ---
LOG_FILE_NAME: str = "best_solution_log_memetic_strict.txt" # Changed log file name

# --- Modular function to update the overall best solution and log it ---
# Added pbar parameter
def update_overall_best(score: float, individual: List[Gene], pbar: Optional[tqdm] = None) -> bool:
    """
    Updates the global overall best solution if the provided score is higher.
    If updated, logs the new best solution details to a file and updates tqdm description.
    Takes the score and the individual (placement list) that produced it.
    Optionally takes a tqdm progress bar object to update its description.
    Returns True if the overall best was updated, False otherwise.
    """
    global overall_best_score, overall_best_grid, overall_best_placed_info, overall_best_individual
    updated: bool = False
    if score > overall_best_score:
        overall_best_score = score
        # Get the grid and placed_info from the individual for storage and logging
        overall_best_grid, overall_best_placed_info = get_grid_and_info_from_individual(individual)
        overall_best_individual = creator.Individual(copy.deepcopy(individual)) # Store the individual itself
        updated = True
        print(f"--- New overall best score found: {overall_best_score} ---") # Print to console immediately

        # --- Update tqdm description immediately ---
        if pbar is not None:
             pbar.set_description(f"GA Progress (Best Score: {overall_best_score})")
             pbar.refresh() # Force immediate redraw


        # --- Log the new best solution ---
        try:
            with open(LOG_FILE_NAME, 'a') as f:
                f.write("="*80 + "\n")
                f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"New Overall Best Score: {overall_best_score}\n")
                f.write(f"Number of pieces successfully placed: {len(overall_best_placed_info) if overall_best_placed_info else 0}\n")
                f.write(f"Individual Length (Proposed Placements): {len(overall_best_individual) if overall_best_individual else 0}\n")

                # --- Log Genotype (Individual) ---
                f.write("\nGenotype (Individual - Proposed Placements):\n")
                if overall_best_individual:
                    # Format as a Python list string for easy copy/paste
                    genotype_str = "[" + ", ".join(str(gene) for gene in overall_best_individual) + "]"
                    f.write(genotype_str + "\n")
                else:
                    f.write("None\n")


                # --- Log Phenotype (Actual Placed Pieces) ---
                f.write("\nPhenotype (Actual Placed Pieces):\n")
                if overall_best_placed_info:
                    # Sort pieces by their top-left corner for consistent output in log
                    final_placed_list = sorted(overall_best_placed_info.values(), key=lambda x: (x['cells'][0][0] if x['cells'] else -1, x['cells'][0][1] if x['cells'] else -1)) # Use first cell as proxy for sorting
                    for details in final_placed_list:
                         piece_id = (details['type'], details['shape_idx'], details['copy_idx'])
                         # Find the proposed start pos from the original gene in the best individual
                         gene_start_pos: Optional[Tuple[int, int]] = None
                         if overall_best_individual:
                             for gene in overall_best_individual:
                                 if gene[0] == piece_id:
                                     gene_start_pos = (gene[2], gene[3])
                                     break
                         f.write(f"  - {PIECE_TYPE_NAMES[details['type']]} Shape {details['shape_idx']} Copy {details['copy_idx']} (ID: {piece_id}) Placed Cells: {details['cells']} (Proposed Start: {gene_start_pos}) Rot Index {details['rotation_idx']}\n")
                else:
                    f.write("None\n")


                f.write("\nGrid Visualization:\n")
                if overall_best_grid and overall_best_placed_info is not None:
                    visual_grid: List[List[str]] = [[' . ' for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
                    piece_char_map: Dict[PieceId, str] = {}
                    char_counter: int = 0
                    chars: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz#@$&*+=-"
                    for piece_id, details in overall_best_placed_info.items():
                         if piece_id not in piece_char_map:
                             piece_char_map[piece_id] = chars[char_counter % len(chars)]
                             char_counter += 1
                         char_to_use: str = piece_char_map[piece_id]
                         for r, c in details['cells']:
                             visual_grid[r][c] = f" {char_to_use} "

                    f.write("   " + "".join([f" {c} " for c in range(GRID_WIDTH)]) + "\n")
                    for r in range(GRID_HEIGHT):
                        f.write(f"{r} |" + "".join(visual_grid[r]) + "\n")
                else:
                    f.write("No grid to visualize.\n")

                f.write("-" * (GRID_WIDTH * 3 + 4) + "\n") # Separator
                f.write("="*80 + "\n\n") # End of entry separator

        except IOError as e:
            print(f"Error writing to log file {LOG_FILE_NAME}: {e}")

    return updated


# --- 6. 运行模因遗传算法 (μ + λ 替换, 周期性LS应用于精英) ---

def main():
    """Runs the memetic algorithm with direct placement representation (Strict Placement)."""
    start_time: float = time.time()
    random.seed(42) # for reproducibility

    # Use parameters from dictionaries
    POPULATION_SIZE = GA_PARAMS["POPULATION_SIZE"]
    CXPB = GA_PARAMS["CXPB"]
    MUTPB = GA_PARAMS["MUTPB"]
    MIN_INITIAL_PIECES = GA_PARAMS["MIN_INITIAL_PIECES"]
    MAX_INITIAL_PIECES = GA_PARAMS["MAX_INITIAL_PIECES"]

    LS_FREQ_START = LS_PARAMS["LS_FREQ_START"]
    LS_FREQ_END = LS_PARAMS["LS_FREQ_END"]
    LS_FREQ_DECAY_GENS = LS_PARAMS["LS_FREQ_DECAY_GENS"]
    NUM_ELITES_FOR_LS = LS_PARAMS["NUM_ELITES_FOR_LS"]
    LS_ITER_START = LS_PARAMS["LS_ITER_START"]
    LS_ITER_END = LS_PARAMS["LS_ITER_END"]
    LS_ITER_GROWTH_GENS = LS_PARAMS["LS_ITER_GROWTH_GENS"]


    print("Starting Memetic Algorithm with Direct Placement Representation (Strict Placement)...")
    print(f"Population Size (μ=λ): {POPULATION_SIZE}")
    print(f"Min/Max Initial Proposed Pieces: {MIN_INITIAL_PIECES}/{MAX_INITIAL_PIECES}")
    print(f"Crossover Probability: {CXPB}, Mutation Probability: {MUTPB}")
    print(f"GA Mutation Probs: {GA_PARAMS['PROB_GENE_MUTATE_POS']=}, {GA_PARAMS['PROB_GENE_MUTATE_ROT']=}, {GA_PARAMS['PROB_ADD_PIECE']=}, {GA_PARAMS['PROB_REMOVE_PIECE']=}, {GA_PARAMS['PROB_SWAP_ORDER']=}, {GA_PARAMS['PROB_REPLACE_PIECE']=}, {GA_PARAMS['POS_MUTATION_RANGE']=}")
    print(f"Local Search Frequency: Starts at {LS_FREQ_START}, ends at {LS_FREQ_END} over {LS_FREQ_DECAY_GENS} gens")
    print(f"Number of Elites for LS: {NUM_ELITES_FOR_LS}")
    print(f"LS Iterations per run: Starts at {LS_ITER_START}, ends at {LS_ITER_END} over {LS_ITER_GROWTH_GENS} gens")
    print(f"LS SA Params: Initial Temp={LS_PARAMS['LS_INITIAL_TEMP']}, Cooling Rate={LS_PARAMS['LS_COOLING_RATE']}")
    print(f"LS Neighbor Probs: {LS_PARAMS['LS_NEIGHBOR_PROBS']}")
    print(f"LS Perturb Params (within 'perturb' neighbor): {LS_PARAMS['LS_PERTURB_PARAMS']}")
    print(f"LS Find Random Max Results: {LS_PARAMS['LS_FIND_RANDOM_MAX_RESULTS']}")
    print(f"Logging best solutions to: {LOG_FILE_NAME}")


    # --- Define known good solutions as chromosomes (using the NEW representation) ---
    # This is the hypothetical 212 solution individual from previous step.
    # Its score with the new placement logic might not be 212.
    # You can replace this with actual individuals derived from known high-score layouts
    # if you can reverse-engineer their piece_id, rot_idx, and a suitable (r,c) anchor.
    # Note: These are just *proposed* positions. Strict placement will determine actual.
    # The initial positions here are just examples, the init_individual_placement will find valid ones.
    # Let's keep this as an example structure, but rely on the improved init for actual starting individuals.
    # solution_individual_hypothetical = creator.Individual([
    #     ((0, 1, 0), 0, 0, 0), # Special Shape 1 Copy 0, Rot 0, Pos (0,0)
    #     ((0, 1, 1), 0, 0, 3), # Special Shape 1 Copy 1, Rot 0, Pos (0,3)
    #     ((0, 1, 2), 1, 0, 6), # Special Shape 1 Copy 2, Rot 1 (90 deg), Pos (0,6) - assuming it fits rotated
    #     ((1, 0, 0), 1, 1, 0), # Land Shape 0 Copy 0, Rot 1 (90 deg), Pos (1,0)
    #     ((1, 1, 0), 0, 1, 2), # Land Shape 1 Copy 0, Rot 0, Pos (1,2)
    #     ((1, 3, 0), 2, 1, 5), # Land Shape 3 Copy 0, Rot 2 (180 deg), Pos (1,5)
    #     ((2, 0, 0), 1, 2, 0), # Large Shape 0 Copy 0, Rot 1 (90 deg), Pos (2,0)
    #     ((2, 0, 1), 3, 2, 3), # Large Shape 0 Copy 1, Rot 3 (270 deg), Pos (2,3)
    #     ((2, 3, 0), 2, 3, 0), # Large Shape 3 Copy 0, Rot 2 (180 deg), Pos (3,0)
    #     ((2, 3, 1), 0, 3, 4), # Large Shape 3 Copy 1, Rot 0, Pos (3,4)
    #     ((3, 0, 0), 0, 4, 2), # Magic1 Shape 0 Copy 0, Rot 0, Pos (4,2)
    #     ((3, 0, 1), 0, 4, 5), # Magic1 Shape 0 Copy 1, Rot 0, Pos (4,5)
    #     ((4, 0, 0), 0, 5, 1), # Magic2 Shape 0 Copy 0, Rot 0, Pos (5,1)
    #     ((4, 0, 1), 0, 5, 4)  # Magic2 Shape 0 Copy 1, Rot 0, Pos (5,4)
    # ])

    # Decide whether to include seeded solutions
    INCLUDE_SEEDED = False # Set to True to include the hypothetical solution
    seeded_solutions_individuals: List[List[Gene]] = []
    if INCLUDE_SEEDED:
        seeded_solutions_individuals = [
            # solution_individual_hypothetical,
            # Add other known solutions here if you can convert them to the new format
        ]
    num_seeded: int = len(seeded_solutions_individuals)


    # Create initial population (mostly random)
    population: List[List[Gene]] = toolbox.population(n=POPULATION_SIZE - num_seeded) # Make space for seeded solutions

    # Add the known good solutions to the population
    population.extend(seeded_solutions_individuals)

    # Shuffle the population to mix seeded and random individuals
    random.shuffle(population)


    # Evaluate the initial population using the parallel map
    print("\nEvaluating initial population...")
    fitnesses: List[Tuple[float,]] = list(toolbox.map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    print(f"Evaluated {len(population)} individuals in initial population (including {num_seeded} seeded).")
    initial_scores: List[float] = [ind.fitness.values[0] for ind in population]
    print(f"Initial min/avg/max fitness: {np.min(initial_scores)}/{np.mean(initial_scores):.2f}/{np.max(initial_scores)}")

    hof = tools.HallOfFame(1) # Track only the single best individual overall
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("avg", np.mean)
    stats.register("min", np.min)

    # Update Hall of Fame and stats with the initial population
    hof.update(population)
    record = stats.compile(population)
    logbook = tools.Logbook()
    logbook.header = ['gen', 'evals'] + stats.fields
    # Record initial state (gen 0)
    logbook.record(gen=0, evals=len(population), **record)


    # Initialize overall best solution found so far using the initial GA best
    initial_ga_best_individual: List[Gene] = hof[0]
    initial_score, = initial_ga_best_individual.fitness.values
    # Use the modular update function - pass None for pbar initially
    update_overall_best(initial_score, initial_ga_best_individual, pbar=None)

    print(f"Initial Overall Best Score: {overall_best_score}") # Now this uses the globally tracked best

    print(f"\nStarting Infinite Memetic Algorithm (Strict Placement)...")
    print("Press Ctrl+C to stop.")

    try:
        # Use tqdm for the infinite loop
        # Pass the initial best score to tqdm description
        pbar = tqdm(itertools.count(1), desc=f"GA Progress (Best Score: {overall_best_score})", unit="gen")
        for gen in pbar:

            # --- Dynamic LS Parameters ---
            # Linear decay for frequency (faster LS)
            ls_frequency = max(LS_FREQ_END, LS_FREQ_START - (LS_FREQ_START - LS_FREQ_END) * gen / LS_FREQ_DECAY_GENS)
            # Linear growth for iterations (deeper LS)
            ls_iterations = min(LS_ITER_END, LS_ITER_START + (LS_ITER_END - LS_ITER_START) * gen / LS_ITER_GROWTH_GENS)

            # Update LS_PARAMS for this generation
            current_ls_params = copy.deepcopy(LS_PARAMS) # Avoid modifying the global dict directly
            current_ls_params["LS_ITER_CURRENT"] = int(ls_iterations) # Pass current iterations to LS function

            # --- GA Generation Steps (μ + λ) ---

            # Select the next generation individuals (parents for reproduction)
            # We select POPULATION_SIZE parents to produce POPULATION_SIZE offspring (λ = μ)
            parents: List[List[Gene]] = toolbox.select(population, POPULATION_SIZE)
            # Clone the selected individuals to create the offspring pool
            offspring: List[List[Gene]] = list(map(toolbox.clone, parents))

            # Apply crossover and mutation on the offspring
            # Crossover is applied to pairs of offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    toolbox.mate(child1, child2)
                    # Invalidate fitness values for children that underwent crossover
                    del child1.fitness.values
                    del child2.fitness.values
            # Mutation is applied to each offspring
            for mutant in offspring:
                if random.random() < MUTPB:
                    # Pass GA_PARAMS to mutation
                    toolbox.mutate(mutant, ga_params=GA_PARAMS)
                    # Invalidate fitness value for mutant
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness (only offspring that were modified)
            invalid_ind: List[List[Gene]] = [ind for ind in offspring if not ind.fitness.valid]
            # Use parallel map for evaluation
            fitnesses: List[Tuple[float,]] = list(toolbox.map(toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # --- Periodic Local Search (on a subset of elites) ---
            ls_evaluated_count: int = 0
            # Use the dynamic frequency
            if gen % int(ls_frequency) == 0:
                # Print inside the tqdm context to avoid interfering with the bar
                pbar.write(f"\n--- Generation {gen}: Running Local Search on {NUM_ELITES_FOR_LS} elite individuals (Iters: {current_ls_params['LS_ITER_CURRENT']}, Freq: {int(ls_frequency)}) ---")
                # Select the top N individuals from the current population
                elite_individuals: List[List[Gene]] = tools.selBest(population, NUM_ELITES_FOR_LS)

                # Prepare arguments for parallel LS calls
                # Each tuple contains (individual, ls_params)
                ls_args: List[Tuple[List[Gene], Dict[str, Any]]] = [(toolbox.clone(elite_ind), current_ls_params) for elite_ind in elite_individuals]

                # Run LS on elites in parallel using the top-level helper function
                ls_results: List[Tuple[List[Gene], float]] = list(toolbox.map(_run_ls_for_map, ls_args))

                improved_elites: List[List[Gene]] = [] # List to store individuals improved by LS

                for i, (ls_best_individual_run, ls_best_score_run) in enumerate(ls_results):
                    original_elite_score: float = elite_individuals[i].fitness.values[0] # Get original score from the elite list
                    ls_evaluated_count += current_ls_params["LS_ITER_CURRENT"] # Count LS evaluations

                    # Check if LS found a better individual than the starting elite
                    if ls_best_score_run > original_elite_score:
                         # Assign the new fitness value
                         ls_best_individual_run.fitness.values = (ls_best_score_run,)
                         improved_elites.append(ls_best_individual_run)
                         # Print inside tqdm context
                         pbar.write(f"  Elite {i+1} improved from {original_elite_score:.2f} to {ls_best_score_run:.2f}")
                    # else:
                         # Print inside tqdm context
                         # pbar.write(f"  Elite {i+1} LS finished. Best score in run: {ls_best_score_run:.2f}. No improvement over starting individual ({original_elite_score:.2f})")

                # Add improved elites to the offspring pool for the replacement step
                # This is the Lamarckian step: the improved genotype is added.
                offspring.extend(improved_elites)
                # pbar.write(f"Added {len(improved_elites)} improved elites to offspring pool.")


            # --- Replacement Step (μ + λ) ---
            # Combine parents and offspring (which might now include improved elites)
            combined_pool: List[List[Gene]] = population + offspring

            # Select the best POPULATION_SIZE individuals from the combined pool for the next generation
            population[:] = tools.selBest(combined_pool, POPULATION_SIZE)


            # Update the Hall of Fame with the new population
            hof.update(population)

            # Update the statistics with the new population
            record = stats.compile(population)
            # evals is the number of individuals evaluated in this generation (offspring + LS evals)
            logbook.record(gen=gen, evals=len(invalid_ind) + ls_evaluated_count, **record)

            # --- Check and Update overall best score ---
            # The Hall of Fame is already updated, so hof[0] is the best in the current population (or the LS improved one if it was better)
            current_best_individual: List[Gene] = hof[0]
            current_best_score, = current_best_individual.fitness.values
            # update_overall_best will handle logging and updating tqdm description if it's a new best
            update_overall_best(current_best_score, current_best_individual, pbar=pbar)

            # The tqdm description is now updated inside update_overall_best if needed
            # pbar.set_description(f"GA Progress (Best Score: {overall_best_score})")
            # --- End of every-generation update ---


    except KeyboardInterrupt:
        print("\nCtrl+C detected. Stopping.")
        # Close the tqdm bar cleanly
        pbar.close()


    # --- Final Output after stopping ---
    print("\n--- Final Best Solution Found ---")
    # These variables now hold the consistent best solution found across GA and LS
    print(f"Max Score: {overall_best_score}")
    print(f"Number of pieces successfully placed: {len(overall_best_placed_info) if overall_best_placed_info else 0}")
    print(f"Individual Length (Proposed Placements): {len(overall_best_individual) if overall_best_individual else 0}")

    # --- Output Genotype ---
    print("\nGenotype (Individual - Proposed Placements):")
    if overall_best_individual:
        # Format as a Python list string for easy copy/paste
        genotype_str: str = "[" + ", ".join(str(gene) for gene in overall_best_individual) + "]"
        print(genotype_str)
    else:
        print("None")

    # --- Output Phenotype ---
    print("\nPhenotype (Actual Placed Pieces):")
    if overall_best_placed_info:
        # Sort pieces by their top-left corner for consistent output in log
        final_placed_list = sorted(overall_best_placed_info.values(), key=lambda x: (x['cells'][0][0] if x['cells'] else -1, x['cells'][0][1] if x['cells'] else -1)) # Use first cell as proxy for sorting
        for details in final_placed_list:
             piece_id = (details['type'], details['shape_idx'], details['copy_idx'])
             # Find the proposed start pos from the original gene in the best individual
             gene_start_pos: Optional[Tuple[int, int]] = None
             if overall_best_individual:
                 for gene in overall_best_individual:
                     if gene[0] == piece_id:
                                 gene_start_pos = (gene[2], gene[3])
                                 break
             print(f"  - {PIECE_TYPE_NAMES[details['type']]} Shape {details['shape_idx']} Copy {details['copy_idx']} (ID: {piece_id}) Placed Cells: {details['cells']} (Proposed Start: {gene_start_pos}) Rot Index {details['rotation_idx']}\n")
    else:
        print("None")


    print("\nGrid Visualization:")
    if overall_best_grid and overall_best_placed_info is not None:
        visual_grid: List[List[str]] = [[' . ' for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        piece_char_map: Dict[PieceId, str] = {}
        char_counter: int = 0
        chars: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz#@$&*+=-"
        for piece_id, details in overall_best_placed_info.items():
             if piece_id not in piece_char_map:
                 piece_char_map[piece_id] = chars[char_counter % len(chars)]
                 char_counter += 1
             char_to_use: str = piece_char_map[piece_id]
             for r, c in details['cells']:
                 visual_grid[r][c] = f" {char_to_use} "

        print("   " + "".join([f" {c} " for c in range(GRID_WIDTH)]))
        for r in range(GRID_HEIGHT):
            print(f"{r} |" + "".join(visual_grid[r]))
    else:
        print("No grid to visualize.")


    end_time: float = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")

    # Shutdown the parallel pool
    # pool.shutdown() # Shutdown is handled in the __main__ block
    # print("Parallel pool shut down.")


if __name__ == "__main__":
    # On some systems (like Windows), the ProcessPoolExecutor must be created
    # within the if __name__ == "__main__": block.
    # Moving the pool creation and toolbox registration here for robustness.
    # This requires moving the toolbox definition outside main or passing it.
    # Let's keep toolbox global for simplicity with DEAP.
    # The pool creation itself needs to be here.
    # The registration can stay outside if toolbox is global.

    # Parallelization setup
    MAX_WORKERS: int = os.cpu_count() if os.cpu_count() else 1
    print(f"Using {MAX_WORKERS} CPU cores for parallelization.")
    # Create the pool within the main block
    pool = concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS)
    # Register the map function with the pool
    toolbox.register("map", pool.map)

    # Run the main function
    main()

    # Shutdown the pool after main finishes
    pool.shutdown()
    print("Parallel pool shut down.")
