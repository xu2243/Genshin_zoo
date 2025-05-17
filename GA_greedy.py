import random
import numpy as np
from deap import base, creator, tools, algorithms
import copy
import math # Needed for simulated annealing
import time # To measure execution time
import itertools # For infinite loop
from tqdm import tqdm # For progress bar
import os # Needed for file operations
import datetime # Needed for timestamp in log

# --- 1. 定义常量和拼图形状 (与之前相同) ---

GRID_HEIGHT = 6
GRID_WIDTH = 7

# 基础形状定义 (相对于 (0,0) 的坐标偏移)
SPECIAL_SHAPES_BASE = [
    [(0, 0), (0, 1), (0, 2)],  # 1x3
    [(0, 0), (0, 1), (0, 2), (0, 3)], # 1x4
    [(0, 0), (0, 1), (1, 0)],  # L shape
    [(0, 0), (0, 1), (1, 1), (1, 2)],  # Z shape
]
SPECIAL_SIZES = [3, 4, 3, 4]

LAND_SHAPES_BASE = [
    [(0, 0), (0, 1), (0, 2)],  # 1x3
    [(0, 0), (0, 1), (1, 0)],  # L shape
    [(0, 0), (0, 1), (1, 0), (1, 1)],  # Square
    [(0, 0), (0, 1), (0, 2), (1, 1)],  # T shape
]
LAND_SIZES = [3, 3, 4, 4]

LARGE_SHAPES_BASE = [
    [(0, 0), (0, 1), (0, 2), (1, 2)],
    [(0, 0), (0, 1), (0, 2), (1, 1)],
    [(0, 1), (0, 2), (1, 0), (1, 1)],
    [(0, 2), (1, 0), (1, 1), (1, 2)],
]
LARGE_SIZES = [4, 4, 4, 4]

MAGIC_SHAPES_BASE = [
    [(0, 0)], # Magic 1 (1x1)
    [(0, 0)], # Magic 2 (1x1)
]
MAGIC_SIZES = [1, 1]

# 拼图库存 (类型, 形状索引, 拷贝索引)
# 类型: 0=Special, 1=Land, 2=Large, 3=Magic1, 4=Magic2
PIECE_INVENTORY = []
for i in range(4): # 4 shapes per type
    for j in range(4): # 4 copies per shape
        PIECE_INVENTORY.append((0, i, j)) # Special
        PIECE_INVENTORY.append((1, i, j)) # Land
        PIECE_INVENTORY.append((2, i, j)) # Large
for j in range(2): # 2 copies per magic type
    PIECE_INVENTORY.append((3, 0, j)) # Magic 1
    PIECE_INVENTORY.append((4, 0, j)) # Magic 2

TOTAL_PIECES_COUNT = len(PIECE_INVENTORY) # 52

PIECE_TYPE_NAMES = {
    0: "Special",
    1: "Land",
    2: "Large",
    3: "Magic1",
    4: "Magic2"
}

PIECE_SHAPES = [SPECIAL_SHAPES_BASE, LAND_SHAPES_BASE, LARGE_SHAPES_BASE, MAGIC_SHAPES_BASE, MAGIC_SHAPES_BASE]
PIECE_SIZES = [SPECIAL_SIZES, LAND_SIZES, LARGE_SIZES, MAGIC_SIZES, MAGIC_SIZES]

def rotate_shape(shape, rotation):
    rotated = []
    for r, c in shape:
        if rotation == 0: rotated.append((r, c))
        elif rotation == 90: rotated.append((c, -r))
        elif rotation == 180: rotated.append((-r, -c))
        elif rotation == 270: rotated.append((-c, r))
        else: raise ValueError("Invalid rotation angle")
    return rotated

def normalize_shape(shape):
    min_r = min(r for r, c in shape)
    min_c = min(c for r, c in shape)
    return sorted([(r - min_r, c - min_c) for r, c in shape])

ALL_ROTATED_SHAPES = {}
for type_idx in range(len(PIECE_SHAPES)):
    for shape_idx, base_shape in enumerate(PIECE_SHAPES[type_idx]):
        unique_rotations = []
        seen_normalized = set()
        for rot_angle in [0, 90, 180, 270]:
            rotated = rotate_shape(base_shape, rot_angle)
            normalized = tuple(normalize_shape(rotated))
            if normalized not in seen_normalized:
                seen_normalized.add(normalized)
                unique_rotations.append(list(normalized))
        ALL_ROTATED_SHAPES[(type_idx, shape_idx)] = unique_rotations

# --- 2. 定义染色体表示 (与之前改进后相同) ---

# Gene: (piece_inventory_id, rotation_index)
# Chromosome: [gene1, gene2, ...]

# DEAP setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# --- Helper function to calculate score for a *given* grid state ---
def calculate_total_score_from_placement(grid, placed_pieces_info):
    """Calculates the total score based on a completed grid and placed pieces info."""
    total_score = 0
    placed_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0} # Count placed pieces by type

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
            adjacent_piece_ids = set()
            for r, c in info['cells']:
                # Check 4 neighbors (up, down, left, right)
                neighbors = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
                for nr, nc in neighbors:
                    if 0 <= nr < GRID_HEIGHT and 0 <= nc < GRID_WIDTH:
                        neighbor_piece_id = grid[nr][nc]
                        # Check if neighbor cell is occupied by a *different* piece that was successfully placed
                        if neighbor_piece_id is not None and neighbor_piece_id != piece_id and neighbor_piece_id in placed_pieces_info:
                            adjacent_piece_ids.add(neighbor_piece_id)
            total_score += len(adjacent_piece_ids) * 2

    # Magic Creature Bonuses
    for piece_id, info in placed_pieces_info.items():
        if info['is_magic1']:
            affected_piece_ids = set()
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
            affected_piece_ids = set()
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

# --- Helper function to place a list of genes into an existing grid using intelligent greedy strategy ---
def place_pieces_into_grid(start_grid, start_placed_info, genes_to_place):
    """
    Attempts to place a list of genes (pieces) into an existing grid and placed_info,
    using the intelligent greedy strategy. Pieces are placed in the order they appear in genes_to_place.
    """
    current_grid = copy.deepcopy(start_grid)
    current_placed_info = copy.deepcopy(start_placed_info)
    used_inventory_ids = set(current_placed_info.keys()) # Pieces already in the grid

    for gene in genes_to_place:
        piece_id, rot_idx = gene
        p_type, p_shape_idx, p_copy_idx = piece_id

        # Skip if this piece instance is already placed
        if piece_id in used_inventory_ids:
            continue

        try:
            shape_coords = ALL_ROTATED_SHAPES[(p_type, p_shape_idx)][rot_idx]
        except IndexError:
             continue # Should not happen with valid genes

        potential_placements = [] # List of (local_score, r, c, piece_cells)

        # Find ALL valid spots and evaluate local score in the *current* grid state
        for r in range(GRID_HEIGHT):
            for c in range(GRID_WIDTH):
                is_valid_spot = True
                current_piece_cells = []
                for dr, dc in shape_coords:
                    grid_row = r + dr
                    grid_col = c + dc

                    # Check bounds
                    if not (0 <= grid_row < GRID_HEIGHT and 0 <= grid_col < GRID_WIDTH):
                        is_valid_spot = False
                        break
                    # Check overlap with already placed pieces in current_grid
                    if current_grid[grid_row][grid_col] is not None:
                        is_valid_spot = False
                        break
                    current_piece_cells.append((grid_row, grid_col))

                if is_valid_spot:
                    # Calculate local score for placing this piece at (r, c)
                    # This score calculation considers interactions with pieces *already* in current_placed_info
                    local_score = 0

                    # 1. Base Score for this piece (used for greedy choice)
                    if p_type in [0, 1]: # Special or Land
                         local_score += 6 if PIECE_SIZES[p_type][p_shape_idx] == 3 else 12
                    elif p_type == 2: # Large
                         local_score += 4
                    # Magic pieces have base score 0

                    # 2. Adjacency Bonus for this piece (if Large) with already placed pieces
                    if p_type == 2: # If current piece is Large
                        adjacent_piece_ids = set()
                        for cell_r, cell_c in current_piece_cells:
                            neighbors = [(cell_r-1, cell_c), (cell_r+1, cell_c), (cell_r, cell_c-1), (cell_r, cell_c+1)]
                            for nr, nc in neighbors:
                                if 0 <= nr < GRID_HEIGHT and 0 <= nc < GRID_WIDTH:
                                    neighbor_piece_id = current_grid[nr][nc]
                                    # Check if neighbor cell is occupied by a *different* piece that was successfully placed
                                    if neighbor_piece_id is not None and neighbor_piece_id in current_placed_info:
                                        adjacent_piece_ids.add(neighbor_piece_id)
                        local_score += len(adjacent_piece_ids) * 2

                    # 3. Magic Bonus contribution *from* this piece *to* already placed pieces
                    if p_type == 3: # If current piece is Magic 1
                         affected_piece_ids = set()
                         for cell_r, cell_c in current_piece_cells: # Magic 1 is 1x1
                             neighbors = [(cell_r-1, c), (cell_r+1, c), (cell_r, c-1), (cell_r, c+1)]
                             for nr, nc in neighbors:
                                 if 0 <= nr < GRID_HEIGHT and 0 <= nc < GRID_WIDTH:
                                     neighbor_piece_id = current_grid[nr][nc]
                                     if neighbor_piece_id is not None and neighbor_piece_id in current_placed_info:
                                         affected_piece_ids.add(neighbor_piece_id)
                         local_score += len(affected_piece_ids) * 3

                    elif p_type == 4: # If current piece is Magic 2
                         affected_piece_ids = set()
                         for cell_r, cell_c in current_piece_cells: # Magic 2 is 1x1
                             neighbors = [(cell_r-1, c-1), (cell_r-1, c), (cell_r-1, c+1),
                                          (cell_r, c-1),             (cell_r, c+1),
                                          (r+1, c-1), (r+1, c), (r+1, c+1)] # Corrected typo here (r+1, c-1)
                             for nr, nc in neighbors:
                                 if 0 <= nr < GRID_HEIGHT and 0 <= nc < GRID_WIDTH:
                                     neighbor_piece_id = current_grid[nr][nc]
                                     if neighbor_piece_id is not None and neighbor_piece_id in current_placed_info:
                                         affected_piece_ids.add(neighbor_piece_id)
                             local_score += len(affected_piece_ids) * 2

                    potential_placements.append((local_score, r, c, current_piece_cells))

        # Choose the best placement based on local score
        if potential_placements:
            best_placement = max(potential_placements, key=lambda item: item[0])
            chosen_local_score, chosen_r, chosen_c, chosen_cells = best_placement

            # Place the piece at the chosen best spot
            for cell_r, cell_c in chosen_cells:
                current_grid[cell_r][cell_c] = piece_id # Mark grid with piece ID

            # Add details for the newly placed piece
            p_type, p_shape_idx, p_copy_idx = piece_id
            current_placed_info[piece_id] = {
                'type': p_type,
                'shape_idx': p_shape_idx,
                'copy_idx': p_copy_idx,
                'cells': chosen_cells,
                'is_large': p_type == 2,
                'is_magic1': p_type == 3,
                'is_magic2': p_type == 4,
                'size': PIECE_SIZES[p_type][p_shape_idx] if p_type < 3 else 1,
                'rotation_idx': rot_idx, # Store for visualization
                'start_pos': (chosen_r, chosen_c) # Store for visualization (using top-left of bounding box)
            }
            used_inventory_ids.add(piece_id)
            # Continue to the next gene

        # If potential_placements is empty, the piece couldn't be placed in any available spot.

    return current_grid, current_placed_info


# --- 3. 实现适应度函数 (包含更智能的放置) ---

def calculate_score_intelligent_constructive(individual):
    """
    Calculates the score for an individual using an intelligent greedy placement strategy.
    Individual is a list of (piece_id, rotation_index) tuples.
    Intelligent strategy: For each piece, find all valid spots and choose the one
    that maximizes the immediate local score (base + adjacency + magic contribution).
    """
    # This function now primarily calls the place_pieces_into_grid helper
    grid = [[None for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
    placed_pieces_info = {} # Start with empty grid and info

    final_grid, final_placed_info = place_pieces_into_grid(grid, placed_pieces_info, individual)

    # Calculate Total Score based on the final grid state
    total_score = calculate_total_score_from_placement(final_grid, final_placed_info)

    return total_score, # DEAP fitness functions must return a tuple

# --- 4. 配置 DEAP 工具箱 (更新变异操作) ---

toolbox = base.Toolbox()

# Helper to generate a random gene (piece_id, rot_idx)
def generate_random_gene(piece_id):
    p_type, p_shape_idx, p_copy_idx = piece_id
    possible_rotations = ALL_ROTATED_SHAPES[(p_type, p_shape_idx)]
    rot_idx = random.randrange(len(possible_rotations))
    return (piece_id, rot_idx)

# Individual initializer: Creates a list of genes (piece_id, rot_idx)
def init_individual_constructive(ind_cls, piece_inventory_list, min_pieces, max_pieces):
    """Initializes an individual with a random number of unique pieces and random rotations."""
    num_pieces = random.randint(min_pieces, max_pieces)
    available_pieces = list(piece_inventory_list) # Copy the list
    random.shuffle(available_pieces) # Shuffle to pick random unique pieces

    individual = ind_cls()

    for piece_id in available_pieces:
        if len(individual) >= num_pieces:
            break

        gene = generate_random_gene(piece_id)
        individual.append(gene)

    return individual

# Register the individual and population creation
MIN_INITIAL_PIECES = 25
MAX_INITIAL_PIECES = TOTAL_PIECES_COUNT

toolbox.register("individual", init_individual_constructive, creator.Individual, PIECE_INVENTORY, MIN_INITIAL_PIECES, MAX_INITIAL_PIECES)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register the evaluation function (using the intelligent placement)
toolbox.register("evaluate", calculate_score_intelligent_constructive)

# Register the genetic operators (crossover same as before)
def custom_crossover_constructive(ind1, ind2):
    size1 = len(ind1)
    size2 = len(ind2)
    if size1 < 1 or size2 < 1:
        return ind1, ind2

    cxpoint1 = random.randint(0, size1)
    cxpoint2 = random.randint(0, size2)

    child1_genes = ind1[:cxpoint1] + ind2[cxpoint2:]
    child2_genes = ind2[:cxpoint2] + ind1[cxpoint1:]

    def remove_duplicate_piece_ids(gene_list):
        seen_ids = set()
        new_gene_list = []
        for gene in gene_list:
            piece_id = gene[0]
            if piece_id not in seen_ids:
                new_gene_list.append(gene)
                seen_ids.add(piece_id)
        return new_gene_list

    child1 = creator.Individual(remove_duplicate_piece_ids(child1_genes))
    child2 = creator.Individual(remove_duplicate_piece_ids(child2_genes))

    return child1, child2

toolbox.register("mate", custom_crossover_constructive)

# Mutation: Custom mutation operators (enhanced)
def custom_mutation_enhanced(individual, prob_gene_rotate, prob_gene_swap, prob_add, prob_remove, prob_block_shuffle, prob_replace_piece):
    """Applies enhanced mutation to an individual (list of genes)."""

    # Mutation 1: Change rotation of existing genes
    for i in range(len(individual)):
        if random.random() < prob_gene_rotate: # Probability per gene
            piece_id, old_rot_idx = individual[i]
            p_type, p_shape_idx, p_copy_idx = piece_id
            possible_rotations = ALL_ROTATED_SHAPES[(p_type, p_shape_idx)]
            if len(possible_rotations) > 1: # Only mutate if there's more than one rotation
                new_rot_idx = random.randrange(len(possible_rotations))
                individual[i] = (piece_id, new_rot_idx)

    # Mutation 2: Swap order of two genes
    if len(individual) > 1 and random.random() < prob_gene_swap:
         swap_idx1, swap_idx2 = random.sample(range(len(individual)), 2)
         individual[swap_idx1], individual[swap_idx2] = individual[swap_idx2], individual[swap_idx1]

    # Mutation 3: Add a new piece
    if random.random() < prob_add:
        placed_ids = {gene[0] for gene in individual}
        available_to_add = [pid for pid in PIECE_INVENTORY if pid not in placed_ids]
        if available_to_add:
            piece_to_add = random.choice(available_to_add)
            new_gene = generate_random_gene(piece_to_add)
            insert_index = random.randint(0, len(individual)) # Allow inserting at end
            individual.insert(insert_index, new_gene)

    # Mutation 4: Remove a piece
    if random.random() < prob_remove and len(individual) > MIN_INITIAL_PIECES:
        remove_index = random.randrange(len(individual))
        individual.pop(remove_index)

    # Mutation 5: Block Shuffle (shuffle a contiguous block of genes)
    if len(individual) > 1 and random.random() < prob_block_shuffle:
        block_size = random.randint(2, min(len(individual), 10)) # Shuffle block of 2 to 10 genes
        start_idx = random.randint(0, len(individual) - block_size)
        block = individual[start_idx : start_idx + block_size]
        random.shuffle(block)
        individual[start_idx : start_idx + block_size] = block

    # Mutation 6: Replace a piece with an unused one
    if len(individual) > 0 and random.random() < prob_replace_piece:
        placed_ids = {gene[0] for gene in individual}
        available_to_add = [pid for pid in PIECE_INVENTORY if pid not in placed_ids]
        if available_to_add:
            replace_index = random.randrange(len(individual))
            # piece_to_replace_id = individual[replace_index][0] # Get the ID of the piece being replaced

            piece_to_add_id = random.choice(available_to_add)
            new_gene = generate_random_gene(piece_to_add_id)

            individual[replace_index] = new_gene # Replace the gene

    return individual,

# Mutation probabilities (adjusted)
PROB_GENE_ROTATE = 0.1
PROB_GENE_SWAP = 0.1
PROB_ADD_PIECE = 0.05
PROB_REMOVE_PIECE = 0.05
PROB_BLOCK_SHUFFLE = 0.05
PROB_REPLACE_PIECE = 0.05


toolbox.register("mutate", custom_mutation_enhanced,
                 prob_gene_rotate=PROB_GENE_ROTATE,
                 prob_gene_swap=PROB_GENE_SWAP,
                 prob_add=PROB_ADD_PIECE,
                 prob_remove=PROB_REMOVE_PIECE,
                 prob_block_shuffle=PROB_BLOCK_SHUFFLE,
                 prob_replace_piece=PROB_REPLACE_PIECE)


toolbox.register("select", tools.selTournament, tournsize=3)

# --- Helper function to get placement details from an individual ---
# This is essentially the placement part of the fitness function,
# but it returns the grid and placed_pieces_info instead of just the score.
# This function is used by the local search.
def get_placement_details(individual):
    """Runs the intelligent constructive placement for an individual and returns grid and placed info."""
    grid = [[None for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
    placed_pieces_info = {} # Start with empty grid and info

    final_grid, final_placed_info = place_pieces_into_grid(grid, placed_pieces_info, individual)

    return final_grid, final_placed_info

# --- Helper functions for Local Search Neighbor Generation ---

def generate_remove_k_neighbor(current_grid, current_placed_info, initial_individual, k_remove):
    """
    Generates a neighbor solution by removing K pieces from the current placement
    and re-placing them using the intelligent greedy strategy.
    """
    if not current_placed_info or len(current_placed_info) < k_remove:
         # Cannot perform remove-K if not enough pieces
         return copy.deepcopy(current_grid), copy.deepcopy(current_placed_info)

    # 1. Choose K random pieces to remove from the *currently placed* pieces
    pieces_to_remove_ids = random.sample(list(current_placed_info.keys()), k_remove)

    # 2. Create a temporary state with these K pieces removed
    temp_grid = [[None for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
    temp_placed_info = {}
    genes_to_re_place = [] # Store genes for pieces to be re-placed

    # Create a map from piece_id to its original index in the initial_individual
    original_order_map = {gene[0]: idx for idx, gene in enumerate(initial_individual)}

    for p_id, details in current_placed_info.items():
        if p_id not in pieces_to_remove_ids:
            # Copy details for pieces that remain
            temp_placed_info[p_id] = copy.deepcopy(details)
            for r, c in details['cells']:
                temp_grid[r][c] = p_id
        else:
             # Reconstruct the gene (piece_id, rot_idx) for the piece to be re-placed
             # Use the rotation_idx stored in the placed_info
             genes_to_re_place.append((p_id, details['rotation_idx']))

    # 3. Attempt to re-place the K pieces using the intelligent greedy strategy
    # Re-place in their original order from the initial_individual if possible,
    # otherwise use the order they were selected for removal.
    # Sorting by original order is better as it respects the chromosome structure.
    genes_to_re_place.sort(key=lambda gene: original_order_map.get(gene[0], float('inf'))) # Sort by original order

    # Use the place_pieces_into_grid helper to place the removed pieces back
    final_grid, final_placed_info = place_pieces_into_grid(temp_grid, temp_placed_info, genes_to_re_place)

    return final_grid, final_placed_info

def generate_translate_subset_neighbor(current_grid, current_placed_info, initial_individual, num_pieces_to_translate):
    """
    Generates a neighbor solution by translating a subset of pieces from the current placement,
    removing conflicts, and filling gaps with available pieces using intelligent placement.
    """
    if not current_placed_info:
        return copy.deepcopy(current_grid), copy.deepcopy(current_placed_info) # Cannot translate if nothing is placed

    placed_piece_ids = list(current_placed_info.keys())
    # Ensure we don't try to translate more pieces than are placed
    num_to_translate = min(num_pieces_to_translate, len(placed_piece_ids))

    if num_to_translate == 0:
         return copy.deepcopy(current_grid), copy.deepcopy(current_placed_info) # Cannot translate if num_to_translate is 0

    # 1. Select pieces and choose a random direction for each selected piece
    pieces_to_translate_ids = random.sample(placed_piece_ids, num_to_translate)
    translate_directions = {pid: random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)]) for pid in pieces_to_translate_ids}

    # 2. Prepare list of pieces to process for the next grid
    # This list determines the order in which pieces attempt to occupy cells in the new grid
    pieces_to_process = [] # List of (piece_id, direction or None)

    # Add pieces to be translated
    for pid in pieces_to_translate_ids:
        pieces_to_process.append((pid, translate_directions[pid]))

    # Add pieces NOT to be translated
    for pid in placed_piece_ids:
        if pid not in pieces_to_translate_ids:
            pieces_to_process.append((pid, None))

    # Shuffle the processing order to handle conflicts randomly
    random.shuffle(pieces_to_process)

    # 3. Build the next grid and placed info based on the processing order
    next_grid = [[None for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
    next_placed_info = {}
    placed_ids_in_next = set() # Keep track of which pieces successfully got placed in next_grid

    for piece_id, direction in pieces_to_process:
        original_details = current_placed_info[piece_id]
        original_cells = original_details['cells']
        original_rot_idx = original_details['rotation_idx'] # Need rotation for piece details

        target_cells = []
        is_valid_placement = True

        if direction is None: # Piece is not translated, try to place it at its original location
            target_cells = original_cells
        else: # Piece is translated
            dr, dc = direction
            target_cells = [(r + dr, c + dc) for r, c in original_cells]
            # Check bounds for translated piece
            for r, c in target_cells:
                if not (0 <= r < GRID_HEIGHT and 0 <= c < GRID_WIDTH):
                    is_valid_placement = False
                    break
            # If out of bounds, skip this piece in this neighbor state
            if not is_valid_placement:
                 continue


        # Check for overlaps in the next_grid being built
        # If any target cell is occupied, this piece cannot be placed in this spot in this neighbor state
        for r, c in target_cells:
            if next_grid[r][c] is not None:
                is_valid_placement = False
                break

        if is_valid_placement:
            # Place the piece in next_grid and add to next_placed_info
            for r, c in target_cells:
                next_grid[r][c] = piece_id

            # Update details for the placed piece
            p_type, p_shape_idx, p_copy_idx = piece_id
            next_placed_info[piece_id] = {
                'type': p_type,
                'shape_idx': p_shape_idx,
                'copy_idx': p_copy_idx,
                'cells': target_cells,
                'is_large': p_type == 2,
                'is_magic1': p_type == 3,
                'is_magic2': p_type == 4,
                'size': PIECE_SIZES[p_type][p_shape_idx] if p_type < 3 else 1,
                'rotation_idx': original_rot_idx, # Keep original rotation
                'start_pos': (min(r for r,c in target_cells), min(c for r,c in target_cells)) # Using top-left as start_pos
            }
            placed_ids_in_next.add(piece_id)


    # 4. Identify available pieces from the original individual that were not placed in next_grid
    # We need the genes (piece_id, rot_idx) for these pieces
    available_genes_for_filling = [gene for gene in initial_individual if gene[0] not in placed_ids_in_next]

    # 5. Fill gaps using intelligent placement on the available pieces
    final_grid, final_placed_info = place_pieces_into_grid(next_grid, next_placed_info, available_genes_for_filling)

    return final_grid, final_placed_info


def generate_uniform_translate_neighbor(current_grid, current_placed_info, initial_individual):
    """
    Generates a neighbor solution by translating ALL placed pieces uniformly,
    removing conflicts/out-of-bounds, and filling gaps using intelligent placement.
    """
    if not current_placed_info:
        return copy.deepcopy(current_grid), copy.deepcopy(current_placed_info) # Cannot translate if nothing is placed

    # 1. Choose a single random direction for all pieces
    dr, dc = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
    # print(f"Attempting uniform translation by ({dr}, {dc})")

    # 2. Prepare list of pieces to process for the next grid
    # Process pieces in their original order from the initial_individual to resolve conflicts
    original_order_map = {gene[0]: idx for idx, gene in enumerate(initial_individual)}
    placed_piece_ids = list(current_placed_info.keys())
    # Create a list of (piece_id, original_index) for sorting
    pieces_to_process_sorted = sorted(
        [(pid, original_order_map.get(pid, float('inf'))) for pid in placed_piece_ids],
        key=lambda item: item[1]
    )

    # 3. Build the next grid and placed info based on the sorted processing order
    next_grid = [[None for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
    next_placed_info = {}
    placed_ids_in_next = set()

    for piece_id, _ in pieces_to_process_sorted:
        # Only process if the piece was actually in the current placement
        if piece_id not in current_placed_info:
             continue

        original_details = current_placed_info[piece_id]
        original_cells = original_details['cells']
        original_rot_idx = original_details['rotation_idx'] # Need rotation for piece details


        # Calculate translated cells
        target_cells = [(r + dr, c + dc) for r, c in original_cells]

        # Check bounds for translated piece
        is_valid_placement = True
        for r, c in target_cells:
            if not (0 <= r < GRID_HEIGHT and 0 <= c < GRID_WIDTH):
                is_valid_placement = False
                break

        if not is_valid_placement:
            # print(f"Piece {piece_id} out of bounds after uniform translation.")
            continue # Piece is out of bounds after translation

        # Check for overlaps in the next_grid being built
        # If any target cell is occupied, this piece cannot be placed in this spot
        for r, c in target_cells:
            if next_grid[r][c] is not None:
                is_valid_placement = False
                # print(f"Piece {piece_id} conflicts at ({r},{c}) after uniform translation.")
                break

        if is_valid_placement:
            # Place the piece in next_grid and add to next_placed_info
            for r, c in target_cells:
                next_grid[r][c] = piece_id

            # Update details for the placed piece
            p_type, p_shape_idx, p_copy_idx = piece_id
            next_placed_info[piece_id] = {
                'type': p_type,
                'shape_idx': p_shape_idx,
                'copy_idx': p_copy_idx,
                'cells': target_cells,
                'is_large': p_type == 2,
                'is_magic1': p_type == 3,
                'is_magic2': p_type == 4,
                'size': PIECE_SIZES[p_type][p_shape_idx] if p_type < 3 else 1,
                'rotation_idx': original_rot_idx, # Keep original rotation
                'start_pos': (min(r for r,c in target_cells), min(c for r,c in target_cells)) # Using top-left as start_pos
            }
            placed_ids_in_next.add(piece_id)

    # 4. Identify available pieces from the original individual that were not placed in next_grid
    available_genes_for_filling = [gene for gene in initial_individual if gene[0] not in placed_ids_in_next]

    # 5. Fill gaps using intelligent placement on the available pieces
    final_grid, final_placed_info = place_pieces_into_grid(next_grid, next_placed_info, available_genes_for_filling)

    return final_grid, final_placed_info


# --- 5. 实现局部搜索 (智能移动选择 + 移除 K 个 + 平移 + 模拟退火) ---

def run_local_search_enhanced(initial_individual, max_ls_iterations=1500, k_remove=3, initial_temp=100.0, cooling_rate=0.998, prob_translate_subset=0.25, num_pieces_to_translate=5, prob_translate_uniform=0.25):
    """
    Applies an enhanced local search with 'remove K and re-place', 'translate subset',
    'translate uniform', and simulated annealing.
    Runs for a fixed number of iterations starting from the placement of initial_individual.
    Returns the best grid, placed_info, and score found *within this LS run*.
    """
    # print("\nStarting Enhanced Local Search (Remove K / Translate + Simulated Annealing)...")
    # print(f"LS Iterations: {max_ls_iterations}, K to Remove: {k_remove}, Initial Temp: {initial_temp}, Cooling Rate: {cooling_rate}, Translate Subset Prob: {prob_translate_subset}, Translate Subset Num: {num_pieces_to_translate}, Translate Uniform Prob: {prob_translate_uniform}")

    # Get the initial placement and score from the provided individual
    current_grid, current_placed_info = get_placement_details(
        
    )
    current_score = calculate_total_score_from_placement(current_grid, current_placed_info)

    best_grid_ls_run = copy.deepcopy(current_grid)
    best_placed_info_ls_run = copy.deepcopy(current_placed_info)
    best_score_ls_run = current_score

    # print(f"LS Starting Score: {current_score}")

    temp = initial_temp

    # Ensure probabilities sum up correctly for selection
    total_translate_prob = prob_translate_subset + prob_translate_uniform
    prob_remove_k = 1.0 - total_translate_prob
    if prob_remove_k < 0:
        prob_remove_k = 0 # Safeguard


    for i in range(max_ls_iterations):
        # Decide which type of neighbor generation to use
        rand_prob = random.random()
        neighbor_grid = None
        neighbor_placed_info = None

        if rand_prob < prob_translate_uniform:
            # --- Generate neighbor using Translate Uniform ---
            neighbor_grid, neighbor_placed_info = generate_uniform_translate_neighbor(
                current_grid,
                current_placed_info,
                initial_individual # Pass initial_individual for re-placement order
            )
        elif rand_prob < prob_translate_uniform + prob_translate_subset:
             # --- Generate neighbor using Translate Subset ---
            neighbor_grid, neighbor_placed_info = generate_translate_subset_neighbor(
                current_grid,
                current_placed_info,
                initial_individual, # Pass initial_individual to get available pieces and original order
                num_pieces_to_translate=random.randint(1, num_pieces_to_translate) # Translate 1 to num_pieces_to_translate
            )
        elif prob_remove_k > 0: # Only attempt remove-K if its probability is > 0
            # --- Generate neighbor using Remove K and Re-place ---
            neighbor_grid, neighbor_placed_info = generate_remove_k_neighbor(
                current_grid,
                current_placed_info,
                initial_individual, # Pass initial_individual for re-placement order
                k_remove
            )

        # If no neighbor was generated (e.g., not enough pieces for remove-K or translate), skip iteration
        if neighbor_grid is None:
             continue

        # Evaluate the new state
        next_score = calculate_total_score_from_placement(neighbor_grid, neighbor_placed_info)

        # --- Acceptance Criteria (Simulated Annealing) ---
        # Accept better solutions
        if next_score > current_score:
            current_grid = neighbor_grid
            current_placed_info = neighbor_placed_info
            current_score = next_score
            # Update best found *in this LS run*
            if current_score > best_score_ls_run:
                best_score_ls_run = current_score
                best_grid_ls_run = copy.deepcopy(current_grid)
                best_placed_info_ls_run = copy.deepcopy(current_placed_info)
                # print(f"LS Iter {i+1}: New best score in run {best_score_ls_run}")
        # Accept worse solutions with a probability
        elif temp > 0:
            delta = next_score - current_score # delta is negative
            acceptance_probability = math.exp(delta / (temp + 1e-9)) # Add epsilon to temp
            if random.random() < acceptance_probability:
                current_grid = neighbor_grid
                current_placed_info = neighbor_placed_info
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

    # Return the best state found *within this specific LS run*
    return best_grid_ls_run, best_placed_info_ls_run, best_score_ls_run


# --- Global variables to track the overall best solution found ---
overall_best_score = -float('inf') # Initialize with a very low score
overall_best_grid = None
overall_best_placed_info = None

# --- Log file configuration ---
LOG_FILE_NAME = "best_solution_log.txt"

# --- Modular function to update the overall best solution and log it ---
def update_overall_best(score, grid, placed_info):
    """
    Updates the global overall best solution if the provided score is higher.
    If updated, logs the new best solution details to a file.
    Returns True if the overall best was updated, False otherwise.
    """
    global overall_best_score, overall_best_grid, overall_best_placed_info
    updated = False
    if score > overall_best_score:
        overall_best_score = score
        # Store deep copies to prevent modification later
        overall_best_grid = copy.deepcopy(grid)
        overall_best_placed_info = copy.deepcopy(placed_info)
        updated = True
        print(f"--- New overall best score found: {overall_best_score} ---") # Print to console immediately

        # --- Log the new best solution ---
        try:
            with open(LOG_FILE_NAME, 'a') as f:
                f.write("="*80 + "\n")
                f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"New Overall Best Score: {overall_best_score}\n")
                f.write(f"Number of pieces successfully placed: {len(overall_best_placed_info)}\n")
                f.write("\nPlaced Pieces:\n")

                # Sort pieces by their top-left corner for consistent output in log
                final_placed_list = sorted(overall_best_placed_info.values(), key=lambda x: (x['start_pos'][0], x['start_pos'][1]))
                for details in final_placed_list:
                     piece_id = (details['type'], details['shape_idx'], details['copy_idx'])
                     f.write(f"  - {PIECE_TYPE_NAMES[details['type']]} Shape {details['shape_idx']} Copy {details['copy_idx']} (ID: {piece_id}) at ({details['start_pos'][0]},{details['start_pos'][1]}) Rot {details['rotation_idx']}\n")

                f.write("\nGrid Visualization:\n")
                visual_grid = [[' . ' for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
                piece_char_map = {}
                char_counter = 0
                chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz#@$&*+=-"
                for piece_id, details in overall_best_placed_info.items():
                     if piece_id not in piece_char_map:
                         piece_char_map[piece_id] = chars[char_counter % len(chars)]
                         char_counter += 1
                     char_to_use = piece_char_map[piece_id]
                     for r, c in details['cells']:
                         visual_grid[r][c] = f" {char_to_use} "

                f.write("   " + "".join([f" {c} " for c in range(GRID_WIDTH)]) + "\n")
                for r in range(GRID_HEIGHT):
                    f.write(f"{r} |" + "".join(visual_grid[r]) + "\n")
                f.write("-" * (GRID_WIDTH * 3 + 4) + "\n") # Separator
                f.write("="*80 + "\n\n") # End of entry separator

        except IOError as e:
            print(f"Error writing to log file {LOG_FILE_NAME}: {e}")

    return updated


# --- 6. 运行遗传算法并接续局部搜索 (修改为无限循环和周期性LS) ---

def main():
    """Runs the genetic algorithm infinitely with periodic local search."""
    start_time = time.time()
    random.seed(42) # for reproducibility

    # GA parameters
    POPULATION_SIZE = 800
    CXPB = 0.7
    MUTPB = 0.3
    # ELITISM_SIZE is implicitly handled by HallOfFame and selection/replacement strategy

    print("Starting Genetic Algorithm with Intelligent Constructive Placement...")
    print(f"Population Size: {POPULATION_SIZE}")
    print(f"Min/Max Initial Pieces: {MIN_INITIAL_PIECES}/{MAX_INITIAL_PIECES}")
    print(f"Logging best solutions to: {LOG_FILE_NAME}")


    # --- Define known good solutions as chromosomes ---
    # 210 Score Solution (from your output)
    solution_210_genes = [
        ((0, 1, 2), 0), ((2, 1, 2), 2), ((2, 0, 3), 1), ((1, 2, 3), 0),
        ((3, 0, 1), 0), ((1, 1, 0), 1), ((3, 0, 0), 0), ((2, 3, 1), 3),
        ((1, 3, 0), 3), ((4, 0, 1), 0), ((4, 0, 0), 0), ((0, 2, 0), 2),
        ((2, 2, 1), 0), ((0, 1, 0), 0)
    ]
    # 208 Score Solution (from previous output)
    solution_208_genes = [
        ((0, 1, 2), 0), ((2, 1, 2), 2), ((1, 2, 3), 0), ((3, 0, 1), 0),
        ((2, 2, 0), 0), ((3, 0, 0), 0), ((2, 3, 1), 3), ((1, 3, 0), 3),
        ((4, 0, 1), 0), ((0, 2, 0), 1), ((4, 0, 0), 0), ((2, 2, 1), 0),
        ((1, 1, 3), 2), ((0, 1, 3), 0)
    ]
     # 211 Score Solution (from your latest output - assuming this order is the placement order)
    solution_211_genes = [
        ((0, 1, 2), 0), # Special Shape 1 Copy 2
        ((2, 1, 2), 2), # Large Shape 1 Copy 2
        ((1, 1, 0), 1), # Land Shape 1 Copy 0
        ((1, 2, 3), 0), # Land Shape 2 Copy 3
        ((2, 0, 3), 1), # Large Shape 0 Copy 3
        ((3, 0, 1), 0), # Magic1 Shape 0 Copy 1
        ((4, 0, 0), 0), # Magic2 Shape 0 Copy 0
        ((2, 3, 1), 3), # Large Shape 3 Copy 1
        ((1, 3, 0), 3), # Land Shape 3 Copy 0
        ((2, 0, 0), 2), # Large Shape 0 Copy 0
        ((4, 0, 1), 0), # Magic2 Shape 0 Copy 1
        ((3, 0, 0), 0), # Magic1 Shape 0 Copy 0
        ((0, 2, 0), 2), # Special Shape 2 Copy 0
        ((0, 1, 0), 0)  # Special Shape 1 Copy 0
    ]

    solution_209_genes = [
        ((0, 0, 2), 0), # Special Shape 0 Copy 2, Rot 0 -> Index 0
        ((2, 0, 2), 0), # Large Shape 0 Copy 2, Rot 0 -> Index 0
        ((0, 1, 3), 1), # Special Shape 1 Copy 3, Rot 1 -> Index 1
        ((1, 2, 3), 0), # Land Shape 2 Copy 3, Rot 0 -> Index 0
        ((2, 0, 1), 2), # Large Shape 0 Copy 1, Rot 3 -> Corrected to Index 2 (last valid for (2,0))
        ((4, 0, 1), 0), # Magic2 Shape 0 Copy 1, Rot 0 -> Index 0
        ((2, 3, 2), 1), # Large Shape 3 Copy 2, Rot 1 -> Index 1
        ((4, 0, 0), 0), # Magic2 Shape 0 Copy 0, Rot 0 -> Index 0
        ((2, 3, 1), 1), # Large Shape 3 Copy 1, Rot 1 -> Index 1
        ((1, 2, 2), 0), # Land Shape 2 Copy 2, Rot 0 -> Index 0
        ((3, 0, 1), 0), # Magic1 Shape 0 Copy 1, Rot 0 -> Index 0
        ((3, 0, 0), 0), # Magic1 Shape 0 Copy 0, Rot 0 -> Index 0
        ((0, 1, 1), 0), # Special Shape 1 Copy 1, Rot 0 -> Index 0
        ((1, 0, 2), 0)  # Land Shape 0 Copy 2, Rot 0 -> Index 0
    ]

    # solution_212_genes = [
    #     ((0, 1, 0), 0),
    #     ((0, 1, 1), 0),
    #     ((0, 1, 2), 1),
    #     ((1, 0, 0), 1),
    #     ((1, 1, 0), 0),
    #     ((1, 3, 0), 2),
    #     ((2, 0, 0), 1),
    #     ((2, 0, 1), 3),
    #     ((2, 3, 0), 2),
    #     ((2, 3, 1), 0),
    #     ((3, 0, 0), 0),
    #     ((3, 0, 1), 0),
    #     ((4, 0, 0), 0),
    #     ((4, 0, 1), 0)
    # ]
    seeded_solutions_genes = [
        # solution_210_genes,
        # solution_208_genes,
        # solution_211_genes,
        # solution_209_genes,
    ]
    num_seeded = len(seeded_solutions_genes)

    # Create initial population (mostly random)
    population = toolbox.population(n=POPULATION_SIZE - num_seeded) # Make space for seeded solutions

    # Add the known good solutions to the population
    seeded_individuals = [creator.Individual(genes) for genes in seeded_solutions_genes]
    population.extend(seeded_individuals)

    # Shuffle the population to mix seeded and random individuals
    random.shuffle(population)


    # Evaluate the initial population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    print(f"Evaluated {len(population)} individuals in initial population (including {num_seeded} seeded).")
    initial_scores = [ind.fitness.values[0] for ind in population]
    print(f"Initial min/avg/max fitness: {np.min(initial_scores)}/{np.mean(initial_scores):.2f}/{np.max(initial_scores)}")

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("avg", np.mean)
    stats.register("min", np.min)

    # Update Hall of Fame and stats with the initial population
    hof.update(population)
    record = stats.compile(population)
    logbook = tools.Logbook()
    logbook.header = ['gen', 'evals'] + stats.fields
    logbook.record(gen=0, evals=len(population), **record) # Record initial state

    # Initialize overall best solution found so far using the initial GA best
    initial_ga_best = hof[0]
    initial_grid, initial_placed_info = get_placement_details(initial_ga_best)
    initial_score = calculate_total_score_from_placement(initial_grid, initial_placed_info)
    # Use the modular update function - this will also log the initial best if it's > -inf
    update_overall_best(initial_score, initial_grid, initial_placed_info)

    print(f"Initial Overall Best Score: {overall_best_score}") # Now this uses the globally tracked best

    # Local Search parameters for periodic runs
    ls_iterations_per_run = 1500 # Number of iterations for each LS call
    ls_k_remove = 3
    ls_initial_temp = 100.0
    ls_cooling_rate = 0.998
    ls_prob_translate_subset = 0.25 # Probability of choosing translate subset neighbor
    ls_num_pieces_to_translate = 5 # Max pieces for subset translation
    ls_prob_translate_uniform = 0.25 # Probability of choosing translate uniform neighbor
    # Remaining probability (1.0 - 0.25 - 0.25 = 0.5) is for remove-K neighbor

    print(f"\nStarting Infinite Genetic Algorithm with Periodic Local Search...")
    print(f"LS runs every 500 generations with {ls_iterations_per_run} iterations.")
    print(f"LS Neighbor Probabilities: Remove-K={1.0 - ls_prob_translate_subset - ls_prob_translate_uniform:.2f}, Translate Subset={ls_prob_translate_subset:.2f} (max {ls_num_pieces_to_translate} pieces), Translate Uniform={ls_prob_translate_uniform:.2f}")
    print(f"LS SA Params: Initial Temp={ls_initial_temp}, Cooling Rate={ls_cooling_rate}")
    print("Press Ctrl+C to stop.")

    try:
        # Use tqdm for the infinite loop
        pbar = tqdm(itertools.count(1), desc=f"GA Progress (Best Score: {overall_best_score})", unit="gen")
        for gen in pbar:

            # --- GA Generation Steps (from eaSimple) ---
            # Select the next generation individuals
            offspring = toolbox.select(population, POPULATION_SIZE)
            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            for mutant in offspring:
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # The population is replaced by the offspring
            population[:] = offspring

            # Update the Hall of Fame with the generated individuals
            hof.update(population)

            # Update the statistics with the new population
            record = stats.compile(population)
            logbook.record(gen=gen, evals=len(invalid_ind), **record)

            # --- Check and Update overall best score from current GA best ---
            current_ga_best = hof[0]
            current_ga_score = current_ga_best.fitness.values[0]
            # Only get placement details if the GA best is potentially better than the current overall best
            if current_ga_score > overall_best_score:
                 current_ga_grid, current_ga_placed_info = get_placement_details(current_ga_best)
                 # update_overall_best will handle logging if it's a new best
                 update_overall_best(current_ga_score, current_ga_grid, current_ga_placed_info)

            # Always update tqdm description with the current overall_best_score
            pbar.set_description(f"GA Progress (Best Score: {overall_best_score})")
            # --- End of every-generation update ---


            # --- Periodic Local Search and Output ---
            if gen % 500 == 0:
                print(f"\n--- Generation {gen}: Running Local Search on current GA best ---")
                # Get the current best individual from the GA's Hall of Fame
                current_ga_best_individual = hof[0]
                ls_grid, ls_placed_info, ls_score = run_local_search_enhanced(
                    current_ga_best_individual, # Pass the current GA best individual to LS
                    max_ls_iterations=ls_iterations_per_run,
                    k_remove=ls_k_remove,
                    initial_temp=ls_initial_temp,
                    cooling_rate=ls_cooling_rate,
                    prob_translate_subset=ls_prob_translate_subset,
                    num_pieces_to_translate=ls_num_pieces_to_translate,
                    prob_translate_uniform=ls_prob_translate_uniform
                )

                # Attempt to update the overall best with the LS result
                # update_overall_best will handle logging if it's a new best
                ls_improved_overall = update_overall_best(ls_score, ls_grid, ls_placed_info)

                # Print messages based on whether LS improved the overall best
                if ls_improved_overall:
                     print(f"--- Generation {gen}: LS found a new overall best score: {overall_best_score} ---")
                else:
                     print(f"--- Generation {gen}: LS finished. Best score in this run: {ls_score}. Overall best remains: {overall_best_score} ---")

                # The pbar description is updated right after this block by the every-gen logic.


    except KeyboardInterrupt:
        print("\nCtrl+C detected. Stopping.")
        # Close the tqdm bar cleanly
        pbar.close()


    # --- Final Output after stopping ---
    print("\n--- Final Best Solution Found ---")
    # These variables now hold the consistent best solution found across GA and LS
    print(f"Max Score: {overall_best_score}")
    print(f"Number of pieces successfully placed: {len(overall_best_placed_info)}")

    print("\nPlaced Pieces:")
    final_placed_list = sorted(overall_best_placed_info.values(), key=lambda x: (x['start_pos'][0], x['start_pos'][1]))
    for details in final_placed_list:
         piece_id = (details['type'], details['shape_idx'], details['copy_idx'])
         print(f"  - {PIECE_TYPE_NAMES[details['type']]} Shape {details['shape_idx']} Copy {details['copy_idx']} (ID: {piece_id}) at ({details['start_pos'][0]},{details['start_pos'][1]}) Rot {details['rotation_idx']}")

    print("\nGrid Visualization:")
    visual_grid = [[' . ' for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
    piece_char_map = {}
    char_counter = 0
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz#@$&*+=-"
    for piece_id, details in overall_best_placed_info.items():
         if piece_id not in piece_char_map:
             piece_char_map[piece_id] = chars[char_counter % len(chars)]
             char_counter += 1
         char_to_use = piece_char_map[piece_id]
         for r, c in details['cells']:
             visual_grid[r][c] = f" {char_to_use} "

    print("   " + "".join([f" {c} " for c in range(GRID_WIDTH)]))
    for r in range(GRID_HEIGHT):
        print(f"{r} |" + "".join(visual_grid[r]))

    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
