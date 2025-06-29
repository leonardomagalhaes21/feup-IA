import numpy as np
import random
import copy
import math
from tqdm import tqdm
from sklearn.cluster import KMeans

class TableOptimizer:
    """Class containing different optimization algorithms for table assignment problems."""
    
    def __init__(self, relationship_matrix, guests, table_size=8):
        SEED = 420
        random.seed(SEED)
        np.random.seed(SEED)

        self.relationship_matrix = relationship_matrix
        self.guests = guests
        self.table_size = table_size
        self.num_tables = (len(guests) + table_size - 1) // table_size
        self.tables = []
        
    def calculate_total_happiness(self, tables):
        """Calculate total happiness across all tables."""
        total_happiness = 0
        
        for table in tables:
            # Get indices of guests at this table
            guest_indices = [self.guests.index(guest) for guest in table]
            
            # Calculate pairwise happiness for this table
            for i, idx1 in enumerate(guest_indices):
                for idx2 in guest_indices[i+1:]:
                    total_happiness += self.relationship_matrix[idx1][idx2]
        
        return total_happiness
    
    # Alias for compatibility
    _calculate_happiness = calculate_total_happiness
    
    def generate_initial_solution(self):
        """Generate a random initial table arrangement."""
        # Create a shuffled list of all guests
        all_guests = copy.deepcopy(self.guests)
        random.shuffle(all_guests)
        
        # Divide guests into tables
        tables = []
        for i in range(0, len(all_guests), self.table_size):
            tables.append(all_guests[i:min(i + self.table_size, len(all_guests))])
            
        return tables
    
    def get_neighbor_solution(self, tables):
        """Generate a neighbor solution by swapping two guests from different tables."""
        new_tables = copy.deepcopy(tables)
        
        # Select two random tables
        table1_idx = random.randint(0, len(new_tables) - 1)
        table2_idx = random.randint(0, len(new_tables) - 1)
        
        # Ensure we select different tables
        while table1_idx == table2_idx:
            table2_idx = random.randint(0, len(new_tables) - 1)
        
        # Select a random guest from each table
        if len(new_tables[table1_idx]) > 0 and len(new_tables[table2_idx]) > 0:
            guest1_idx = random.randint(0, len(new_tables[table1_idx]) - 1)
            guest2_idx = random.randint(0, len(new_tables[table2_idx]) - 1)
            
            # Swap guests
            new_tables[table1_idx][guest1_idx], new_tables[table2_idx][guest2_idx] = \
                new_tables[table2_idx][guest2_idx], new_tables[table1_idx][guest1_idx]
        
        return new_tables
    
    def hill_climbing(self, iterations=1000, max_stagnation=100, num_neighbors=3, restarts=1):
        """
        Improved hill climbing algorithm with steepest ascent and random restarts.
        
        Args:
            iterations: Maximum number of iterations per restart
            max_stagnation: Maximum iterations without improvement before early stopping
            num_neighbors: Number of neighbors to generate at each step
            restarts: Number of random restarts
            
        Returns:
            Tuple of (best solution, best score)
        """
        best_solution = None
        best_score = float('-inf')
        
        for restart in range(restarts):
            current_solution = self.assign_tables_greedy()
            current_score = self.calculate_total_happiness(current_solution)
            stagnation_counter = 0
            
            for _ in tqdm(range(iterations), desc=f"Hill Climbing (Restart {restart+1}/{restarts})"):
                # Generate multiple neighbors and select the best one
                best_neighbor = None
                best_neighbor_score = current_score
                
                for _ in range(num_neighbors):
                    neighbor_solution = self.get_neighbor_solution(current_solution)
                    neighbor_score = self.calculate_total_happiness(neighbor_solution)
                    
                    if neighbor_score > best_neighbor_score:
                        best_neighbor = neighbor_solution
                        best_neighbor_score = neighbor_score
                
                # If we found a better neighbor, move to it
                if best_neighbor_score > current_score:
                    current_solution = best_neighbor
                    current_score = best_neighbor_score
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                    
                # Update global best solution
                if current_score > best_score:
                    best_solution = copy.deepcopy(current_solution)
                    best_score = current_score
                    
                # Early stopping if no improvement for max_stagnation iterations
                if stagnation_counter >= max_stagnation:
                    break
        
        return best_solution, best_score
    
    def simulated_annealing(self, initial_temp=200, cooling_rate=0.98, iterations=1000, 
                            neighbors_per_iter=7, max_stagnation=100, reheat_factor=1.5):
        """
        Implement enhanced simulated annealing algorithm with:
        - Multiple neighbor evaluation
        - Reheating when stuck
        - Stagnation detection and early stopping
        
        Args:
            initial_temp: Starting temperature
            cooling_rate: Rate at which temperature decreases
            iterations: Maximum number of iterations
            neighbors_per_iter: Number of neighbors to evaluate per iteration
            max_stagnation: Maximum iterations without improvement before reheating
            reheat_factor: Factor by which to increase temperature when reheating
        
        Returns:
            Tuple of (best solution, best score)
        """
        # Calculate a more appropriate iterations count based on dataset size
        adaptive_iterations = min(iterations, 300 + len(self.guests) * 2)
        
        # Start with the greedy solution for a better initial point
        current_solution = self.assign_tables_greedy()
        current_score = self.calculate_total_happiness(current_solution)
        best_solution = copy.deepcopy(current_solution)
        best_score = current_score
        
        # Skip hill climbing warmup to save time for larger datasets
        if len(self.guests) < 50:  # Only use hill climbing warmup for small datasets
            hc_solution, hc_score = self.hill_climbing(iterations=50, max_stagnation=20, restarts=1)
            if hc_score > best_score:
                current_solution = hc_solution
                current_score = hc_score
                best_solution = copy.deepcopy(current_solution)
                best_score = current_score
        
        # Improved parameters
        temperature = initial_temp
        stagnation_counter = 0
        global_stagnation = 0
        
        # Optimization: pre-calculate max temperature ratio for scaling calculations
        max_temp_ratio = initial_temp / 100
        
        # Cache relationship matrix lookups for performance
        neighbor_cache = {}
        
        for it in tqdm(range(adaptive_iterations), desc="Simulated Annealing"):
            # Early stopping for efficiency
            if temperature < 0.1 or global_stagnation > max_stagnation * 2:
                break
                
            # Generate multiple neighbors and pick the best one
            best_neighbor = None
            best_neighbor_score = float('-inf')
            
            # Adaptive neighbors - fewer neighbors at lower temperatures to save time
            current_neighbors = max(2, int(neighbors_per_iter * min(1.0, temperature / initial_temp)))
            
            for _ in range(current_neighbors):
                neighbor_solution = self.get_neighbor_solution(current_solution)
                
                # Use a solution hash to avoid recalculating happiness for identical solutions
                neighbor_hash = str(neighbor_solution)
                if neighbor_hash in neighbor_cache:
                    neighbor_score = neighbor_cache[neighbor_hash]
                else:
                    neighbor_score = self.calculate_total_happiness(neighbor_solution)
                    # Only cache if we're not at risk of memory issues
                    if len(neighbor_cache) < 1000:
                        neighbor_cache[neighbor_hash] = neighbor_score
                
                if neighbor_score > best_neighbor_score:
                    best_neighbor = neighbor_solution
                    best_neighbor_score = neighbor_score
            
            # Ensure we have a valid neighbor
            if best_neighbor is None:
                continue
                
            # Calculate acceptance probability
            delta = best_neighbor_score - current_score
            
            # Accept improving moves always, and worsening moves with probability based on temperature
            if delta > 0:
                acceptance_probability = 1.0
            else:
                # More aggressive acceptance of worse solutions when at higher temperatures
                acceptance_probability = math.exp(delta / (temperature * 0.7))
            
            # Decide whether to accept the neighbor
            if random.random() < acceptance_probability:
                current_solution = best_neighbor
                current_score = best_neighbor_score
                
                # Reset stagnation if we made an improving move
                if delta > 0:
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
            else:
                stagnation_counter += 0.5  # Slower stagnation count when rejecting moves
            
            # Update best solution if current is better
            if current_score > best_score:
                best_solution = copy.deepcopy(current_solution)
                best_score = current_score
                global_stagnation = 0
            else:
                global_stagnation += 1
            
            # Periodically try to inject diversity - less frequent for large datasets
            inject_period = 50 if len(self.guests) < 50 else 100
            if it % inject_period == 0 and random.random() < 0.3:
                # Make a strong perturbation occasionally
                perturbed = self._strong_perturbation(current_solution)
                perturbed_score = self.calculate_total_happiness(perturbed)
                
                # Accept with some probability based on score
                if perturbed_score > current_score * 0.9:
                    current_solution = perturbed
                    current_score = perturbed_score
            
            # Reheat if stuck in local optima - more aggressive reheating
            if stagnation_counter >= max_stagnation:
                # Higher reheat factor for better escape from local optima
                temperature = initial_temp * 0.8  # Reheat to 80% of initial temperature
                stagnation_counter = 0
                
                # Strong perturbation when reheating - simpler for speed
                current_solution = self._strong_perturbation(current_solution)
                current_score = self.calculate_total_happiness(current_solution)
            else:
                # Faster cooling for larger datasets
                cool_rate = cooling_rate if len(self.guests) < 50 else cooling_rate * 0.99
                temperature *= cool_rate
            
            # Skip global stagnation resets for large datasets to save time
            if len(self.guests) < 50 and global_stagnation >= 2 * max_stagnation:
                if random.random() < 0.7:  # 70% chance to use best solution as base
                    current_solution = self._strong_perturbation(best_solution)
                else:  # 30% chance to start fresh with greedy
                    current_solution = self.assign_tables_greedy()
                
                current_score = self.calculate_total_happiness(current_solution)
                temperature = initial_temp * 0.5  # Start at half temperature after reset
                global_stagnation = 0
        
        # Skip final hill climbing for large datasets to save time
        if len(self.guests) < 400:
            # Final hill climbing phase to optimize the best solution found
            final_solution, final_score = self.hill_climbing(
                iterations=min(200, len(self.guests)),  # Adaptive iterations 
                max_stagnation=100, 
                num_neighbors=5,  # Reduced neighbors for speed
                restarts=2
            )
            
            if final_score > best_score:
                best_solution = final_solution
                best_score = final_score
            
        return best_solution, best_score
    
    def tabu_search(self, tabu_size=15, iterations=1000, max_stagnation=100, strategic_oscillation=True):
        """
        Enhanced tabu search algorithm with dynamic tabu tenure, strategic oscillation,
        and intensification/diversification phases.
        
        Args:
            tabu_size: Initial size of the tabu list
            iterations: Maximum number of iterations
            max_stagnation: Max iterations without improvement before diversification
            strategic_oscillation: Whether to use strategic oscillation
            
        Returns:
            Tuple of (best solution, best score)
        """
        # Generate initial solution
        current_solution = self.assign_tables_greedy()
        current_score = self.calculate_total_happiness(current_solution)
        best_solution = copy.deepcopy(current_solution)
        best_score = current_score
        
        # Initialize tabu structures
        tabu_moves = {}  # Dictionary: (guest1, guest2) -> expiration iteration
        frequency = {}   # Frequency-based memory for diversification
        stagnation = 0   # Count iterations without improvement
        diversify_mode = False
        dynamic_tabu_size = tabu_size
        
        for iter_count in tqdm(range(iterations), desc="Tabu Search"):
            best_neighbor = None
            best_neighbor_score = float('-inf')
            best_move = None
            
            # Generate and evaluate a larger neighborhood
            candidates = []
            for _ in range(10):  # Examine more neighbors
                neighbor = self.get_neighbor_solution(current_solution)
                neighbor_score = self.calculate_total_happiness(neighbor)
                
                # Identify the move (which guests were swapped)
                move = self._identify_move(current_solution, neighbor)
                if move:
                    candidates.append((neighbor, neighbor_score, move))
            
            # Sort candidates by score (descending)
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Select the best non-tabu move or apply aspiration criterion
            for neighbor, neighbor_score, move in candidates:
                is_tabu = move in tabu_moves and tabu_moves[move] > iter_count
                
                # Accept if not tabu, or if aspiration criterion met
                # (aspiration = better than best solution or in diversification mode)
                if (not is_tabu) or (neighbor_score > best_score) or \
                   (diversify_mode and neighbor_score > current_score * 0.95):  # Accept slightly worse in diversify mode
                    best_neighbor = neighbor
                    best_neighbor_score = neighbor_score
                    best_move = move
                    break
            
            # If no valid move found (unlikely), get first candidate
            if best_neighbor is None and candidates:
                best_neighbor, best_neighbor_score, best_move = candidates[0]
            
            # If still no valid move, generate a random solution
            if best_neighbor is None:
                best_neighbor = self.generate_initial_solution()
                best_neighbor_score = self.calculate_total_happiness(best_neighbor)
                best_move = None
            
            # Move to new solution
            current_solution = best_neighbor
            current_score = best_neighbor_score
            
            # Update frequency memory
            if best_move:
                frequency[best_move] = frequency.get(best_move, 0) + 1
                
                # Add move to tabu list
                tenure = dynamic_tabu_size + random.randint(-2, 2)  # Add some randomness to tenure
                tabu_moves[best_move] = iter_count + max(1, tenure)
            
            # Update best solution if improved
            if current_score > best_score:
                best_solution = copy.deepcopy(current_solution)
                best_score = current_score
                stagnation = 0
            else:
                stagnation += 1
            
            # Clean expired tabu moves
            tabu_moves = {k: v for k, v in tabu_moves.items() if v > iter_count}
            
            # Strategic oscillation: alternate between intensification and diversification
            if stagnation >= max_stagnation:
                if strategic_oscillation:
                    diversify_mode = not diversify_mode
                    if diversify_mode:
                        # Diversification: increase tabu tenure to force exploration
                        dynamic_tabu_size = int(tabu_size * 1.5)
                        # Make a larger perturbation to escape local optima
                        current_solution = self._strong_perturbation(current_solution)
                        current_score = self.calculate_total_happiness(current_solution)
                    else:
                        # Intensification: reduce tabu tenure to focus on good regions
                        dynamic_tabu_size = max(3, int(tabu_size * 0.7))
                stagnation = 0
                
            # Early stopping if tabu list gets too large or no movement possible
            if len(tabu_moves) > 3 * tabu_size and stagnation > 2 * max_stagnation:
                break
        
        return best_solution, best_score

    def _identify_move(self, old_solution, new_solution):
        """Identify which guests were swapped between two solutions."""
        # Find tables where old and new solutions differ
        different_tables = []
        for i, (old_table, new_table) in enumerate(zip(old_solution, new_solution)):
            if set(old_table) != set(new_table):
                different_tables.append(i)
        
        # Find the swapped guests
        if len(different_tables) == 2:
            t1, t2 = different_tables
            old_t1, old_t2 = set(old_solution[t1]), set(old_solution[t2])
            new_t1, new_t2 = set(new_solution[t1]), set(new_solution[t2])
            
            moved_to_t1 = list(new_t1 - old_t1)
            moved_to_t2 = list(new_t2 - old_t2)
            
            if len(moved_to_t1) == 1 and len(moved_to_t2) == 1:
                # Sort the guests to ensure consistent representation
                return tuple(sorted([moved_to_t1[0], moved_to_t2[0]]))
        
        # If couldn't identify a simple swap, return None
        return None

    def _strong_perturbation(self, solution):
        """Make a stronger perturbation to escape local optima."""
        perturbed = copy.deepcopy(solution)
        
        # Multiple random swaps
        num_swaps = random.randint(2, 5)
        for _ in range(num_swaps):
            table1_idx = random.randint(0, len(perturbed) - 1)
            table2_idx = random.randint(0, len(perturbed) - 1)
            
            while table1_idx == table2_idx:
                table2_idx = random.randint(0, len(perturbed) - 1)
            
            if perturbed[table1_idx] and perturbed[table2_idx]:
                guest1_idx = random.randint(0, len(perturbed[table1_idx]) - 1)
                guest2_idx = random.randint(0, len(perturbed[table2_idx]) - 1)
                
                perturbed[table1_idx][guest1_idx], perturbed[table2_idx][guest2_idx] = \
                    perturbed[table2_idx][guest2_idx], perturbed[table1_idx][guest1_idx]
        
        return perturbed
    
    def genetic_algorithm(self, population_size=100, generations=200, mutation_rate=0.05, elite_size=5):
        """
        Optimize table assignments using a genetic algorithm.
        
        Args:
            population_size: Size of the population
            generations: Number of generations
            mutation_rate: Probability of mutation
            elite_size: Number of elite solutions to preserve between generations
            
        Returns:
            Tuple of (tables, total happiness score)
        """
        print(f"Running Genetic Algorithm with population size {population_size}, {generations} generations")
        
        # Initialize population with greedy solution and variations
        population = []
        
        # Start with a greedy solution
        greedy_solution = self.assign_tables_greedy()
        greedy_fitness = self.calculate_total_happiness(greedy_solution)
        population.append((greedy_solution, greedy_fitness))
        
        # Add some variations of the greedy solution
        for _ in range(min(int(population_size * 0.3), 20)):  # 30% of population or max 20
            variant = copy.deepcopy(greedy_solution)
            # Apply multiple mutations to create diverse variants
            for _ in range(random.randint(2, 5)):
                variant = self._mutation(variant, mutation_rate * 2)  # Increased mutation rate for diversity
            variant_fitness = self.calculate_total_happiness(variant)
            population.append((variant, variant_fitness))
        
        # Fill the rest with random solutions
        while len(population) < population_size:
            solution = self._create_random_solution()
            fitness = self.calculate_total_happiness(solution)
            population.append((solution, fitness))
            
        # Track the best solution seen
        best_solution = max(population, key=lambda x: x[1])
        
        # Run the genetic algorithm for the specified number of generations
        for generation in range(generations):
            # Sort population by fitness (descending)
            population.sort(key=lambda x: x[1], reverse=True)
            
            # Keep track of best solution
            if population[0][1] > best_solution[1]:
                best_solution = population[0]
                if generation % 10 == 0:  # Print progress every 10 generations
                    print(f"Generation {generation}: Best fitness improved to {best_solution[1]}")
            
            # Create new generation
            new_population = []
            
            # Elitism: Keep the best solutions
            new_population.extend(population[:elite_size])
            
            # Create rest of the population through selection, crossover, and mutation
            while len(new_population) < population_size:
                # Selection (tournament selection)
                parent1 = self._tournament_selection(population, tournament_size=3)
                parent2 = self._tournament_selection(population, tournament_size=3)
                
                # Crossover
                if random.random() < 0.8:  # 80% chance for crossover
                    child = self._crossover(parent1[0], parent2[0])
                else:
                    # Copy one parent if no crossover
                    child = copy.deepcopy(parent1[0]) if random.random() < 0.5 else copy.deepcopy(parent2[0])
                
                # Mutation
                child = self._mutation(child, mutation_rate)
                
                # Evaluate fitness
                fitness = self.calculate_total_happiness(child)
                
                # Add to new population
                new_population.append((child, fitness))
            
            # Replace old population with new generation
            population = new_population
            
        # Get the best solution from final population
        best_solution = max(population, key=lambda x: x[1])
        solution, score = best_solution
        
        return solution, score
        
    def _tournament_selection(self, population, tournament_size=3):
        """Perform tournament selection to choose a parent."""
        # Randomly select tournament_size individuals from the population
        tournament = random.sample(population, tournament_size)
        # Return the individual with the highest fitness
        return max(tournament, key=lambda x: x[1])
        
    def _crossover(self, parent1, parent2):
        """Perform crossover between two parents to create a child solution."""
        child = []
        
        # Get all table indices
        num_tables = len(parent1)
        
        # Randomly choose a split point
        split_point = random.randint(1, num_tables - 1)
        
        # Take first part from parent1
        for i in range(split_point):
            child.append(parent1[i].copy())
        
        # Take remaining guests from parent2, ensuring no guest is duplicated
        assigned_guests = set()
        for table in child:
            assigned_guests.update(table)
        
        # Add remaining tables from parent2
        remaining_tables = []
        for i in range(split_point, num_tables):
            # Only include guests not already in the child solution
            remaining_table = [g for g in parent2[i] if g not in assigned_guests]
            assigned_guests.update(remaining_table)
            remaining_tables.append(remaining_table)
        
        # Add remaining tables to child
        child.extend(remaining_tables)
        
        # Ensure all guests are assigned - add any missing guests
        all_guests_set = set(self.guests)
        assigned_guests = set()
        
        for table in child:
            assigned_guests.update(table)
        
        missing_guests = all_guests_set - assigned_guests
        
        # Distribute missing guests to tables not at capacity
        for guest in missing_guests:
            # Find a table that's not at capacity
            for table in child:
                if len(table) < self.table_size:
                    table.append(guest)
                    break
        
        return child
        
    def _mutation(self, solution, mutation_rate):
        """Apply mutation to a solution based on mutation rate."""
        mutated_solution = [table.copy() for table in solution]
        
        # For each table, consider mutation
        for i in range(len(mutated_solution)):
            if random.random() < mutation_rate and mutated_solution[i]:
                # Pick a random guest from this table
                guest_idx = random.randint(0, len(mutated_solution[i]) - 1)
                guest = mutated_solution[i][guest_idx]
                
                # Choose mutation type
                if random.random() < 0.5:
                    # Swap with another guest from a different table
                    other_table_idx = random.choice([j for j in range(len(mutated_solution)) if j != i and mutated_solution[j]])
                    
                    if other_table_idx is not None and mutated_solution[other_table_idx]:
                        other_guest_idx = random.randint(0, len(mutated_solution[other_table_idx]) - 1)
                        other_guest = mutated_solution[other_table_idx][other_guest_idx]
                        
                        # Perform swap
                        mutated_solution[i][guest_idx] = other_guest
                        mutated_solution[other_table_idx][other_guest_idx] = guest
                else:
                    # Move to another table
                    other_table_idx = random.choice([j for j in range(len(mutated_solution)) if j != i])
                    
                    # Only move if destination table is not full
                    if len(mutated_solution[other_table_idx]) < self.table_size:
                        mutated_solution[i].pop(guest_idx)
                        mutated_solution[other_table_idx].append(guest)
        
        return mutated_solution

    def _create_random_solution(self):
        """
        Creates a random solution for table assignments.
        
        Returns:
            A list of tables, each containing a list of guest names.
        """
        # Create a copy of guest list to work with
        guests_copy = list(self.guests)
        random.shuffle(guests_copy)
        
        # Initialize tables
        tables = [[] for _ in range(self.num_tables)]
        
        # Assign guests to tables
        for i, guest in enumerate(guests_copy):
            table_idx = i % self.num_tables
            tables[table_idx].append(guest)
            
        return tables
        
    def assign_tables_kmeans(self, n_inits=15, balance_method='smart'):
        """
        Use enhanced K-means clustering to assign guests to tables based on relationship scores.
        
        Args:
            n_inits: Number of different initializations for K-means
            balance_method: Method for balancing tables ('smart' or 'basic')
        
        Returns:
            List of tables with guest assignments
        """
        # Transform the relationship matrix to a distance matrix
        # Invert relationships: higher relationship scores = closer distance
        max_score = np.max(self.relationship_matrix)
        distance_matrix = np.array([
            [max_score - self.relationship_matrix[i][j] if i != j else 0 
             for j in range(len(self.guests))]
            for i in range(len(self.guests))
        ])
        
        # Try multiple K-means runs and select the best one
        best_tables = None
        best_happiness = float('-inf')
        
        for _ in range(n_inits):
            # Run K-means with different initialization
            kmeans = KMeans(n_clusters=self.num_tables, n_init=10, random_state=random.randint(0, 999))
            clusters = kmeans.fit_predict(distance_matrix)
            
            # Initialize tables
            tables = [[] for _ in range(self.num_tables)]
            
            # Assign guests to tables based on their cluster
            for i, cluster in enumerate(clusters):
                tables[cluster].append(self.guests[i])
            
            # Calculate happiness before balancing
            happiness = self.calculate_total_happiness(tables)
            
            if happiness > best_happiness:
                best_happiness = happiness
                best_tables = tables
        
        self.tables = best_tables
        
        # Balance tables with smart approach
        if balance_method == 'smart':
            self._smart_balance_tables()
        else:
            self.balance_tables()
        
        return self.tables

    def _smart_balance_tables(self):
        """Balance tables while trying to maintain high happiness scores."""
        # Calculate target size for each table
        total_guests = len(self.guests)
        min_guests_per_table = total_guests // self.num_tables
        extra_guests = total_guests % self.num_tables
        
        # Identify tables that need guests (too small) and those with extra guests (too large)
        tables_too_small = []
        tables_too_large = []
        
        for i, table in enumerate(self.tables):
            target_size = min_guests_per_table + (1 if i < extra_guests else 0)
            if len(table) < target_size:
                tables_too_small.append((i, target_size - len(table)))  # (table_idx, needed_guests)
            elif len(table) > target_size:
                tables_too_large.append((i, len(table) - target_size))  # (table_idx, extra_guests)
        
        # Move guests from large to small tables based on minimizing happiness loss
        for small_idx, needed in tables_too_small:
            for _ in range(needed):
                best_happiness_loss = float('inf')
                best_move = None
                
                # Find the best guest to move from an oversized table
                for large_idx, _ in tables_too_large:
                    if len(self.tables[large_idx]) <= min_guests_per_table:
                        continue  # Skip if table no longer has extra guests
                    
                    for guest_idx, guest in enumerate(self.tables[large_idx]):
                        # Calculate happiness loss if this guest is moved
                        original = self.calculate_total_happiness([self.tables[large_idx], self.tables[small_idx]])
                        
                        # Simulate move
                        new_large = self.tables[large_idx].copy()
                        new_small = self.tables[small_idx].copy()
                        guest = new_large.pop(guest_idx)
                        new_small.append(guest)
                        
                        new_happiness = self.calculate_total_happiness([new_large, new_small])
                        happiness_loss = original - new_happiness
                        
                        if happiness_loss < best_happiness_loss:
                            best_happiness_loss = happiness_loss
                            best_move = (large_idx, guest_idx, guest)
                
                # Apply the best move if found
                if best_move:
                    large_idx, guest_idx, guest = best_move
                    self.tables[large_idx].pop(guest_idx)
                    self.tables[small_idx].append(guest)
                    
                    # Update large tables list
                    for i, (idx, count) in enumerate(tables_too_large):
                        if idx == large_idx:
                            if count > 1:
                                tables_too_large[i] = (idx, count - 1)
                            else:
                                tables_too_large.pop(i)
                            break
    
    def assign_tables_greedy(self):
        """Use a greedy approach to assign guests to tables based on relationship scores."""
        # Initialize tables
        self.tables = [[] for _ in range(self.num_tables)]
        
        # Create a copy of the guest list to work with
        remaining_guests = list(enumerate(self.guests))
        
        # Randomly select initial guests for each table
        for i in range(self.num_tables):
            if not remaining_guests:
                break
                
            idx = random.randint(0, len(remaining_guests) - 1)
            guest_idx, guest_name = remaining_guests.pop(idx)
            self.tables[i].append(guest_name)
        
        # Assign remaining guests one by one
        while remaining_guests:
            best_score = float('-inf')
            best_assignment = (0, 0)  # (table_idx, guest_idx)
            
            for table_idx, table in enumerate(self.tables):
                if len(table) >= self.table_size:
                    continue
                    
                for i, (guest_idx, guest_name) in enumerate(remaining_guests):
                    # Calculate average relationship score with everyone at the table
                    score = 0
                    if table:  # If table is not empty
                        for person in table:
                            person_idx = self.guests.index(person)
                            score += self.relationship_matrix[guest_idx][person_idx]
                        score /= len(table)
                    
                    if score > best_score:
                        best_score = score
                        best_assignment = (table_idx, i)
            
            # Assign the best guest to the best table
            table_idx, guest_pos = best_assignment
            guest_idx, guest_name = remaining_guests.pop(guest_pos)
            self.tables[table_idx].append(guest_name)
        
        return self.tables

