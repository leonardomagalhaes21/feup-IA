import numpy as np
import random
import copy
import math
from tqdm import tqdm
from sklearn.cluster import KMeans

class TableOptimizer:
    """Class containing different optimization algorithms for table assignment problems."""
    
    def __init__(self, relationship_matrix, guests, table_size=8):
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
    
    def hill_climbing(self, iterations=1000):
        """Implement hill climbing algorithm."""
        current_solution = self.generate_initial_solution()
        current_score = self.calculate_total_happiness(current_solution)
        
        for _ in tqdm(range(iterations), desc="Hill Climbing"):
            # Get a random neighbor
            neighbor_solution = self.get_neighbor_solution(current_solution)
            neighbor_score = self.calculate_total_happiness(neighbor_solution)
            
            # If the neighbor is better, move to it
            if neighbor_score > current_score:
                current_solution = neighbor_solution
                current_score = neighbor_score
        
        return current_solution, current_score
    
    def simulated_annealing(self, initial_temp=100, cooling_rate=0.95, iterations=1000):
        """Implement simulated annealing algorithm."""
        current_solution = self.generate_initial_solution()
        current_score = self.calculate_total_happiness(current_solution)
        best_solution = copy.deepcopy(current_solution)
        best_score = current_score
        
        temperature = initial_temp
        
        for _ in tqdm(range(iterations), desc="Simulated Annealing"):
            # Get a random neighbor
            neighbor_solution = self.get_neighbor_solution(current_solution)
            neighbor_score = self.calculate_total_happiness(neighbor_solution)
            
            # Calculate acceptance probability
            delta = neighbor_score - current_score
            acceptance_probability = 1.0 if delta > 0 else math.exp(delta / temperature)
            
            # Decide whether to accept the neighbor
            if random.random() < acceptance_probability:
                current_solution = neighbor_solution
                current_score = neighbor_score
            
            # Update best solution if current is better
            if current_score > best_score:
                best_solution = copy.deepcopy(current_solution)
                best_score = current_score
            
            # Cool down the temperature
            temperature *= cooling_rate
        
        return best_solution, best_score
    
    def tabu_search(self, tabu_size=10, iterations=1000):
        """Implement tabu search algorithm."""
        current_solution = self.generate_initial_solution()
        current_score = self.calculate_total_happiness(current_solution)
        best_solution = copy.deepcopy(current_solution)
        best_score = current_score
        
        tabu_list = []  # Store hash values of recent solutions
        
        for _ in tqdm(range(iterations), desc="Tabu Search"):
            # Generate multiple neighbors and select the best non-tabu neighbor
            best_neighbor = None
            best_neighbor_score = float('-inf')
            
            # Generate several neighbors
            for _ in range(5):  # Consider 5 neighbors
                neighbor = self.get_neighbor_solution(current_solution)
                neighbor_hash = hash(str(neighbor))
                neighbor_score = self.calculate_total_happiness(neighbor)
                
                # Select best non-tabu neighbor or if aspiration criterion is met
                if neighbor_hash not in tabu_list and neighbor_score > best_neighbor_score:
                    best_neighbor = neighbor
                    best_neighbor_score = neighbor_score
                
                # Aspiration criterion: accept tabu solution if better than best so far
                elif neighbor_score > best_score:
                    best_neighbor = neighbor
                    best_neighbor_score = neighbor_score
            
            # Move to the selected neighbor
            if best_neighbor:
                current_solution = best_neighbor
                current_score = best_neighbor_score
                
                # Add to tabu list
                tabu_list.append(hash(str(current_solution)))
                if len(tabu_list) > tabu_size:
                    tabu_list.pop(0)  # Remove oldest entry
                
                # Update best solution if needed
                if current_score > best_score:
                    best_solution = copy.deepcopy(current_solution)
                    best_score = current_score
        
        return best_solution, best_score
    
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
        
        # Initialize population with random solutions
        population = []
        for _ in range(population_size):
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
        
    def assign_tables_kmeans(self):
        """Use K-means clustering to assign guests to tables based on relationship scores."""
        # Use K-means to cluster guests based on their relationship scores
        kmeans = KMeans(n_clusters=self.num_tables, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(self.relationship_matrix)
        
        # Initialize tables
        self.tables = [[] for _ in range(self.num_tables)]
        
        # Assign guests to tables based on their cluster
        for i, cluster in enumerate(clusters):
            self.tables[cluster].append(self.guests[i])
        
        # Balance tables if needed
        self.balance_tables()
        
        return self.tables
    
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
    
    def assign_tables_hill_climbing(self, iterations=1000):
        """Use hill climbing to optimize table assignments."""
        tables, _ = self.hill_climbing(iterations)
        self.tables = tables
        return self.tables
    
    def assign_tables_simulated_annealing(self, initial_temp=100, cooling_rate=0.95, iterations=1000):
        """Use simulated annealing to optimize table assignments."""
        tables, _ = self.simulated_annealing(initial_temp, cooling_rate, iterations)
        self.tables = tables
        return self.tables
    
    def assign_tables_tabu_search(self, tabu_size=10, iterations=1000):
        """Use tabu search to optimize table assignments."""
        tables, _ = self.tabu_search(tabu_size, iterations)
        self.tables = tables
        return self.tables
    
    def assign_tables_genetic(self, population_size=100, generations=200, mutation_rate=0.05):
        """Use genetic algorithm to optimize table assignments."""
        tables, _ = self.genetic_algorithm(
            population_size=population_size, 
            generations=generations, 
            mutation_rate=mutation_rate,
            elite_size=5
        )
        self.tables = tables
        return self.tables
    
    def balance_tables(self):
        """Ensure all tables have roughly the same number of guests."""
        # Calculate ideal table size
        total_guests = len(self.guests)
        min_guests_per_table = total_guests // self.num_tables
        
        # Count how many tables need an extra person
        extra_guests = total_guests % self.num_tables
        
        # Adjust table sizes
        for i in range(self.num_tables):
            target_size = min_guests_per_table + (1 if i < extra_guests else 0)
            
            # Move guests from oversized tables to undersized ones
            while len(self.tables[i]) > target_size:
                # Find a table with fewer than the target number of guests
                for j in range(self.num_tables):
                    if len(self.tables[j]) < min_guests_per_table + (1 if j < extra_guests else 0):
                        guest = self.tables[i].pop()
                        self.tables[j].append(guest)
                        break
    
    def calculate_table_happiness(self):
        """Calculate the total happiness score for each table."""
        table_scores = []
        
        for table in self.tables:
            table_score = 0
            guest_indices = [self.guests.index(guest) for guest in table]
            
            # Calculate pairwise relationship scores
            for i, idx1 in enumerate(guest_indices):
                for idx2 in guest_indices[i+1:]:
                    table_score += self.relationship_matrix[idx1][idx2]
            
            table_scores.append(table_score)
        
        return table_scores
