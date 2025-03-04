import numpy as np
from sklearn.cluster import KMeans
import random
import pandas as pd
import os
from src.matrix_builder import MatrixBuilder

class TableAssignment:
    def __init__(self, relationship_matrix, guests, table_size=8):
        self.relationship_matrix = relationship_matrix
        self.guests = guests
        self.table_size = table_size
        self.num_tables = (len(guests) + table_size - 1) // table_size  # Ceiling division
        self.tables = []
        
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
    
    def generate_report(self, output_file="table_assignments.csv"):
        """Generate a CSV report of table assignments and happiness scores."""
        if not os.path.exists("output"):
            os.makedirs("output")
            
        file_path = os.path.join("output", output_file)
        
        # Calculate happiness scores
        table_scores = self.calculate_table_happiness()
        
        # Create a DataFrame for the table assignments
        max_table_size = max(len(table) for table in self.tables)
        data = {}
        
        for i, table in enumerate(self.tables):
            column_name = f"Table {i+1} (Score: {table_scores[i]})"
            # Pad with empty strings to ensure all columns have the same length
            padded_table = table + [""] * (max_table_size - len(table))
            data[column_name] = padded_table
        
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
        
        print(f"Table assignments saved to {file_path}")
        return file_path
