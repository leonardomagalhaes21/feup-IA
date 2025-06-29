import time
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox

from matrix_builder import MatrixBuilder
from table_visualizer import TableVisualizer
from table_optimizer import TableOptimizer

class Menu:
    def __init__(self):
        # Initialize matrix_builder during class initialization
        self.matrix_builder = None
        
    def show(self):
        self.root = tk.Tk()
        self.root.title("Seating Arrangement Menu")

        # Set protocol to handle window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # table size
        lbl_size = tk.Label(self.root, text="Table Size:")
        lbl_size.grid(row=0, column=0, padx=10, pady=10)
        self.entry_table_size = tk.Entry(self.root)
        self.entry_table_size.insert(0, "8")
        self.entry_table_size.grid(row=0, column=1, padx=10, pady=10)
        
        # Display option
        lbl_display = tk.Label(self.root, text="Display Option:")
        lbl_display.grid(row=1, column=0, padx=10, pady=10)
        self.combo_display = ttk.Combobox(self.root, values=["Table Visualizer"], state="readonly")
        self.combo_display.current(0)
        self.combo_display.grid(row=1, column=1, padx=10, pady=10)

        # Dataset size selection
        dataset_size_frame = ttk.LabelFrame(self.root, text="Dataset Size")
        dataset_size_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        
        # Dataset size selection using radio buttons
        self.dataset_size_var = tk.StringVar(value="small")
        dataset_sizes = [("Small", "small"), ("Medium", "medium"), ("Large", "large")]
        
        for i, (text, value) in enumerate(dataset_sizes):
            ttk.Radiobutton(
                dataset_size_frame,
                text=text,
                value=value,
                variable=self.dataset_size_var,
                command=self.update_dataset_paths
            ).pack(side=tk.LEFT, anchor="w", padx=20, pady=2)

        # Optimization algorithm options to menu
        optimization_frame = ttk.LabelFrame(self.root, text="Optimization Algorithms")
        optimization_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        
        algorithms = [
            ("Random", self.assign_tables_random),
            ("Hill Climbing", self.assign_tables_hill_climbing),
            ("Simulated Annealing", self.assign_tables_simulated_annealing),
            ("Tabu Search", self.assign_tables_tabu_search),
            ("Genetic Algorithm", self.assign_tables_genetic),
            ("K-means Clustering", self.assign_tables_kmeans),
            ("Greedy", self.assign_tables_greedy)
        ]
        
        # Create radio buttons for algorithm selection
        self.algorithm_var = tk.StringVar(value="Random")
        for text, command in algorithms:
            ttk.Radiobutton(
                optimization_frame,
                text=text,
                value=text,
                variable=self.algorithm_var,
                command=self.update_algorithm_options
            ).pack(anchor="w", padx=10, pady=2)
        
        # Create a frame for algorithm specific parameters
        self.algorithm_options_frame = ttk.LabelFrame(self.root, text="Algorithm Parameters")
        self.algorithm_options_frame.grid(row=4, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        
        # Initialize matrix builder frame
        self.matrix_builder_frame = ttk.LabelFrame(self.root, text="Dataset Paths")
        self.matrix_builder_frame.grid(row=5, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        
        # Add dataset path entries
        ttk.Label(self.matrix_builder_frame, text="Guest List:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.guestlist_path = tk.StringVar(value="../dataset/small/guestlist_small.csv")
        ttk.Entry(self.matrix_builder_frame, textvariable=self.guestlist_path, width=30).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(self.matrix_builder_frame, text="Likes:").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.likes_path = tk.StringVar(value="../dataset/small/likes_small.csv")
        ttk.Entry(self.matrix_builder_frame, textvariable=self.likes_path, width=30).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(self.matrix_builder_frame, text="Dislikes:").grid(row=2, column=0, padx=5, pady=2, sticky="w")
        self.dislikes_path = tk.StringVar(value="../dataset/small/dislikes_small.csv")
        ttk.Entry(self.matrix_builder_frame, textvariable=self.dislikes_path, width=30).grid(row=2, column=1, padx=5, pady=2)
        
        # Note about automatic initialization
        note_label = ttk.Label(
            self.matrix_builder_frame, 
            text="Note: Matrix will be automatically initialized when running algorithms.",
            font=("", 8, "italic")
        )
        note_label.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        
        # Button to run the selected algorithm
        ttk.Button(
            self.root,
            text="Optimize Tables",
            command=self.run_selected_algorithm
        ).grid(row=6, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        
        # Button to compare all algorithms
        ttk.Button(
            self.root,
            text="Compare All Algorithms",
            command=self.compare_algorithms
        ).grid(row=7, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        
        # Initialize algorithm options
        self.update_algorithm_options()

        # Status label to show matrix builder status
        self.status_label = ttk.Label(self.root, text="Matrix Builder Status: Not initialized")
        self.status_label.grid(row=8, column=0, columnspan=2, padx=10, pady=5)

        # Result text area
        self.result_text = tk.Text(self.root, height=10, width=50)
        self.result_text.grid(row=9, column=0, columnspan=2, padx=10, pady=10)
        
        self.root.mainloop()
        
    def on_close(self):
        """Handles the menu window close event."""
        plt.close('all')
        self.root.destroy()
        sys.exit(0)
        
    def update_dataset_paths(self):
        """Update dataset paths based on selected dataset size"""
        size = self.dataset_size_var.get()
        base_path = f"../dataset/{size}"
        
        self.guestlist_path.set(f"{base_path}/guestlist_{size}.csv")
        self.likes_path.set(f"{base_path}/likes_{size}.csv")
        self.dislikes_path.set(f"{base_path}/dislikes_{size}.csv")
        
        # Reset the matrix builder when dataset changes
        self.matrix_builder = None
        self.status_label.config(text="Matrix Builder Status: Reset (dataset changed)")
        
    def initialize_matrix_builder(self):
        """Initialize the matrix builder with the provided dataset paths"""
        try:
            guestlist_path = self.guestlist_path.get()
            likes_path = self.likes_path.get()
            dislikes_path = self.dislikes_path.get()
            
            self.matrix_builder = MatrixBuilder(guestlist_path, likes_path, dislikes_path)
            self.matrix_builder.build_matrix()
            
            self.status_label.config(text="Matrix Builder Status: Initialized successfully")
            messagebox.showinfo("Success", "Matrix builder initialized successfully!")
        except Exception as e:
            self.status_label.config(text=f"Matrix Builder Status: Error - {str(e)}")
            messagebox.showerror("Error", f"Failed to initialize matrix builder: {str(e)}")

    def on_generate(self):
        try:
            table_size = int(self.entry_table_size.get())
        except ValueError:
            table_size = 8
        display_option = self.combo_display.get()
        self.root.destroy()
        self.generate_seating(table_size, "Random", display_option)  # Hardcoded "Random" as arrangement_type

    def generate_seating(self, table_size, arrangement_type, display_option):
        print(f"Selected table size: {table_size}")
        print(f"Selected arrangement type: {arrangement_type}")
        print(f"Selected display option: {display_option}")

        # Initialize matrix builder if not already initialized
        if self.matrix_builder is None:
            # Use the paths based on the selected dataset size
            self.matrix_builder = MatrixBuilder(
                self.guestlist_path.get(), 
                self.likes_path.get(), 
                self.dislikes_path.get()
            )
            self.matrix_builder.build_matrix()
            
        data = self.matrix_builder.get_matrix_data()

        planner = SeatPlanner(data, table_size)
        
        if arrangement_type == "Random":
            tables = planner.get_random_arrangement()
        else:
            print("Invalid arrangement type. Using random arrangement.")
            tables = planner.get_random_arrangement()

        self.matrix_builder.save_matrix_to_csv("relationship_matrix.csv")

        relationship_matrix = None
        guests = None
        
        if "relationship_matrix" in data:
            relationship_matrix = data["relationship_matrix"]
        elif "matrix" in data:
            relationship_matrix = data["matrix"]
            
        if "guests" in data:
            guests = data["guests"]
        elif "names" in data:
            guests = data["names"]

        if display_option == "Table Visualizer":
            visualizer = TableVisualizer(tables, relationship_matrix, guests)
            visualizer.show()
        else:
            print("Unknown display option.")

    def update_algorithm_options(self):
        """Updates the parameter options based on the selected algorithm."""
        # Clear existing options
        for widget in self.algorithm_options_frame.winfo_children():
            widget.destroy()
        
        selected_algorithm = self.algorithm_var.get()
        
        if selected_algorithm == "Hill Climbing":
            # Hill Climbing parameters
            ttk.Label(self.algorithm_options_frame, text="Iterations:").grid(row=0, column=0, padx=5, pady=2)
            self.hc_iterations = tk.StringVar(value="1000")
            ttk.Entry(self.algorithm_options_frame, textvariable=self.hc_iterations, width=10).grid(row=0, column=1, padx=5, pady=2)
            
        elif selected_algorithm == "Simulated Annealing":
            # Simulated Annealing parameters
            ttk.Label(self.algorithm_options_frame, text="Initial Temperature:").grid(row=0, column=0, padx=5, pady=2)
            self.sa_temp = tk.StringVar(value="100")
            ttk.Entry(self.algorithm_options_frame, textvariable=self.sa_temp, width=10).grid(row=0, column=1, padx=5, pady=2)
            
            ttk.Label(self.algorithm_options_frame, text="Cooling Rate:").grid(row=1, column=0, padx=5, pady=2)
            self.sa_cooling = tk.StringVar(value="0.95")
            ttk.Entry(self.algorithm_options_frame, textvariable=self.sa_cooling, width=10).grid(row=1, column=1, padx=5, pady=2)
            
            ttk.Label(self.algorithm_options_frame, text="Iterations:").grid(row=2, column=0, padx=5, pady=2)
            self.sa_iterations = tk.StringVar(value="1000")
            ttk.Entry(self.algorithm_options_frame, textvariable=self.sa_iterations, width=10).grid(row=2, column=1, padx=5, pady=2)
            
        elif selected_algorithm == "Tabu Search":
            # Tabu Search parameters
            ttk.Label(self.algorithm_options_frame, text="Tabu Size:").grid(row=0, column=0, padx=5, pady=2)
            self.ts_size = tk.StringVar(value="10")
            ttk.Entry(self.algorithm_options_frame, textvariable=self.ts_size, width=10).grid(row=0, column=1, padx=5, pady=2)
            
            ttk.Label(self.algorithm_options_frame, text="Iterations:").grid(row=1, column=0, padx=5, pady=2)
            self.ts_iterations = tk.StringVar(value="1000")
            ttk.Entry(self.algorithm_options_frame, textvariable=self.ts_iterations, width=10).grid(row=1, column=1, padx=5, pady=2)
            
        elif selected_algorithm == "Genetic Algorithm":
            # Genetic Algorithm parameters
            ttk.Label(self.algorithm_options_frame, text="Population Size:").grid(row=0, column=0, padx=5, pady=2)
            self.ga_population = tk.StringVar(value="100")
            ttk.Entry(self.algorithm_options_frame, textvariable=self.ga_population, width=10).grid(row=0, column=1, padx=5, pady=2)
            
            ttk.Label(self.algorithm_options_frame, text="Generations:").grid(row=1, column=0, padx=5, pady=2)
            self.ga_generations = tk.StringVar(value="200")
            ttk.Entry(self.algorithm_options_frame, textvariable=self.ga_generations, width=10).grid(row=1, column=1, padx=5, pady=2)
            
            ttk.Label(self.algorithm_options_frame, text="Mutation Rate:").grid(row=2, column=0, padx=5, pady=2)
            self.ga_mutation = tk.StringVar(value="0.05")
            ttk.Entry(self.algorithm_options_frame, textvariable=self.ga_mutation, width=10).grid(row=2, column=1, padx=5, pady=2)
    
    def get_optimizer(self):
        """Creates and returns a TableOptimizer instance with current data."""
        # Check if matrix_builder is initialized
        if self.matrix_builder is None:
            raise ValueError("Matrix Builder not initialized. Please click 'Initialize Matrix Builder' first.")
            
        # Get the matrix data
        data = self.matrix_builder.get_matrix_data()
        
        # Print the structure of data to debug
        print("Data keys:", data.keys())
        
        # Access the relationship matrix
        if "relationship_matrix" in data:
            relationship_matrix = data["relationship_matrix"]
        elif "matrix" in data:
            relationship_matrix = data["matrix"]
        else:
            # As a last resort, assume data itself is the matrix
            relationship_matrix = data
        
        # Get guest names - prioritize "guests" key which is more likely to contain actual names
        if "guests" in data:
            guests = data["guests"]
        elif "names" in data:
            guests = data["names"]
        elif hasattr(self.matrix_builder, "guests") and self.matrix_builder.guests:
            guests = self.matrix_builder.guests
        elif hasattr(self.matrix_builder, "get_guests") and callable(getattr(self.matrix_builder, "get_guests")):
            guests = self.matrix_builder.get_guests()
        else:
            # Try to access the guests directly from the MatrixBuilder instance's properties
            matrix_data = self.matrix_builder.get_matrix_data()
            if isinstance(matrix_data, dict) and "guests" in matrix_data:
                guests = matrix_data["guests"]
            else:
                # Last resort: create generic guest names
                guests = [f"Guest {i+1}" for i in range(len(relationship_matrix))]
        
        # Print what we found to help with debugging
        print(f"Guest names sample: {guests[:5]}...")
        
        table_size = int(self.entry_table_size.get()) if hasattr(self, 'entry_table_size') else 8
        
        print(f"Relationship matrix shape: {np.array(relationship_matrix).shape}")
        print(f"Number of guests: {len(guests)}")
        
        return TableOptimizer(relationship_matrix, guests, table_size)
    
    def calculate_table_happiness(self, optimizer, tables):
        """
        Calculate happiness score for each table.
        Returns a dictionary with table indices as keys and happiness scores as values.
        """
        table_scores = {}
        
        for i, table in enumerate(tables):
            # Skip empty tables
            if not table:
                continue
                
            score = 0
            # Get indices of guests at this table
            guest_indices = [optimizer.guests.index(guest) for guest in table]
            
            # Calculate pairwise happiness for this table
            for idx1_pos, idx1 in enumerate(guest_indices):
                for idx2 in guest_indices[idx1_pos+1:]:
                    score += optimizer.relationship_matrix[idx1][idx2]
                    
            table_scores[i] = score
            
        return table_scores

    def run_selected_algorithm(self):
        """Runs the selected algorithm with the specified parameters."""
        try:
            # Auto-initialize matrix builder if not already initialized
            if self.matrix_builder is None:
                self.initialize_matrix_builder()
                
            algorithm = self.algorithm_var.get()
            optimizer = self.get_optimizer()
            
            start_time = time.time()
            
            if algorithm == "Random":
                tables = optimizer.generate_initial_solution()
                happiness = optimizer.calculate_total_happiness(tables)
            elif algorithm == "Hill Climbing":
                iterations = int(self.hc_iterations.get())
                tables, happiness = optimizer.hill_climbing(iterations)
            elif algorithm == "Simulated Annealing":
                initial_temp = float(self.sa_temp.get())
                cooling_rate = float(self.sa_cooling.get())
                iterations = int(self.sa_iterations.get())
                tables, happiness = optimizer.simulated_annealing(initial_temp, cooling_rate, iterations)
            elif algorithm == "Tabu Search":
                tabu_size = int(self.ts_size.get())
                iterations = int(self.ts_iterations.get())
                tables, happiness = optimizer.tabu_search(tabu_size, iterations)
            elif algorithm == "Genetic Algorithm":
                population = int(self.ga_population.get())
                generations = int(self.ga_generations.get())
                mutation_rate = float(self.ga_mutation.get())
                tables, happiness = optimizer.genetic_algorithm(population, generations, mutation_rate)
            elif algorithm == "K-means Clustering":
                tables = optimizer.assign_tables_kmeans()
                happiness = optimizer.calculate_total_happiness(tables)
            elif algorithm == "Greedy":
                tables = optimizer.assign_tables_greedy()
                happiness = optimizer.calculate_total_happiness(tables)
            
            end_time = time.time()
            
            # Display results
            self.display_results(tables, happiness, end_time - start_time)
            
            # Calculate individual table happiness scores
            table_scores = self.calculate_table_happiness(optimizer, tables)
            
            # Visualize the tables with relationship lines and table scores
            visualizer = TableVisualizer(
                tables=tables,
                relationship_matrix=optimizer.relationship_matrix,
                guests=optimizer.guests,
                table_scores=table_scores  # Pass table scores to visualizer
            )
            visualizer.show()
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            print(f"Error in run_selected_algorithm: {str(e)}")
    
    def assign_tables_random(self):
        """Simple wrapper for random table assignment."""
        if self.matrix_builder is None:
            raise ValueError("Matrix Builder not initialized. Please initialize it first.")
            
        optimizer = self.get_optimizer()
        tables = optimizer.generate_initial_solution()
        happiness = optimizer.calculate_total_happiness(tables)
        return tables, happiness
    
    def assign_tables_hill_climbing(self):
        """Wrapper for Hill Climbing algorithm."""
        if self.matrix_builder is None:
            raise ValueError("Matrix Builder not initialized. Please initialize it first.")
            
        optimizer = self.get_optimizer()
        iterations = int(self.hc_iterations.get())
        tables, score = optimizer.hill_climbing(iterations)
        return tables, score
    
    def assign_tables_simulated_annealing(self):
        """Wrapper for Simulated Annealing algorithm."""
        optimizer = self.get_optimizer()
        initial_temp = float(self.sa_temp.get())
        cooling_rate = float(self.sa_cooling.get())
        iterations = int(self.sa_iterations.get())
        tables, score = optimizer.simulated_annealing(initial_temp, cooling_rate, iterations)
        return tables, score
    
    def assign_tables_tabu_search(self):
        """Wrapper for Tabu Search algorithm."""
        optimizer = self.get_optimizer()
        tabu_size = int(self.ts_size.get())
        iterations = int(self.ts_iterations.get())
        tables, score = optimizer.tabu_search(tabu_size, iterations)
        return tables, score
    
    def assign_tables_genetic(self):
        """Wrapper for Genetic Algorithm."""
        optimizer = self.get_optimizer()
        population = int(self.ga_population.get())
        generations = int(self.ga_generations.get())
        mutation_rate = float(self.ga_mutation.get())
        tables, score = optimizer.genetic_algorithm(population, generations, mutation_rate)
        return tables, score
    
    def assign_tables_kmeans(self):
        """Wrapper for K-means Clustering algorithm."""
        optimizer = self.get_optimizer()
        tables = optimizer.assign_tables_kmeans()
        happiness = optimizer.calculate_total_happiness(tables)
        return tables, happiness
    
    def assign_tables_greedy(self):
        """Wrapper for Greedy algorithm."""
        optimizer = self.get_optimizer()
        tables = optimizer.assign_tables_greedy()
        happiness = optimizer.calculate_total_happiness(tables)
        return tables, happiness
    
    def display_results(self, tables, happiness, execution_time):
        """Displays the optimization results."""
        # Clear previous results
        self.result_text.delete(1.0, tk.END)
        
        # Display new results
        result_text = f"Total Happiness: {happiness:.2f}\n"
        result_text += f"Execution Time: {execution_time:.2f} seconds\n\n"
        
        # Display table assignments
        for i, table in enumerate(tables):
            result_text += f"Table {i+1}: {', '.join(table)}\n"
        
        # Insert into result text widget
        self.result_text.insert(tk.END, result_text)
    
    def compare_algorithms(self):
        """Compares algorithm performance using data from saved result files."""
        try:      
            dataset_size = self.dataset_size_var.get()
            
            # Check if results directory exists
            results_dir = os.path.join("results", dataset_size)
            if not os.path.exists(results_dir):
                messagebox.showerror("Error", f"No results found for {dataset_size} dataset. Please run algorithms first.")
                return
            
            # Check if comparison file exists
            comparison_file = os.path.join(results_dir, "algorithm_comparison.csv")
            if not os.path.exists(comparison_file):
                messagebox.showerror("Error", f"Comparison data not found for {dataset_size} dataset. Please run algorithms first.")
                return
                
            # Update loading status
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Preparing visualization...")
            self.root.update()
                
            # Load comparison data
            df = pd.read_csv(comparison_file)
            
            # Create a new window
            chart_window = tk.Toplevel(self.root)
            chart_window.title(f"Algorithm Comparison - {dataset_size.capitalize()} Dataset")
            
            # Platform-specific window maximization
            import platform
            if platform.system() == "Windows":
                chart_window.state('zoomed')  # For Windows
            else:
                chart_window.attributes('-zoomed', True)  # For Linux/Unix
            
            # Function to handle window close event
            def on_window_close():
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, f"Comparison window for {dataset_size} dataset closed.")
                chart_window.destroy()
                
            # Set the protocol for window close event
            chart_window.protocol("WM_DELETE_WINDOW", on_window_close)
            
            # Create notebook for tabs
            notebook = ttk.Notebook(chart_window)
            notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # PAGE 1: Happiness Score and Execution Time
            page1 = ttk.Frame(notebook)
            notebook.add(page1, text="Happiness & Time")
            
            fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            df_sorted = df.sort_values('Happiness Score', ascending=False)
            
            x1 = np.arange(len(df_sorted))
            bars1 = ax1.bar(x1, df_sorted['Happiness Score'], color='blue')
            ax1.set_xticks(x1)
            ax1.set_xticklabels(df_sorted['Algorithm'], rotation=45, ha='right')
            ax1.set_ylabel('Happiness Score')
            ax1.set_title('Happiness Score by Algorithm')
            
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                        f'{height:.0f}', ha='center', va='bottom')
            
            bars2 = ax2.bar(x1, df_sorted['Execution Time (s)'], color='orange')
            ax2.set_xticks(x1)
            ax2.set_xticklabels(df_sorted['Algorithm'], rotation=45, ha='right')
            ax2.set_ylabel('Execution Time (seconds)')
            ax2.set_title('Execution Time by Algorithm')
            
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.2f}s', ha='center', va='bottom')
            
            plt.tight_layout()
            
            canvas1 = FigureCanvasTkAgg(fig1, master=page1)
            canvas1.draw()
            canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # PAGE 2: Efficiency and Trade-off
            page2 = ttk.Frame(notebook)
            notebook.add(page2, text="Efficiency & Trade-off")
            
            df_no_random = df[df['Algorithm'] != 'Random'].copy()
            
            tradeoff_file = os.path.join(results_dir, "performance_tradeoff.csv")
            if os.path.exists(tradeoff_file):
                df_tradeoff = pd.read_csv(tradeoff_file)
                
                fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(10, 8))
                
                df_tradeoff = df_tradeoff.sort_values('Efficiency', ascending=False)
                
                # Plot efficiency (happiness per second)
                x2 = np.arange(len(df_tradeoff))
                bars3 = ax3.bar(x2, df_tradeoff['Efficiency'], color='green')
                ax3.set_xticks(x2)
                ax3.set_xticklabels(df_tradeoff['Algorithm'], rotation=45, ha='right')
                ax3.set_ylabel('Efficiency (Happiness/Second)')
                ax3.set_title('Algorithm Efficiency')
                
                for bar in bars3:
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height + 2,
                            f'{height:.1f}', ha='center', va='bottom')
                
                ax4.scatter(df_tradeoff['Execution Time (s)'], df_tradeoff['Happiness Score'], 
                           s=100, c='purple', alpha=0.7)
                
                for i, algo in enumerate(df_tradeoff['Algorithm']):
                    ax4.annotate(algo, 
                                (df_tradeoff['Execution Time (s)'].iloc[i], 
                                 df_tradeoff['Happiness Score'].iloc[i]),
                                textcoords="offset points", 
                                xytext=(0,10), 
                                ha='center')
                
                ax4.set_xlabel('Execution Time (seconds)')
                ax4.set_ylabel('Happiness Score')
                ax4.set_title('Performance Trade-off: Happiness vs. Execution Time')
                ax4.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # Embed in frame
                canvas2 = FigureCanvasTkAgg(fig2, master=page2)
                canvas2.draw()
                canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            else:
                ttk.Label(page2, text="Trade-off data not found").pack(pady=20)
            
            # PAGES 3+: Algorithm Detail Pages
            for algo in df_no_random['Algorithm']:
                if algo in ['Random', 'Greedy']:  # Skip these algorithms
                    continue
                
                algo_page = ttk.Frame(notebook)
                notebook.add(algo_page, text=f"{algo} Details")
                
                algo_details_file = os.path.join(results_dir, 
                                              f"{algo.replace(' ', '_').lower()}_details.csv")
                
                if os.path.exists(algo_details_file):
                    df_algo = pd.read_csv(algo_details_file)
                    
                    df_algo_filtered = df_algo[df_algo['Table Number'] != 'Total'].copy()
                    
                    df_algo_filtered['Table_Num'] = df_algo_filtered['Table Number'].str.extract(r'Table (\d+)', expand=False).astype(int)
                    df_algo_filtered = df_algo_filtered.sort_values('Table_Num')
                    
                    # Create figure with a horizontal layout: left plot and two right plots (top and bottom)
                    fig_algo = plt.figure(figsize=(16, 10))
                    
                    # Create special grid layout with different widths and heights for panels
                    gs = fig_algo.add_gridspec(2, 2, width_ratios=[1, 1.5], height_ratios=[1, 1.3])
                    
                    # 1. Guest distribution (left)
                    ax_guests = fig_algo.add_subplot(gs[:, 0])  # Spans both rows in first column
                    bars_guests = ax_guests.bar(df_algo_filtered['Table_Num'], df_algo_filtered['Guest Count'], color='blue')
                    ax_guests.set_xlabel('Table Number')
                    ax_guests.set_ylabel('Number of Guests')
                    ax_guests.set_title(f'Guest Distribution - {algo}')
                    ax_guests.set_xticks(df_algo_filtered['Table_Num'])
                    ax_guests.set_xticklabels([f'{x}' for x in df_algo_filtered['Table_Num']], rotation=90)
                    
                    # Add value labels
                    for bar in bars_guests:
                        height = bar.get_height()
                        ax_guests.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                    f'{height:.0f}', ha='center', va='bottom')
                    
                    # 2. Happiness score per table (top right)
                    ax_score = fig_algo.add_subplot(gs[0, 1])  # First row, second column
                    bars_score = ax_score.bar(df_algo_filtered['Table_Num'], df_algo_filtered['Happiness Score'], color='green')
                    ax_score.set_xlabel('Table Number')
                    ax_score.set_ylabel('Happiness Score')
                    ax_score.set_title(f'Happiness Score by Table - {algo}')
                    ax_score.set_xticks(df_algo_filtered['Table_Num'])
                    ax_score.set_xticklabels([f'{x}' for x in df_algo_filtered['Table_Num']], rotation=90)
                    
                    # Add value labels
                    for bar in bars_score:
                        height = bar.get_height()
                        ax_score.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                    f'{height:.0f}', ha='center', va='bottom')
                    
                    # 3. Contribution to total happiness as a larger pie chart (bottom right)
                    ax_contrib = fig_algo.add_subplot(gs[1, 1])  # Second row, second column
                    
                    # Create simplified table labels
                    contribution_values = df_algo_filtered['Contribution (%)'].values
                    table_numbers = df_algo_filtered['Table_Num'].values
                    
                    # Handle label placement for better readability
                    if len(table_numbers) > 8:
                        # For many tables, show values directly on pie slices without legend
                        wedges, texts, autotexts = ax_contrib.pie(
                            contribution_values, 
                            labels=[f'Table {num}' for num in table_numbers],
                            autopct='%1.1f%%',
                            startangle=90,
                            shadow=False,
                            explode=None,
                            wedgeprops={'edgecolor': 'white', 'linewidth': 1},
                            textprops={'fontsize': 11}
                        )
                        # Make percentage labels visible
                        for autotext in autotexts:
                            autotext.set_fontsize(10)
                    else:
                        # For fewer tables, we can show labels directly
                        wedges, texts, autotexts = ax_contrib.pie(
                            contribution_values, 
                            labels=[f'Table {num}' for num in table_numbers], 
                            labeldistance=1.1,
                            autopct='%1.1f%%',
                            pctdistance=0.75,
                            startangle=90, 
                            shadow=False,
                            explode=None,
                            wedgeprops={'edgecolor': 'white', 'linewidth': 1},
                            textprops={'fontsize': 11}
                        )
                        # Make percentage labels visible
                        for autotext in autotexts:
                            autotext.set_fontsize(10)
                    
                    ax_contrib.set_title(f'Contribution to Total Happiness - {algo}', fontsize=12)
                    ax_contrib.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle
                    
                    # Adjust spacing between subplots
                    plt.tight_layout(h_pad=1.5, w_pad=1.5)
                    
                    # Embed in frame
                    canvas_algo = FigureCanvasTkAgg(fig_algo, master=algo_page)
                    canvas_algo.draw()
                    canvas_algo.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                else:
                    ttk.Label(algo_page, text=f"Details data not found for {algo}").pack(pady=20)
            
            # Add dataset size selector
            control_frame = ttk.Frame(chart_window)
            control_frame.pack(fill=tk.X, padx=10, pady=5)
            
            ttk.Label(control_frame, text="Dataset Size:").pack(side=tk.LEFT, padx=5, pady=5)
            
            # Check which dataset sizes have results
            available_sizes = []
            for size in ["small", "medium", "large"]:
                if os.path.exists(os.path.join("results", size, "algorithm_comparison.csv")):
                    available_sizes.append(size)
            
            size_var = tk.StringVar(value=dataset_size)
            size_combo = ttk.Combobox(control_frame, textvariable=size_var, 
                                      values=available_sizes, state="readonly")
            size_combo.pack(side=tk.LEFT, padx=5, pady=5)
            
            # Function to refresh charts when dataset size changes
            def on_size_change(event=None):
                self.dataset_size_var.set(size_var.get())
                chart_window.destroy()
                self.compare_algorithms()
                
            size_combo.bind("<<ComboboxSelected>>", on_size_change)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            print(f"Error in compare_algorithms: {str(e)}")

