import time
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox

from matrix_builder import MatrixBuilder
from seat_planner import SeatPlanner
from table_visualizer import TableVisualizer
from table_optimizer import TableOptimizer

class Menu:
    def __init__(self):
        # Initialize matrix_builder during class initialization
        self.matrix_builder = None
        
    def show(self):
        self.root = tk.Tk()
        self.root.title("Seating Arrangement Menu")

        # table size
        lbl_size = tk.Label(self.root, text="Table Size:")
        lbl_size.grid(row=0, column=0, padx=10, pady=10)
        self.entry_table_size = tk.Entry(self.root)
        self.entry_table_size.insert(0, "8")
        self.entry_table_size.grid(row=0, column=1, padx=10, pady=10)

        # Remove the arrangement type combo box that only has "Random" option
        
        # display option
        lbl_display = tk.Label(self.root, text="Display Option:")
        lbl_display.grid(row=1, column=0, padx=10, pady=10)  # Adjusted row from 2 to 1
        self.combo_display = ttk.Combobox(self.root, values=["Table Visualizer"], state="readonly")
        self.combo_display.current(0)
        self.combo_display.grid(row=1, column=1, padx=10, pady=10)  # Adjusted row from 2 to 1

        # Add optimization algorithm options to your menu
        optimization_frame = ttk.LabelFrame(self.root, text="Optimization Algorithms")
        optimization_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky="ew")  # Adjusted row from 3 to 2
        
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
        self.algorithm_options_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=5, sticky="ew")  # Adjusted row from 4 to 3
        
        # Initialize matrix builder frame
        self.matrix_builder_frame = ttk.LabelFrame(self.root, text="Dataset Paths")
        self.matrix_builder_frame.grid(row=4, column=0, columnspan=2, padx=10, pady=5, sticky="ew")  # Adjusted row from 5 to 4
        
        # Add dataset path entries
        ttk.Label(self.matrix_builder_frame, text="Guest List:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.guestlist_path = tk.StringVar(value="dataset/guestlist.csv")
        ttk.Entry(self.matrix_builder_frame, textvariable=self.guestlist_path, width=30).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(self.matrix_builder_frame, text="Likes:").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.likes_path = tk.StringVar(value="dataset/likes.csv")
        ttk.Entry(self.matrix_builder_frame, textvariable=self.likes_path, width=30).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(self.matrix_builder_frame, text="Dislikes:").grid(row=2, column=0, padx=5, pady=2, sticky="w")
        self.dislikes_path = tk.StringVar(value="dataset/dislikes.csv")
        ttk.Entry(self.matrix_builder_frame, textvariable=self.dislikes_path, width=30).grid(row=2, column=1, padx=5, pady=2)
        
        # Add a note about automatic initialization
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
        ).grid(row=5, column=0, columnspan=2, padx=10, pady=5, sticky="ew")  # Adjusted row from 6 to 5
        
        # Button to compare all algorithms
        ttk.Button(
            self.root,
            text="Compare All Algorithms",
            command=self.compare_algorithms
        ).grid(row=6, column=0, columnspan=2, padx=10, pady=5, sticky="ew")  # Adjusted row from 7 to 6
        
        # Initialize algorithm options
        self.update_algorithm_options()

        # Status label to show matrix builder status
        self.status_label = ttk.Label(self.root, text="Matrix Builder Status: Not initialized")
        self.status_label.grid(row=7, column=0, columnspan=2, padx=10, pady=5)  # Adjusted row from 9 to 8

        # Result text area
        self.result_text = tk.Text(self.root, height=10, width=50)
        self.result_text.grid(row=8, column=0, columnspan=2, padx=10, pady=10)  # Adjusted row from 10 to 9
        
        self.root.mainloop()
        
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
        # Remove the arrangement_type variable since we removed the combo box
        display_option = self.combo_display.get()
        self.root.destroy()
        # for now only generate seating arrangement with Random arrangement
        self.generate_seating(table_size, "Random", display_option)  # Hardcoded "Random" as arrangement_type

    def generate_seating(self, table_size, arrangement_type, display_option):
        print(f"Selected table size: {table_size}")
        # We still print the arrangement type for debugging, but it will always be "Random"
        print(f"Selected arrangement type: {arrangement_type}")
        print(f"Selected display option: {display_option}")

        # Initialize matrix builder if not already initialized
        if self.matrix_builder is None:
            self.matrix_builder = MatrixBuilder("dataset/guestlist.csv", "dataset/likes.csv", "dataset/dislikes.csv")
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
        
        # Let's print the structure of data to debug
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
            
            # Visualize the tables with relationship lines
            visualizer = TableVisualizer(
                tables=tables,
                relationship_matrix=optimizer.relationship_matrix,
                guests=optimizer.guests
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
        """Compares all optimization algorithms and displays performance charts."""
        try:
            # Auto-initialize matrix builder if not already initialized
            if self.matrix_builder is None:
                self.initialize_matrix_builder()
                
            # Set default values for algorithm parameters
            # These will be used if the attributes don't exist yet
            self.hc_iterations = getattr(self, 'hc_iterations', tk.StringVar(value="1000"))
            self.sa_temp = getattr(self, 'sa_temp', tk.StringVar(value="100"))
            self.sa_cooling = getattr(self, 'sa_cooling', tk.StringVar(value="0.95"))
            self.sa_iterations = getattr(self, 'sa_iterations', tk.StringVar(value="1000"))
            self.ts_size = getattr(self, 'ts_size', tk.StringVar(value="10"))
            self.ts_iterations = getattr(self, 'ts_iterations', tk.StringVar(value="1000"))
            self.ga_population = getattr(self, 'ga_population', tk.StringVar(value="100"))
            self.ga_generations = getattr(self, 'ga_generations', tk.StringVar(value="200"))
            self.ga_mutation = getattr(self, 'ga_mutation', tk.StringVar(value="0.05"))
            
            # Define algorithms to compare
            algorithms = [
                ("Random", self.assign_tables_random),
                ("Hill Climbing", self.assign_tables_hill_climbing),
                ("Simulated Annealing", self.assign_tables_simulated_annealing),
                ("Tabu Search", self.assign_tables_tabu_search),
                ("Genetic Algorithm", self.assign_tables_genetic),
                ("K-means", self.assign_tables_kmeans),
                ("Greedy", self.assign_tables_greedy)
            ]
            
            # Store results
            happiness_scores = []
            execution_times = []
            names = []
            
            # Run each algorithm and record performance
            for name, algorithm in algorithms:
                try:        
                    start_time = time.time()
                    tables, happiness = algorithm()
                    end_time = time.time()
                    happiness_scores.append(happiness)
                    execution_times.append(end_time - start_time)
                    names.append(name)
                    print(f"{name}: Happiness={happiness:.2f}, Time={end_time-start_time:.2f}s")
                except Exception as e:
                    print(f"Error with {name}: {str(e)}")
                    messagebox.showwarning("Algorithm Error", f"Error running {name}: {str(e)}")
            
            # Only proceed with chart if we have results
            if not happiness_scores:
                messagebox.showerror("Error", "No successful algorithm runs to display.")
                return
                
            # Create a new window for charts
            chart_window = tk.Toplevel(self.root)
            chart_window.title("Algorithm Performance Comparison")
            chart_window.geometry("800x600")
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Plot happiness scores
            x = np.arange(len(names))
            ax1.bar(x, happiness_scores, width=0.6, align='center')
            ax1.set_xticks(x)
            ax1.set_xticklabels(names, rotation=45, ha='right')
            ax1.set_ylabel('Happiness Score')
            ax1.set_title('Happiness Score by Algorithm')
            
            # Format y-axis to show integer values
            ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
            
            # Add value labels on bars
            for i, v in enumerate(happiness_scores):
                ax1.text(i, v + 0.5, f"{v:.1f}", ha='center')
            
            # Plot execution times
            ax2.bar(x, execution_times, width=0.6, align='center', color='orange')
            ax2.set_xticks(x)
            ax2.set_xticklabels(names, rotation=45, ha='right')
            ax2.set_ylabel('Execution Time (seconds)')
            ax2.set_title('Execution Time by Algorithm')
            
            # Add value labels on bars
            for i, v in enumerate(execution_times):
                ax2.text(i, v + 0.05, f"{v:.2f}s", ha='center')
            
            # Adjust layout and embed in tkinter
            plt.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=chart_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Save results button
            ttk.Button(
                chart_window,
                text="Save Results",
                command=lambda: self.save_comparison_results(names, happiness_scores, execution_times)
            ).pack(pady=10)
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during comparison: {str(e)}")
            print(f"Comparison error: {str(e)}")
    
    def save_comparison_results(self, names, happiness_scores, execution_times):
        """Saves the comparison results to a file."""
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"algorithm_comparison_{timestamp}.txt"
            
            with open(filename, "w") as f:
                f.write("Algorithm Comparison Results\n")
                f.write("===========================\n\n")
                f.write("Algorithm\tHappiness\tTime (s)\n")
                f.write("===========================\n\n")
                for i, name in enumerate(names):
                    f.write(f"{name}\t{happiness_scores[i]:.2f}\t{execution_times[i]:.2f}\n")
            
            messagebox.showinfo("Success", f"Results saved to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")