# feup-IA-1 -> Seating Arrangement Optimizer

A tool for optimizing seating arrangements based on guest relationships. This application implements various optimization algorithms to find the best seating arrangement that maximizes the overall happiness of guests.

## Project Structure

- **src/**: Contains all source code files
  - **main.py**: Entry point of the application
  - **menu.py**: GUI interface for the application
  - **matrix_builder.py**: Builds relationship matrices from input data
  - **seat_planner.py**: Handles planning the seating arrangements
  - **table_optimizer.py**: Implements various optimization algorithms
  - **table_visualizer.py**: Visualizes the seating arrangements
  
- **dataset/**: Contains input data files organized by size
  - **small/**: Small-scale test datasets
    - **guestlist_small.csv**: List of guests (smaller dataset)
    - **likes_small.csv**: Preferences between guests
    - **dislikes_small.csv**: Conflicts between guests
  - **medium/**: Medium-scale datasets
    - **guestlist_medium.csv**: List of guests (medium dataset)
    - **likes_medium.csv**: Preferences between guests
    - **dislikes_medium.csv**: Conflicts between guests
  - **large/**: Large-scale datasets
    - **guestlist_large.csv**: List of guests (larger dataset)
    - **likes_large.csv**: Preferences between guests
    - **dislikes_large.csv**: Conflicts between guests
  
- **results/**: Output directory for optimization results
  - **algorithm_comparison.txt**: Performance comparison of different algorithms
  - **solution_*.txt**: Detailed solution files for each algorithm

## Requirements

- Python 3.8 or higher
- Required packages:
  - matplotlib
  - numpy
  - tkinter
  - scikit-learn
  - tqdm

## Installation

1. Clone this repository:
   ```
   git clone git@github.com:leonardomagalhaes21/feup-IA-1.git

   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```
   
   Or install packages individually:
   ```
   pip install numpy matplotlib scikit-learn tqdm
   ```

## Running the Program

1. Navigate to the project directory:
   ```
   cd ../src/
   ```

2. Run the main script:
   ```
   python main.py
   ```

## How to Use

### Dataset Structure

Before running the program, prepare your dataset files in the `dataset/` folder:

1. **guestlist_*.csv**: One guest name per line
2. **likes_*.csv**: Each line contains two names separated by a comma, indicating a positive relationship
3. **dislikes_*.csv**: Each line contains two names separated by a comma, indicating a negative relationship

The application supports multiple dataset sizes (small, medium, large) which can be selected from the interface.


### User Interface

1. **Table Size**: Enter the maximum number of people per table.

2. **Display Option**: Select how to visualize the results.

3. **Dataset Size**: Select the size of the dataset to use (small, medium, large).

4. **Optimization Algorithms**: Select one of the available algorithms:
- **Random**: Random assignment (baseline)
- **Hill Climbing**: Simple local search optimization
- **Simulated Annealing**: Temperature-based optimization with cooling schedule
- **Tabu Search**: Search with memory to avoid revisiting solutions
- **Genetic Algorithm**: Population-based evolutionary optimization
- **K-means Clustering**: Groups guests based on relationship similarities
- **Greedy**: Makes locally optimal choices at each stage

5. **Algorithm Parameters**: Adjust parameters specific to the selected algorithm.

6. **Dataset Paths**: Specify the locations of your dataset files.

7. **Action Buttons**:
- **Optimize Tables**: Run the selected algorithm and display results
- **Compare All Algorithms**: Run all algorithms and compare their performance

### Visualization

The table visualizer shows:
- Tables as circles
- Guests arranged around each table
- Relationship lines between guests:
  - Green lines: Positive relationships (thicker = stronger)
  - Red lines: Negative relationships (thicker = stronger)
  - White lines: Neutral relationships

Navigate between pages using the left and right arrow keys, and exit with the ESC key.

### Results

The optimization results will be displayed in:
1. The main window text area
2. A graphical visualization of tables
3. Performance comparison charts when comparing algorithms

When saving results, the program creates:
- `algorithm_comparison.txt`: Overview of all algorithm performances
- Individual solution files for each algorithm detailing table assignments

## Algorithms Explained

1. **Random**: Creates a random assignment of guests to tables.

2. **Hill Climbing**: Starts with a random solution and iteratively makes small improvements by swapping guests between tables if it increases happiness.

3. **Simulated Annealing**: Similar to hill climbing but allows accepting worse solutions with a probability that decreases over time, helping to escape local optima.

4. **Tabu Search**: Keeps a "tabu list" of recently visited solutions to avoid cycling and explore more of the solution space.

5. **Genetic Algorithm**: Evolves a population of solutions through selection, crossover, and mutation over multiple generations.

6. **K-means Clustering**: Groups guests with similar relationship patterns into clusters corresponding to tables.

7. **Greedy**: Assigns guests one by one, maximizing happiness at each step.

## Performance Metrics

- **Happiness Score**: Total sum of relationship values between guests at the same table. Higher is better.
- **Execution Time**: Time taken for the algorithm to complete in seconds.
- **Efficiency**: Happiness score per second, showing algorithm performance relative to computational cost.
- **Table Distribution**: Analysis of how guests and happiness are distributed across tables.
- **Guest Distribution**: Analysis of how many guests are placed at each table.
- **Happiness Distribution**: Distribution of happiness scores across different tables.

## Comparison Features

The application includes a comprehensive algorithm comparison tool with multiple visualization pages:

1. **Basic Performance**: Bar charts showing happiness scores and execution times for all algorithms.
2. **Efficiency Analysis**: Comparison of algorithms based on happiness per second and performance tradeoff.
3. **Algorithm-Specific Analysis**: Detailed pages for each optimization algorithm showing:
   - Guest distribution across tables
   - Happiness scores for each table
   - Percentage contribution of each table to the total happiness