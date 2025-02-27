import pandas as pd
import numpy as np
import os

class MatrixBuilder:
    def __init__(self, guest_file, likes_file, dislikes_file):
        self.guest_file = guest_file
        self.likes_file = likes_file
        self.dislikes_file = dislikes_file

        self.guests = []
        self.relationship_matrix = None
        self.guest_index = {}  # Maps guest names to indices

        self.likes = set()
        self.dislikes = set()

    def load_guests(self):
        """Loads guest names into a list and creates an index map."""
        try:
            df = pd.read_csv(self.guest_file, sep=';')
            self.guests = df['Guest'].tolist()
            self.guest_index = {name: i for i, name in enumerate(self.guests)}

        except FileNotFoundError:
            print(f"Error: {self.guest_file} not found.")
    
    def load_linkings(self):
        """Loads the likes and dislikes from files into separate attributes."""
        try:
            likes_df = pd.read_csv(self.likes_file, sep=';')
            self.likes = set(zip(likes_df['Guest_A'], likes_df['Guest_B']))

            dislikes_df = pd.read_csv(self.dislikes_file, sep=';')
            self.dislikes = set(zip(dislikes_df['Guest_A'], dislikes_df['Guest_B']))
            
        except FileNotFoundError as e:
            print(f"Error: {e.filename} not found.")

    def initialize_matrix(self):
        """Creates an empty relationship matrix (0s)."""
        size = len(self.guests)
        self.relationship_matrix = np.zeros((size, size), dtype=int)

    def calculate_relationship_value(self, person1, person2):
        #TODO: Add aditional logic to calculate the relationship value

        if (person1, person2) in self.likes:
            return 1
        elif (person1, person2) in self.dislikes:
            return -1
        else:
            return 0

    def load_relationships(self):
        """Iterates through the matrix and fills in values using calculate_relationship_value."""
        for i in range(len(self.guests)):
            for j in range(i + 1, len(self.guests)):
                person1 = self.guests[i]
                person2 = self.guests[j]
                
                value = self.calculate_relationship_value(person1, person2)
                
                # Populate both (i, j) and (j, i) because the relationship is mutual
                self.relationship_matrix[i][j] = value
                self.relationship_matrix[j][i] = value

    def build_matrix(self):
        """Runs the full process of building the relationship matrix."""
        self.load_guests()
        self.load_linkings()
        self.initialize_matrix()
        self.load_relationships()

    def get_matrix_data(self):
        """Returns the guest list and relationship matrix."""
        return {
            "guests": self.guests,
            "relationship_matrix": self.relationship_matrix
        }

    def save_matrix_to_csv(self, file_name="relationship_matrix.csv"):
        if not os.path.exists("bin"):
            os.makedirs("bin")

        file_path = os.path.join("bin", file_name)

        df = pd.DataFrame(self.relationship_matrix, index=self.guests, columns=self.guests)
        df.to_csv(file_path, sep=';', index=True)

        print(f"Matrix saved to {file_path}")