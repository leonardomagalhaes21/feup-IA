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
        self.guest_details = {}  # Maps guest names to their characteristics

        self.likes = set()
        self.dislikes = set()

    def load_guests(self):
        """Loads guest names into a list and creates an index map."""
        try:
            df = pd.read_csv(self.guest_file, sep=';')
            self.guests = df['Guest'].tolist()

            for _, row in df.iterrows():
                self.guest_details[row['Guest']] = {
                    'Age': int(row['Age']),
                    'Group': row['Group'],
                    'Interests': row['Interests']
                }

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

    def calculate_asymmetric_relationship(self, person1, person2):
        """Handles the asymmetric relationship (likes/dislikes)."""
        if (person1, person2) in self.likes:
            return 3
        elif (person1, person2) in self.dislikes:
            return -3
        return 0
    

    def calculate_symmetric_relationship(self, person1, person2):
        """Handles the symmetric relationship (age, group, interests)."""
        details1 = self.guest_details.get(person1, {})
        details2 = self.guest_details.get(person2, {})

        if not details1 or not details2:
            return 0  # If details are missing, return 0

        score = 0

        # Handle age difference
        age_diff = abs(details1['Age'] - details2['Age'])
        if age_diff <= 5:
            score += 1
        elif age_diff > 15:
            score -= 1

        # Handle group
        if details1['Group'] == details2['Group']:
            score += 4

        # Handle interests
        if details1['Interests'] == details2['Interests']:
            score += 1

        return score

    def calculate_relationship_value(self, person1, person2):
        """Calculates the relationship value between two people."""
        symmetric_value = self.calculate_symmetric_relationship(person1, person2)

        value1 = self.calculate_asymmetric_relationship(person1, person2) + symmetric_value
        value2 = self.calculate_asymmetric_relationship(person2, person1) + symmetric_value

        return value1, value2
        

    def load_relationships(self):
        """Iterates through the matrix and fills in values using calculate_relationship_value."""
        for i in range(len(self.guests)):
            for j in range(i + 1, len(self.guests)):
                person1 = self.guests[i]
                person2 = self.guests[j]
                
                value1, value2 = self.calculate_relationship_value(person1, person2)
                
                self.relationship_matrix[i][j] = value1
                self.relationship_matrix[j][i] = value2

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

