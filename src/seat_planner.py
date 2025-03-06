import random
import pandas as pd

class SeatPlanner:
    def __init__(self, data, table_size):
        """
        data: dictionary with keys 'guests' and 'relationship_matrix'
        table_size: number of guests per table.
        """
        self.data = data
        self.table_size = table_size
        
        # Create a DataFrame using the guest list for row and column labels.
        self.relationship_df = pd.DataFrame(
            data['relationship_matrix'],
            index=data['guests'],
            columns=data['guests']
        )

    def get_random_arrangement(self):
        """
        Randomly shuffles the guests and partitions them into tables based on table_size.
        Returns a list of tables, where each table is a list of guest names.
        """
        guests = list(self.relationship_df.index)
        random.shuffle(guests)
        num_tables = (len(guests) + self.table_size - 1) // self.table_size
        tables = [guests[i * self.table_size:(i + 1) * self.table_size] for i in range(num_tables)]
        return tables
    
