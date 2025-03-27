import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.patches import Circle

class TableVisualizer:
    def __init__(self, tables, tables_per_page=4, relationship_matrix=None, guests=None, table_scores=None):
        """
        tables: list of tables, where each table is a list of guest names.
        tables_per_page: number of tables to display per page.
        relationship_matrix: matrix indicating relationships between guests.
        guests: list of all guests.
        table_scores: dictionary mapping table indices to happiness scores.
        """
        self.tables = tables
        self.tables_per_page = tables_per_page
        self.relationship_matrix = relationship_matrix
        self.guests = guests
        self.table_scores = table_scores or {}  # Initialize empty dict if None
        # Split the tables into pages.
        n_tables = len(tables)
        self.pages = [tables[i:i+tables_per_page] for i in range(0, n_tables, tables_per_page)]
        self.current_page = 0
        self.fig = None
        self.axes = None

    def draw_page(self, page_idx):
        """Draws the tables for the specified page index."""
        self.fig.clf()
        page_tables = self.pages[page_idx]
        num_tables = len(page_tables)
        cols = 2  # Fixed two columns per page.
        rows = math.ceil(num_tables / cols)
        
        # Create subplots for this page.
        axes = self.fig.subplots(rows, cols)
        # Force axes to be a 1D array of Axes objects.
        axes = np.atleast_1d(axes).flatten()
        self.axes = axes

        for idx, table in enumerate(page_tables):
            ax = self.axes[idx]
            ax.set_xlim(-2.5, 2.5)
            ax.set_ylim(-2.5, 2.5)
            ax.set_aspect('equal')
            ax.axis('off')

            # Draw table circle.
            table_circle = plt.Circle((0, 0), 1.8, color='lightblue', fill=True, alpha=0.5)
            ax.add_artist(table_circle)

            num_seats = len(table)
            angles = np.linspace(0, 2 * np.pi, num_seats, endpoint=False)

            # Adjust font size and text offset based on the number of seats.
            font_size = max(8, 14 - num_seats // 3)
            text_offset = 2.0

            guest_positions = []
            for angle, guest in zip(angles, table):
                # Draw the seat marker.
                seat_x, seat_y = np.cos(angle) * 1.5, np.sin(angle) * 1.5
                seat_marker = plt.Circle((seat_x, seat_y), 0.1, color='orange')
                ax.add_artist(seat_marker)
                guest_positions.append((seat_x, seat_y))
                # Draw guest name.
                text_x, text_y = np.cos(angle) * text_offset, np.sin(angle) * text_offset
                ax.text(text_x, text_y, guest, ha='center', va='center', fontsize=font_size, wrap=True)
            
            # Draw relationship lines between guests if relationship_matrix is provided
            if self.relationship_matrix is not None and self.guests is not None:
                for i, guest1 in enumerate(table):
                    guest1_idx = self.guests.index(guest1)
                    for j, guest2 in enumerate(table):
                        if i < j:  # Only process each pair once
                            guest2_idx = self.guests.index(guest2)
                            relationship = self.relationship_matrix[guest1_idx][guest2_idx]
                            
                            # Determine line color based on relationship value
                            if relationship > 0:
                                color = 'green'  # Positive relationship
                                linewidth = min(relationship / 2, 3)  # Adjust thickness based on strength
                            elif relationship < 0:
                                color = 'red'  # Negative relationship
                                linewidth = min(abs(relationship) / 2, 3)  # Adjust thickness based on strength
                            else:
                                color = 'white'  # Neutral relationship
                                linewidth = 1
                            
                            # Draw line between guests
                            x1, y1 = guest_positions[i]
                            x2, y2 = guest_positions[j]
                            ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, alpha=0.7)

            table_num = idx + 1 + page_idx * self.tables_per_page
            
            # Display table score if available
            title_text = f"Table {table_num}"
            if self.table_scores and table_num - 1 in self.table_scores:  # Adjust for 0-based indexing
                title_text += f" (Score: {self.table_scores[table_num - 1]:.1f})"
            
            ax.set_title(title_text, fontsize=14)
        
        # Remove any unused subplots.
        for j in range(idx + 1, len(self.axes)):
            self.fig.delaxes(self.axes[j])
        
        self.fig.suptitle(f"Page {page_idx + 1} of {len(self.pages)}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        self.fig.canvas.draw()

    def on_key(self, event):
        """Handles key press events for navigation."""
        if event.key == 'right':
            if self.current_page < len(self.pages) - 1:
                self.current_page += 1
                self.draw_page(self.current_page)
        elif event.key == 'left':
            if self.current_page > 0:
                self.current_page -= 1
                self.draw_page(self.current_page)
        elif event.key == 'escape':
            plt.close(self.fig)

    def show(self):
        """Displays the interactive window with arrow key navigation."""
        self.fig = plt.figure(figsize=(12, 8))
        self.draw_page(self.current_page)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        plt.show()

