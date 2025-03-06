import matplotlib.pyplot as plt
import numpy as np
import math

class TableVisualizer:
    def __init__(self, tables, tables_per_page=4):
        """
        tables: list of tables, where each table is a list of guest names.
        tables_per_page: number of tables to display per page.
        """
        self.tables = tables
        self.tables_per_page = tables_per_page
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

            for angle, guest in zip(angles, table):
                # Draw the seat marker.
                seat_x, seat_y = np.cos(angle) * 1.5, np.sin(angle) * 1.5
                seat_marker = plt.Circle((seat_x, seat_y), 0.1, color='orange')
                ax.add_artist(seat_marker)
                # Draw guest name.
                text_x, text_y = np.cos(angle) * text_offset, np.sin(angle) * text_offset
                ax.text(text_x, text_y, guest, ha='center', va='center', fontsize=font_size, wrap=True)
            
            table_num = idx + 1 + page_idx * self.tables_per_page
            ax.set_title(f"Table {table_num}", fontsize=14)
        
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

