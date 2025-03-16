import tkinter as tk
from tkinter import ttk
from matrix_builder import MatrixBuilder
from seat_planner import SeatPlanner
from table_visualizer import TableVisualizer

class Menu:

    def show(self):
        self.root = tk.Tk()
        self.root.title("Seating Arrangement Menu")

        # table size
        lbl_size = tk.Label(self.root, text="Table Size:")
        lbl_size.grid(row=0, column=0, padx=10, pady=10)
        self.entry_table_size = tk.Entry(self.root)
        self.entry_table_size.insert(0, "8")
        self.entry_table_size.grid(row=0, column=1, padx=10, pady=10)

        # arrangement type
        lbl_arr = tk.Label(self.root, text="Arrangement Type:")
        lbl_arr.grid(row=1, column=0, padx=10, pady=10)
        self.combo_arrangement = ttk.Combobox(self.root, values=["Random"], state="readonly")
        self.combo_arrangement.current(0)
        self.combo_arrangement.grid(row=1, column=1, padx=10, pady=10)

        # display option
        lbl_display = tk.Label(self.root, text="Display Option:")
        lbl_display.grid(row=2, column=0, padx=10, pady=10)
        self.combo_display = ttk.Combobox(self.root, values=["Table Visualizer"], state="readonly")
        self.combo_display.current(0)
        self.combo_display.grid(row=2, column=1, padx=10, pady=10)

        # generate button
        btn_generate = tk.Button(self.root, text="Generate", command=self.on_generate)
        btn_generate.grid(row=3, column=0, columnspan=2, pady=20)

        self.root.mainloop()

    def on_generate(self):
        try:
            table_size = int(self.entry_table_size.get())
        except ValueError:
            table_size = 8
        arrangement_type = self.combo_arrangement.get()
        display_option = self.combo_display.get()
        self.root.destroy()
        # for now only generate seating arrangement
        self.generate_seating(table_size, arrangement_type, display_option)


    def generate_seating(self, table_size, arrangement_type, display_option):
        print(f"Selected table size: {table_size}")
        print(f"Selected arrangement type: {arrangement_type}")
        print(f"Selected display option: {display_option}")

        builder = MatrixBuilder("dataset/guestlist.csv", "dataset/likes.csv", "dataset/dislikes.csv")
        builder.build_matrix()
        data = builder.get_matrix_data()

        planner = SeatPlanner(data, table_size)
        
        if arrangement_type == "Random":
            tables = planner.get_random_arrangement()
        else:
            print("Invalid arrangement type. Using random arrangement.")
            tables = planner.get_random_arrangement()

        builder.save_matrix_to_csv("relationship_matrix.csv")

        if display_option == "Table Visualizer":
            visualizer = TableVisualizer(tables)
            visualizer.show()
        else:
            print("Unknown display option.")