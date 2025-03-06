from matrix_builder import MatrixBuilder
from seat_planner import SeatPlanner
from table_visualizer import TableVisualizer


builder = MatrixBuilder("dataset/guestlist.csv", "dataset/likes.csv", "dataset/dislikes.csv")
builder.build_matrix()

data = builder.get_matrix_data()

table_size = 8
planner = SeatPlanner(data, table_size)
tables = planner.get_random_arrangement()

builder.save_matrix_to_csv("relationship_matrix.csv")

visualizer = TableVisualizer(tables)
visualizer.show()

