from matrix_builder import MatrixBuilder
import pandas as pd


builder = MatrixBuilder("dataset/guestlist.csv", "dataset/likes.csv", "dataset/dislikes.csv")
builder.build_matrix()

data = builder.get_matrix_data()

print("Guest List:")
print(data["guests"])

print("\nRelationship Matrix:")
print(pd.DataFrame(data["relationship_matrix"], index=data["guests"], columns=data["guests"]))

builder.save_matrix_to_csv("relationship_matrix.csv")



