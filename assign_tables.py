from src.matrix_builder import MatrixBuilder
from src.table_assignment import TableAssignment

def main():
    # Paths para os arquivos CSV
    guest_file = 'dataset/guestlist.csv'
    likes_file = 'dataset/likes.csv'
    dislikes_file = 'dataset/dislikes.csv'

    # Construir a matriz de relacionamento
    print("Building relationship matrix...")
    builder = MatrixBuilder(guest_file, likes_file, dislikes_file)
    builder.build_matrix()
    builder.save_matrix_to_csv()  # Salvar a matriz para referência

    matrix_data = builder.get_matrix_data()
    guests = matrix_data["guests"]
    relationship_matrix = matrix_data["relationship_matrix"]

    # Número de pessoas por mesa (padrão: 8)
    table_size = 8
    
    print(f"Assigning {len(guests)} guests to tables of {table_size}...")
    
    # Criar o objeto de atribuição de mesas
    table_assigner = TableAssignment(relationship_matrix, guests, table_size)
    
    # Usar o método greedy para atribuir mesas (geralmente produz melhores resultados)
    tables = table_assigner.assign_tables_greedy()
    
    # Alternativamente, usar K-means (descomente a linha abaixo para usar K-means)
    # tables = table_assigner.assign_tables_kmeans()
    
    # Gerar e salvar o relatório
    report_file = table_assigner.generate_report()
    
    # Mostrar resultados
    print(f"\nTable assignments complete!")
    print(f"Number of tables: {len(tables)}")
    
    table_scores = table_assigner.calculate_table_happiness()
    total_happiness = sum(table_scores)
    
    print(f"Total happiness score: {total_happiness}")
    
    print("\nTable assignments:")
    for i, table in enumerate(tables):
        print(f"Table {i+1} (Score: {table_scores[i]}): {', '.join(table)}")
    
    print(f"\nDetailed report saved to {report_file}")

if __name__ == "__main__":
    main()
