import pymysql
import pandas as pd

# Establecer la conexión con MySQL
connection = pymysql.connect(
    host="201.175.13.78",
    user="root",
    password="test1234",
    port=32236,
    database="KIOgrafIA"
)

# Leer el archivo CSV en un DataFrame de pandas
csv_path = "./data/QL.csv"
df = pd.read_csv(csv_path).fillna(value=0)

# Inferir nombres de columnas y tipos de datos
column_names = df.columns.tolist()
column_types = df.dtypes.to_dict()

# Generar la declaración SQL para crear la tabla
table_name = 'CMDB'
create_table_sql = f"CREATE TABLE {table_name} ("
for name, dtype in column_types.items():
    if name == 'descripcion_quote':
        create_table_sql += f"{name} LONGTEXT, "  # Aumenta el tamaño de VARCHAR
    elif dtype == 'object':
        create_table_sql += f"{name} VARCHAR(255), "
    elif dtype == 'int64':
        create_table_sql += f"{name} INT, "
    elif dtype == 'float64':
        create_table_sql += f"{name} FLOAT, "
    elif dtype == 'bool':
        create_table_sql += f"{name} TINYINT(1), "
    elif dtype == 'datetime64[ns]':
        create_table_sql += f"{name} DATETIME, "
create_table_sql = create_table_sql[:-2] + ")"
try:
    # Crear la tabla en la base de datos MySQL
    with connection.cursor() as cursor:
        cursor.execute(create_table_sql)
        print(f"Tabla {table_name} creada con éxito.")

    # Insertar los datos en la tabla por bloques
    chunk_size = 1000
    num_rows = df.shape[0]

    for start in range(0, num_rows, chunk_size):
        end = min(start + chunk_size, num_rows)
        chunk_df = df.iloc[start:end]
        
        # Crear las consultas SQL para cada bloque
        insert_sql = f"INSERT INTO {table_name} ({', '.join(column_names)}) VALUES "
        rows = []
        for row in chunk_df.itertuples(index=False, name=None):
            row_values = ', '.join(f"'{str(value)}'" if isinstance(value, str) else str(value) for value in row)
            rows.append(f"({row_values})")
        
        insert_sql += ', '.join(rows)
        
        with connection.cursor() as cursor:
            cursor.execute(insert_sql)
            print(f"Filas {start}-{end} insertadas en la tabla {table_name}.")
    
    # Confirmar los cambios
    connection.commit()

except Exception as e:
    print("Error:", e)

finally:
    connection.close()
