import sqlite3

import matplotlib.pyplot as plt


def save(db: str, result: dict, table_name: str = 'results'):
    """
    Cria uma tabela SQLite a partir da estrutura de um dicionário e insere os dados.

    Args:
        db: Nome do arquivo do banco de dados SQLite.
        table_name: Nome da tabela a ser criada.
        result: Dicionário contendo os dados a serem inseridos. As chaves do dicionário
                   serão as colunas da tabela, e os valores serão inseridos como linhas.
    """

    with sqlite3.connect(db) as conn:
        cursor = conn.cursor()

        columns = [f"{col} TEXT NOT NULL" for col in result.keys()]
        create_table_sql = f"CREATE TABLE IF NOT EXISTS {table_name} (id INTEGER PRIMARY KEY AUTOINCREMENT, {', '.join(columns)})"
        cursor.execute(create_table_sql)

        insert_sql = f"""INSERT OR REPLACE INTO {table_name} ({', '.join(result.keys())}) VALUES ({', '.join(['?'] * len(result))})"""
        cursor.execute(insert_sql, [*result.values()])

        conn.commit()


def get_analysis(db: str, query: str, x: str, y: str):
    """
    Plota um gráfico de linha com marcadores a partir de dados de um banco de dados SQLite.

    Args:
        db (str): O nome do arquivo do banco de dados SQLite.
        table (str): O nome da tabela a ser consultada.
        x (str): O nome da coluna a ser usada no eixo X.
        y (str): O nome da coluna a ser usada no eixo Y.
    """

    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute(query)

    column_names = [desc[0] for desc in cursor.description]

    rows = [dict(zip(column_names, row)) for row in cursor.fetchall()]

    conn.close()

    xs = [float(row[x]) for row in rows]
    ys = [float(row[y]) for row in rows]

    plt.figure(figsize=(10, 10))
    plt.plot(xs, ys, marker='o', linestyle='-')

    plt.title(f'Gráfico {x} x {y}')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid(axis='y', linestyle='--')

    plt.show()


if __name__ == '__main__':
    query = """
    SELECT * from results where instance = "ejor/separated/instance_01_2pol.txt" AND alpha = 0.95 AND final_t0 = 0.000001 ORDER BY final_t0 DESC;
    """

    get_analysis(db="simulated_annealing_v3_using_disturbance_temperature.sqlite",
                 query=query,
                 x="t0",
                 y="distance")
