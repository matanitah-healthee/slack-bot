"""
Script to print all data from tables containing pgvector columns in the configured PostgreSQL database.
"""

import os
import sys

def get_pg_config():
    # Try to import config if available, else use environment variables
    try:
        from config import config as app_config
        return {
            "host": getattr(app_config, "POSTGRES_HOST", os.getenv("POSTGRES_HOST", "localhost")),
            "port": int(getattr(app_config, "POSTGRES_PORT", os.getenv("POSTGRES_PORT", 5432))),
            "database": getattr(app_config, "POSTGRES_DB", os.getenv("POSTGRES_DB", "vectordb")),
            "user": getattr(app_config, "POSTGRES_USER", os.getenv("POSTGRES_USER", "postgres")),
            "password": getattr(app_config, "POSTGRES_PASSWORD", os.getenv("POSTGRES_PASSWORD", "password")),
        }
    except Exception:
        return {
            "host": os.getenv("POSTGRES_HOST", "localhost"),
            "port": int(os.getenv("POSTGRES_PORT", 5432)),
            "database": os.getenv("POSTGRES_DB", "vectordb"),
            "user": os.getenv("POSTGRES_USER", "postgres"),
            "password": os.getenv("POSTGRES_PASSWORD", "password"),
        }

def main():
    try:
        import pg8000
    except ImportError:
        print("pg8000 is not installed. Please install it with 'pip install pg8000'")
        sys.exit(1)

    config = get_pg_config()
    print(f"Connecting to PostgreSQL at {config['host']}:{config['port']}, database '{config['database']}'...")

    try:
        conn = pg8000.connect(
            host=config["host"],
            port=config["port"],
            database=config["database"],
            user=config["user"],
            password=config["password"],
        )
        cursor = conn.cursor()
    except Exception as e:
        print(f"Failed to connect to PostgreSQL: {e}")
        sys.exit(1)

    # Find all tables with a 'vector' column (pgvector extension)
    try:
        cursor.execute("""
            SELECT table_schema, table_name, column_name
            FROM information_schema.columns
            WHERE udt_name = 'vector'
        """)
        vector_tables = cursor.fetchall()
    except Exception as e:
        print(f"Error querying for vector tables: {e}")
        conn.close()
        sys.exit(1)

    if not vector_tables:
        print("No tables with pgvector columns found. Nothing to print.")
        conn.close()
        return

    # Collect unique tables
    tables = set((schema, table) for schema, table, _ in vector_tables)
    print("The following tables with vector columns will be printed:")
    for schema, table in tables:
        print(f"  - {schema}.{table}")

    for schema, table in tables:
        print(f"\n=== Data from {schema}.{table} ===")
        try:
            # Get all columns for this table and identify vector columns
            cursor.execute("""
                SELECT column_name, udt_name
                FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s
                ORDER BY ordinal_position
            """, (schema, table))
            column_info = cursor.fetchall()
            columns = [row[0] for row in column_info]
            vector_columns = {row[0] for row in column_info if row[1] == 'vector'}
            
            col_str = ', '.join(f'"{col}"' for col in columns)
            cursor.execute(f'SELECT {col_str} FROM "{schema}"."{table}"')
            rows = cursor.fetchall()
            if not rows:
                print("  (No rows)")
            else:
                # Print as a table
                print(" | ".join(columns))
                print("-" * (len(" | ".join(columns)) + 5))
                for row in rows:
                    # Truncate vector values for readability
                    pretty_row = []
                    for i, val in enumerate(row):
                        col_name = columns[i]
                        if col_name in vector_columns:
                            # This is a vector column, show abbreviated form
                            if val is None:
                                pretty_row.append("NULL")
                            else:
                                val_str = str(val)
                                if len(val_str) > 50:
                                    pretty_row.append(f"[vector: {val_str[:20]}...{val_str[-10:]}]")
                                else:
                                    pretty_row.append(f"[vector: {val_str}]")
                        else:
                            pretty_row.append(str(val))
                    print(" | ".join(pretty_row))
        except Exception as e:
            print(f"Failed to fetch data from {schema}.{table}: {e}")

    conn.close()
    print("\n✅ Done printing all pgvector table data.")

if __name__ == "__main__":
    main()
