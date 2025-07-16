"""
Script to clear all data from pgvector tables in the configured PostgreSQL database.
Uses pg8000 for direct database access.

WARNING: This will DELETE ALL DATA from all tables containing vector columns!
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
        print("No tables with pgvector columns found. Nothing to clean.")
        conn.close()
        return

    # Collect unique tables
    tables = set((schema, table) for schema, table, _ in vector_tables)
    print("The following tables with vector columns will be cleared:")
    for schema, table in tables:
        print(f"  - {schema}.{table}")

    confirm = input("Are you sure you want to DELETE ALL DATA from these tables? Type 'yes' to confirm: ")
    if confirm.strip().lower() != "yes":
        print("Aborted.")
        conn.close()
        return

    # Disable foreign key checks temporarily (if needed)
    try:
        cursor.execute("SET session_replication_role = 'replica';")
    except Exception:
        pass  # Not all users can do this

    # Truncate each table
    for schema, table in tables:
        try:
            print(f"Truncating {schema}.{table} ...")
            cursor.execute(f'TRUNCATE TABLE "{schema}"."{table}" RESTART IDENTITY CASCADE;')
        except Exception as e:
            print(f"Failed to truncate {schema}.{table}: {e}")

    # Restore foreign key checks
    try:
        cursor.execute("SET session_replication_role = 'origin';")
    except Exception:
        pass

    conn.commit()
    conn.close()
    print("✅ All pgvector tables have been cleared.")

if __name__ == "__main__":
    main()
