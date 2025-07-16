"""
Script to clear all nodes and relationships from the configured Neo4j database.

WARNING: This will DELETE ALL DATA from the Neo4j database!
"""

import os
import sys

def get_neo4j_config():
    # Try to import config if available, else use environment variables
    try:
        from config import config as app_config
        return {
            "uri": getattr(app_config, "NEO4J_URI", os.getenv("NEO4J_URI", "bolt://localhost:7687")),
            "user": getattr(app_config, "NEO4J_USER", os.getenv("NEO4J_USER", "neo4j")),
            "password": getattr(app_config, "NEO4J_PASSWORD", os.getenv("NEO4J_PASSWORD", "password")),
        }
    except Exception:
        return {
            "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            "user": os.getenv("NEO4J_USER", "neo4j"),
            "password": os.getenv("NEO4J_PASSWORD", "password"),
        }

def main():
    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("neo4j Python driver is not installed. Please install it with 'pip install neo4j'")
        sys.exit(1)

    config = get_neo4j_config()
    print(f"Connecting to Neo4j at {config['uri']} as user '{config['user']}'...")

    try:
        driver = GraphDatabase.driver(config["uri"], auth=(config["user"], config["password"]))
        session = driver.session()
    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")
        sys.exit(1)

    # Count nodes and relationships before deletion
    try:
        node_count = session.run("MATCH (n) RETURN count(n) AS count").single()["count"]
        rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]
    except Exception as e:
        print(f"Error querying Neo4j: {e}")
        session.close()
        driver.close()
        sys.exit(1)

    if node_count == 0 and rel_count == 0:
        print("Neo4j database is already empty. Nothing to clean.")
        session.close()
        driver.close()
        return

    print(f"Database contains {node_count} nodes and {rel_count} relationships.")
    confirm = input("Are you sure you want to DELETE ALL DATA from Neo4j? Type 'yes' to confirm: ")
    if confirm.strip().lower() != "yes":
        print("Aborted.")
        session.close()
        driver.close()
        return

    # Delete all nodes and relationships
    try:
        print("Deleting all nodes and relationships...")
        session.run("MATCH (n) DETACH DELETE n")
        print("✅ All nodes and relationships have been deleted from Neo4j.")
    except Exception as e:
        print(f"Failed to delete data: {e}")
    finally:
        session.close()
        driver.close()

if __name__ == "__main__":
    main()
