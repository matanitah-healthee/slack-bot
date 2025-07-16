"""
Script to print the schema of the Neo4j graph database.
Shows all node labels, relationship types, and property keys.

Usage: python utils/check_neo4j.py
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

    try:
        # Get all node labels
        labels_result = session.run("CALL db.labels()")
        labels = [record["label"] for record in labels_result]
        print("\n=== Node Labels ===")
        if labels:
            for label in labels:
                print(f"  - {label}")
        else:
            print("  (No node labels found)")

        # Get all relationship types
        rels_result = session.run("CALL db.relationshipTypes()")
        rel_types = [record["relationshipType"] for record in rels_result]
        print("\n=== Relationship Types ===")
        if rel_types:
            for rel in rel_types:
                print(f"  - {rel}")
        else:
            print("  (No relationship types found)")

        # Get all property keys
        prop_result = session.run("CALL db.propertyKeys()")
        prop_keys = [record["propertyKey"] for record in prop_result]
        print("\n=== Property Keys ===")
        if prop_keys:
            for key in prop_keys:
                print(f"  - {key}")
        else:
            print("  (No property keys found)")

        # Print per-label property keys and sample counts
        print("\n=== Node Label Details ===")
        if labels:
            for label in labels:
                print(f"\nLabel: {label}")
                # Get property keys for this label
                props_query = (
                    f"MATCH (n:`{label}`) "
                    "UNWIND keys(n) AS key "
                    "RETURN key, count(*) AS count "
                    "ORDER BY count DESC"
                )
                props = session.run(props_query)
                props_list = list(props)
                if props_list:
                    print("  Properties:")
                    for record in props_list:
                        print(f"    - {record['key']} (in {record['count']} nodes)")
                else:
                    print("  (No properties found for this label)")
                # Count nodes
                count_query = f"MATCH (n:`{label}`) RETURN count(n) AS count"
                count = session.run(count_query).single()["count"]
                print(f"  Node count: {count}")

        # Print per-relationship type property keys and sample counts
        print("\n=== Relationship Type Details ===")
        if rel_types:
            for rel in rel_types:
                print(f"\nRelationship: {rel}")
                # Get property keys for this relationship type
                props_query = (
                    f"MATCH ()-[r:`{rel}`]->() "
                    "UNWIND keys(r) AS key "
                    "RETURN key, count(*) AS count "
                    "ORDER BY count DESC"
                )
                props = session.run(props_query)
                props_list = list(props)
                if props_list:
                    print("  Properties:")
                    for record in props_list:
                        print(f"    - {record['key']} (in {record['count']} relationships)")
                else:
                    print("  (No properties found for this relationship type)")
                # Count relationships
                count_query = f"MATCH ()-[r:`{rel}`]->() RETURN count(r) AS count"
                count = session.run(count_query).single()["count"]
                print(f"  Relationship count: {count}")

        print("\n✅ Schema printed successfully.")

    except Exception as e:
        print(f"Error querying Neo4j: {e}")
    finally:
        session.close()
        driver.close()

if __name__ == "__main__":
    main()
