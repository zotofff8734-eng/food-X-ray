import sqlite3
import os

DATABASE_NAME = "my_food_database.db"

def initialize_database():
    """
    Initializes the SQLite database by creating the necessary tables
    if they do not already exist.
    """
    # Check if the database file already exists. If so, do nothing.
    if os.path.exists(DATABASE_NAME):
        print(f"Database '{DATABASE_NAME}' already exists. Initialization skipped.")
        return

    conn = None
    try:
        print(f"Creating new database: {DATABASE_NAME}")
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()

        # SQL statement for creating the 'dishes' table
        create_dishes_table_sql = """
        CREATE TABLE dishes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name_ru TEXT NOT NULL UNIQUE,
            calories REAL,
            health_impact TEXT,
            recipe TEXT
        );
        """

        # SQL statement for creating the 'ingredients' table
        create_ingredients_table_sql = """
        CREATE TABLE ingredients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dish_id INTEGER NOT NULL,
            name_ru TEXT NOT NULL,
            is_healthy BOOLEAN,
            reasoning TEXT,
            FOREIGN KEY (dish_id) REFERENCES dishes(id) ON DELETE CASCADE
        );
        """
        
        print("Executing CREATE TABLE statement for 'dishes'...")
        cursor.execute(create_dishes_table_sql)
        print("Table 'dishes' created successfully.")
        
        print("Executing CREATE TABLE statement for 'ingredients'...")
        cursor.execute(create_ingredients_table_sql)
        print("Table 'ingredients' created successfully.")

        conn.commit()
        print("Database initialized successfully.")

    except sqlite3.Error as e:
        print(f"An error occurred during database initialization: {e}")
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    initialize_database()
