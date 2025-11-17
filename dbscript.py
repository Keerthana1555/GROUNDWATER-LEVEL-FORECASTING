"""
Database bootstrap script for Fruits & Vegetable Adulteration app.
- Drops database and tables
- Creates database and tables fresh

Spec notes:
- No foreign key constraints
- No hashing for passwords
- No ORM
"""
import mysql.connector

DB_NAME = "underwater_2025"
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
}


def reset_database():
    # Connect without database first
    conn = mysql.connector.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Drop database if exists
    cur.execute(f"DROP DATABASE IF EXISTS {DB_NAME}")

    # Create database
    cur.execute(
        f"CREATE DATABASE {DB_NAME} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
    )

    # Use database
    cur.execute(f"USE {DB_NAME}")

    # Create tables (no foreign keys as per spec)
    cur.execute(
        """
        CREATE TABLE users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            email VARCHAR(120) NOT NULL UNIQUE,
            password VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB
        """
    )

    # Create predictions table
    cur.execute(
        """
        CREATE TABLE predictions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            latitude DECIMAL(10, 7) NOT NULL,
            longitude DECIMAL(10, 7) NOT NULL,
            prediction_date DATE NOT NULL,
            predicted_dtwl DECIMAL(10, 4) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        ) ENGINE=InnoDB
        """
    )

    conn.commit()
    cur.close()
    conn.close()


if __name__ == "__main__":
    reset_database()
    print("Database reset and tables created for 'underwater_2025'.")