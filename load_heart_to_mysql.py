import pandas as pd
import mysql.connector
from mysql.connector import Error
import os

def connect_to_db():
    \"\"\"Establish connection to MySQL database.\"\"\"
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',  # Update with your MySQL password
            database='heart_db'
        )
        if connection.is_connected():
            print('Successfully connected to MySQL database')
            return connection
    except Error as e:
        print(f'Error connecting to MySQL: {e}')
        return None

def create_database_and_table(cursor):
    \"\"\"Create database and table if not exists.\"\"\"
    try:
        # Create database
        cursor.execute('CREATE DATABASE IF NOT EXISTS heart_db')
        cursor.execute('USE heart_db')
        
        # Drop table if exists for clean load
        cursor.execute('DROP TABLE IF EXISTS heart_data')
        
        # Create table with matching columns
        create_table_sql = '''
        CREATE TABLE heart_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            age INT,
            sex INT,
            cp INT,
            trestbps INT,
            chol INT,
            fbs INT,
            restecg INT,
            thalach INT,
            exang INT,
            oldpeak FLOAT,
            slope INT,
            ca INT,
            thal INT,
            target INT
        )
        '''
        cursor.execute(create_table_sql)
        print('Table heart_data created successfully')
    except Error as e:
        print(f'Error creating database/table: {e}')

def load_csv_to_db(connection, csv_path='heart.csv'):
    \"\"\"Load CSV data into heart_data table.\"\"\"
    try:
        if not os.path.exists(csv_path):
            print(f'Error: {csv_path} not found. Please place heart.csv in the project directory.')
            return
        
        # Read CSV with explicit column mapping
        df = pd.read_csv(csv_path)
        
        # Expected columns
        expected_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        
        if not all(col in df.columns for col in expected_columns):
            missing = [col for col in expected_columns if col not in df.columns]
            print(f'Error: Missing columns in CSV: {missing}. Expected: {expected_columns}')
            return
        
        # Select and reorder columns
        df = df[expected_columns]
        
        cursor = connection.cursor()
        
        # Prepare insert query
        insert_query = '''
        INSERT INTO heart_data (age, sex, cp, trestbps, chol, fbs, restecg, thalach, 
                               exang, oldpeak, slope, ca, thal, target)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        '''
        
        # Batch insert with commits
        batch_size = 1000
        data_to_insert = [tuple(row) for row in df.itertuples(index=False)]
        
        for i in range(0, len(data_to_insert), batch_size):
            batch = data_to_insert[i:i + batch_size]
            cursor.executemany(insert_query, batch)
            connection.commit()
            print(f'Committed batch {i//batch_size + 1} ({len(batch)} rows)')
        
        print(f'Successfully loaded {len(df)} rows into heart_data table.')
        
    except Exception as e:
        print(f'Error loading data: {e}')
    finally:
        if 'cursor' in locals():
            cursor.close()

def main():
    connection = connect_to_db()
    if connection:
        try:
            cursor = connection.cursor()
            create_database_and_table(cursor)
            load_csv_to_db(connection)
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
                print('MySQL connection closed.')

if __name__ == '__main__':
    main()

