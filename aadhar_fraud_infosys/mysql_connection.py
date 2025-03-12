import mysql.connector
from mysql.connector import Error
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_connection():
    """
    Create a connection to the MySQL database
    Returns the connection object
    """
    try:
        connection = mysql.connector.connect(
            host=os.getenv('MYSQL_HOST', 'localhost'),
            user=os.getenv('MYSQL_USER', 'root'),
            password=os.getenv('MYSQL_PASSWORD', 'Attri@20589'),
            database=os.getenv('MYSQL_DB', 'input_db')
        )
        
        if connection.is_connected():
            logger.info("Connected to MySQL database")
            create_tables(connection)
        return connection
    except Error as e:
        logger.error(f"Error connecting to MySQL: {e}")
        return None

def create_tables(connection):
    """
    Create the required tables if they don't exist
    """
    try:
        cursor = connection.cursor()
        
        # Create main verification results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS verification_results (
                id INT AUTO_INCREMENT PRIMARY KEY,
                image_no VARCHAR(255),
                document_type VARCHAR(50),
                status VARCHAR(20),
                final_remark VARCHAR(255),
                uid_match_score FLOAT,
                name_match_score FLOAT,
                address_match_score FLOAT,
                name_match_type VARCHAR(100),
                session_id VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create table for extracted data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS extracted_data (
                id INT AUTO_INCREMENT PRIMARY KEY,
                result_id INT,
                extracted_uid VARCHAR(20),
                extracted_name VARCHAR(255),
                extracted_address TEXT,
                input_uid VARCHAR(20),
                input_name VARCHAR(255),
                input_address TEXT,
                FOREIGN KEY (result_id) REFERENCES verification_results(id) ON DELETE CASCADE
            )
        ''')
        
        # Create table for processing sessions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processing_sessions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(255) UNIQUE,
                total_documents INT,
                accepted_documents INT,
                rejected_documents INT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        connection.commit()
        logger.info("Database tables created successfully")
    except Error as e:
        logger.error(f"Error creating tables: {e}")
    finally:
        if cursor:
            cursor.close()

def save_verification_results(connection, results, session_id):
    """
    Save verification results and extracted data to database
    
    Args:
        connection: MySQL connection object
        results: List of result dictionaries
        session_id: Unique session identifier
    
    Returns:
        success: Boolean indicating success/failure
    """
    try:
        cursor = connection.cursor()
        
        # First save the session summary
        total_docs = len(results)
        accepted = sum(1 for r in results if r['Accepted/Rejected'] == 'Accepted')
        rejected = total_docs - accepted
        
        cursor.execute('''
            INSERT INTO processing_sessions (session_id, total_documents, accepted_documents, rejected_documents)
            VALUES (%s, %s, %s, %s)
        ''', (session_id, total_docs, accepted, rejected))
        
        # Save each result
        for result in results:
            # Insert into verification_results table
            cursor.execute('''
                INSERT INTO verification_results 
                (image_no, document_type, status, final_remark, uid_match_score, 
                name_match_score, address_match_score, name_match_type, session_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ''', (
                result.get('Image No.', ''),
                result.get('Document Type', 'Unknown'),
                result.get('Accepted/Rejected', 'Unknown'),
                result.get('Final Remarks', ''),
                result.get('UID Match Score', 0),
                result.get('Name Match Score', 0),
                result.get('Address Match Score', 0),
                result.get('Name Match Type', 'No match'),
                session_id
            ))
            
            # Get the last inserted ID for the foreign key reference
            result_id = cursor.lastrowid
            
            # Insert into extracted_data table
            cursor.execute('''
                INSERT INTO extracted_data
                (result_id, extracted_uid, extracted_name, extracted_address, 
                input_uid, input_name, input_address)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            ''', (
                result_id,
                result.get('Extracted UID', ''),
                result.get('Extracted Name', ''),
                result.get('Extracted Address', ''),
                result.get('Input UID', ''),
                result.get('Input Name', ''),
                result.get('Input Address', '')
            ))
        
        connection.commit()
        logger.info(f"Successfully saved {len(results)} verification results to database")
        return True
    except Error as e:
        logger.error(f"Error saving results to database: {e}")
        connection.rollback()
        return False
    finally:
        if cursor:
            cursor.close()

def export_to_excel(connection, session_id):
    """
    Export results from database to Excel for a specific session
    
    Args:
        connection: MySQL connection object
        session_id: Session ID to export
        
    Returns:
        df: Pandas DataFrame containing the results
    """
    try:
        import pandas as pd
        
        query = '''
            SELECT 
                vr.image_no AS 'Image No.',
                vr.document_type AS 'Document Type',
                vr.status AS 'Status',
                vr.final_remark AS 'Final Remarks',
                vr.uid_match_score AS 'UID Match Score',
                vr.name_match_score AS 'Name Match Score',
                vr.name_match_type AS 'Name Match Type',
                vr.address_match_score AS 'Address Match Score',
                ed.extracted_uid AS 'Extracted UID',
                ed.extracted_name AS 'Extracted Name',
                ed.extracted_address AS 'Extracted Address',
                ed.input_uid AS 'Input UID',
                ed.input_name AS 'Input Name',
                ed.input_address AS 'Input Address'
            FROM verification_results vr
            LEFT JOIN extracted_data ed ON vr.id = ed.result_id
            WHERE vr.session_id = %s
            ORDER BY vr.id
        '''
        
        df = pd.read_sql(query, connection, params=(session_id,))
        return df
    except Error as e:
        logger.error(f"Error exporting results to Excel: {e}")
        return None
    except Exception as e:
        logger.error(f"General error during export: {e}")
        return None

def close_connection(connection):
    """
    Close the database connection
    """
    if connection and connection.is_connected():
        connection.close()
        logger.info("Database connection closed")