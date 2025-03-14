import os
import pandas as pd
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def save_to_excel(results, session_id, excel_dir='static/excel_outputs', include_timestamp=True):
    """
    Automatically save verification results to an Excel file
    
    Args:
        results: List of result dictionaries
        session_id: Unique session identifier
        excel_dir: Directory to save Excel files
        include_timestamp: Whether to include timestamp in filename
        
    Returns:
        excel_path: Path to the saved Excel file or None if failed
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(excel_dir, exist_ok=True)
        
        # Create filename with optional timestamp
        if include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"verification_report_{session_id}_{timestamp}.xlsx"
        else:
            filename = f"verification_report_{session_id}.xlsx"
        
        # Full path for Excel file
        excel_path = os.path.join(excel_dir, filename)
        
        # Convert results to DataFrame
        df = pd.DataFrame(results)
        
        # Write to Excel file
        df.to_excel(excel_path, index=False, engine='openpyxl')
        
        logger.info(f"Excel report saved to {excel_path}")
        return excel_path
    
    except Exception as e:
        logger.error(f"Error saving Excel file: {str(e)}")
        return None

def save_from_database(connection, session_id, excel_dir='static/excel_outputs', include_timestamp=True):
    """
    Query database and save results to Excel
    
    Args:
        connection: MySQL database connection
        session_id: Session ID to export
        excel_dir: Directory to save Excel files
        include_timestamp: Whether to include timestamp in filename
        
    Returns:
        excel_path: Path to the saved Excel file or None if failed
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(excel_dir, exist_ok=True)
        
        # Create filename with optional timestamp
        if include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"verification_report_{session_id}_{timestamp}.xlsx"
        else:
            filename = f"verification_report_{session_id}.xlsx"
        
        # Full path for Excel file
        excel_path = os.path.join(excel_dir, filename)
        
        # SQL query for full report data
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
        
        # Execute query and create DataFrame
        df = pd.read_sql(query, connection, params=(session_id,))
        
        # Additional sheet with summary data
        summary_query = '''
            SELECT * FROM processing_sessions WHERE session_id = %s
        '''
        summary_df = pd.read_sql(summary_query, connection, params=(session_id,))
        
        # Create Excel writer
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Verification Results', index=False)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        logger.info(f"Excel report from database saved to {excel_path}")
        return excel_path
    
    except Exception as e:
        logger.error(f"Error saving Excel from database: {str(e)}")
        return None

def update_app_process_files(app_py_path):
    """
    Function to update app.py to include auto-save Excel functionality
    This is a helper function and not meant to be called directly in production
    """
    try:
        with open(app_py_path, 'r') as file:
            app_py = file.read()
        
        # Find the location to insert the code
        target_line = "# Save results to database if connection is available"
        
        # Code to insert for auto-saving Excel
        excel_save_code = """
        # Save results to database if connection is available
        if DB_CONNECTION and DB_CONNECTION.is_connected():
            try:
                db_success = db.save_verification_results(DB_CONNECTION, output, session_id)
                if db_success:
                    logger.info(f"Results saved to database successfully for session {session_id}")
                    # Auto-save Excel file
                    import auto_save_excel
                    excel_path = auto_save_excel.save_from_database(DB_CONNECTION, session_id)
                    if excel_path:
                        session['excel_path'] = excel_path
                else:
                    logger.warning(f"Failed to save results to database for session {session_id}")
                    # Fallback to saving Excel directly from results
                    import auto_save_excel
                    excel_path = auto_save_excel.save_to_excel(output, session_id)
                    if excel_path:
                        session['excel_path'] = excel_path
            except Exception as e:
                logger.error(f"Error saving to database: {str(e)}")
                # Fallback to saving Excel directly from results
                import auto_save_excel
                excel_path = auto_save_excel.save_to_excel(output, session_id)
                if excel_path:
                    session['excel_path'] = excel_path
        else:
            # Database not available, save Excel directly from results
            import auto_save_excel
            excel_path = auto_save_excel.save_to_excel(output, session_id)
            if excel_path:
                session['excel_path'] = excel_path
        """
        
        # Replace the target line with the new code
        updated_app_py = app_py.replace(target_line, excel_save_code)
        
        # Write updated app.py
        with open(app_py_path, 'w') as file:
            file.write(updated_app_py)
            
        return True
    except Exception as e:
        logger.error(f"Error updating app.py: {str(e)}")
        return False