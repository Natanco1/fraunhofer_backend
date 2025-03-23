import sqlite3
from django.conf import settings
import logging
import os

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(settings.BASE_DIR, 'local_db.sqlite')

table = 'collection'

def insert_collection_record(request_id, name=None):
    """Insert a new record into the collection table with optional name."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute(f'''
            INSERT INTO {table} (id, name, createdAt, updatedAt)
            VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        ''', (request_id, name if name else request_id))

        conn.commit()
        conn.close()

    except sqlite3.Error as e:
        logger.error(f"Error inserting into collection table: {e}")
        raise Exception("Failed to insert data into the database.")

        
def get_collection_record_by_id(request_id):
    """Fetch a record from the collection table by id."""
    try:
        request_id_str = str(request_id)

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute(f'''
            SELECT * FROM {table} WHERE id = ?
        ''', (request_id_str,))
        
        record = cursor.fetchone()
        conn.close()

        if record:
            return record
        else:
            return None

    except sqlite3.Error as e:
        logger.error(f"Error fetching collection with ID {request_id}: {e}")
        raise Exception("Failed to retrieve data from the database.")



def get_all_collections():
    """Fetch all records from the collection table."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute(f'''
            SELECT * FROM {table}
        ''')

        records = cursor.fetchall()
        conn.close()

        return records

    except sqlite3.Error as e:
        logger.error(f"Error fetching all from collection table: {e}")
        raise Exception("Failed to retrieve data from the database.")

def update_collection_name(request_id, new_name):
    """Update the name of a collection record by id."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute(f'''
            UPDATE {table}
            SET name = ?, updatedAt = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (new_name, request_id))

        conn.commit()
        conn.close()

    except sqlite3.Error as e:
        logger.error(f"Error updating collection name: {e}")
        raise Exception("Failed to update collection data in the database.")
