#!/bin/bash

DB_FILE="local_db.sqlite"

if [ -f "$DB_FILE" ]; then
  echo "Database file already exists. Skipping creation."
else
  sqlite3 $DB_FILE <<EOF
    -- Create collection table
    CREATE TABLE collection (
      id TEXT PRIMARY KEY NOT NULL,
      name TEXT NOT NULL,
      createdAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      updatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Set up a trigger to automatically update the updatedAt field
    CREATE TRIGGER update_collection_updatedAt
    AFTER UPDATE ON collection
    FOR EACH ROW
    BEGIN
      UPDATE collection SET updatedAt = CURRENT_TIMESTAMP WHERE id = OLD.id;
    END;
EOF
  echo "Database and table created successfully."
fi
