"""Base repository with common CRUD operations."""

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from ..database.connection import DatabaseConnection, get_database


class BaseRepository(ABC):
    """Base repository class with common CRUD operations."""
    
    def __init__(self, db: Optional[DatabaseConnection] = None):
        """Initialize repository with database connection.
        
        Args:
            db: Database connection instance
        """
        self.db = db or get_database()
        self.table_name = self._get_table_name()
        self.primary_key = "id"
    
    @abstractmethod
    def _get_table_name(self) -> str:
        """Get the table name for this repository."""
        pass
    
    def find_all(self, limit: Optional[int] = None, offset: int = 0) -> List[Dict[str, Any]]:
        """Find all records.
        
        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip
            
        Returns:
            List of records as dictionaries
        """
        query = f"SELECT * FROM {self.table_name}"
        
        if limit is not None:
            query += f" LIMIT {limit} OFFSET {offset}"
        
        rows = self.db.execute_query(query)
        return [dict(row) for row in rows]
    
    def find_by_id(self, record_id: int) -> Optional[Dict[str, Any]]:
        """Find record by ID.
        
        Args:
            record_id: Primary key value
            
        Returns:
            Record dictionary or None if not found
        """
        query = f"SELECT * FROM {self.table_name} WHERE {self.primary_key} = ?"
        rows = self.db.execute_query(query, (record_id,))
        
        if rows:
            return dict(rows[0])
        return None
    
    def find_by_field(self, field: str, value: Any) -> List[Dict[str, Any]]:
        """Find records by specific field value.
        
        Args:
            field: Field name to search
            value: Value to match
            
        Returns:
            List of matching records
        """
        if self.db.db_type == "sqlite":
            query = f"SELECT * FROM {self.table_name} WHERE {field} = ?"
            params = (value,)
        else:  # PostgreSQL
            query = f"SELECT * FROM {self.table_name} WHERE {field} = %s"
            params = (value,)
        
        rows = self.db.execute_query(query, params)
        return [dict(row) for row in rows]
    
    def find_by_fields(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find records by multiple field criteria.
        
        Args:
            criteria: Dictionary of field names to values
            
        Returns:
            List of matching records
        """
        if not criteria:
            return self.find_all()
        
        where_clauses = []
        params = []
        
        for field, value in criteria.items():
            if self.db.db_type == "sqlite":
                where_clauses.append(f"{field} = ?")
            else:  # PostgreSQL
                where_clauses.append(f"{field} = %s")
            params.append(value)
        
        query = f"SELECT * FROM {self.table_name} WHERE {' AND '.join(where_clauses)}"
        rows = self.db.execute_query(query, tuple(params))
        return [dict(row) for row in rows]
    
    def create(self, data: Dict[str, Any]) -> Optional[int]:
        """Create new record.
        
        Args:
            data: Record data dictionary
            
        Returns:
            ID of created record or None if failed
        """
        # Remove any None values and prepare data
        clean_data = {k: v for k, v in data.items() if v is not None}
        
        if not clean_data:
            return None
        
        fields = list(clean_data.keys())
        values = list(clean_data.values())
        
        # Convert datetime objects to strings
        for i, value in enumerate(values):
            if isinstance(value, datetime):
                values[i] = value.isoformat()
            elif isinstance(value, (dict, list)):
                values[i] = json.dumps(value)
        
        if self.db.db_type == "sqlite":
            placeholders = ", ".join(["?" for _ in fields])
            query = f"INSERT INTO {self.table_name} ({', '.join(fields)}) VALUES ({placeholders})"
        else:  # PostgreSQL
            placeholders = ", ".join(["%s" for _ in fields])
            query = f"INSERT INTO {self.table_name} ({', '.join(fields)}) VALUES ({placeholders}) RETURNING {self.primary_key}"
        
        return self.db.execute_insert(query, tuple(values))
    
    def update(self, record_id: int, data: Dict[str, Any]) -> bool:
        """Update existing record.
        
        Args:
            record_id: Primary key value
            data: Updated data dictionary
            
        Returns:
            True if record was updated, False otherwise
        """
        # Remove any None values and prepare data
        clean_data = {k: v for k, v in data.items() if v is not None and k != self.primary_key}
        
        if not clean_data:
            return False
        
        # Add updated_at timestamp if column exists
        clean_data['updated_at'] = datetime.now().isoformat()
        
        fields = list(clean_data.keys())
        values = list(clean_data.values())
        
        # Convert complex types to JSON
        for i, value in enumerate(values):
            if isinstance(value, datetime):
                values[i] = value.isoformat()
            elif isinstance(value, (dict, list)):
                values[i] = json.dumps(value)
        
        if self.db.db_type == "sqlite":
            set_clauses = [f"{field} = ?" for field in fields]
            query = f"UPDATE {self.table_name} SET {', '.join(set_clauses)} WHERE {self.primary_key} = ?"
        else:  # PostgreSQL
            set_clauses = [f"{field} = %s" for field in fields]
            query = f"UPDATE {self.table_name} SET {', '.join(set_clauses)} WHERE {self.primary_key} = %s"
        
        values.append(record_id)
        affected_rows = self.db.execute_update(query, tuple(values))
        return affected_rows > 0
    
    def delete(self, record_id: int) -> bool:
        """Delete record by ID.
        
        Args:
            record_id: Primary key value
            
        Returns:
            True if record was deleted, False otherwise
        """
        if self.db.db_type == "sqlite":
            query = f"DELETE FROM {self.table_name} WHERE {self.primary_key} = ?"
        else:  # PostgreSQL
            query = f"DELETE FROM {self.table_name} WHERE {self.primary_key} = %s"
        
        affected_rows = self.db.execute_update(query, (record_id,))
        return affected_rows > 0
    
    def count(self, criteria: Optional[Dict[str, Any]] = None) -> int:
        """Count records matching criteria.
        
        Args:
            criteria: Optional filtering criteria
            
        Returns:
            Number of matching records
        """
        if criteria:
            where_clauses = []
            params = []
            
            for field, value in criteria.items():
                if self.db.db_type == "sqlite":
                    where_clauses.append(f"{field} = ?")
                else:  # PostgreSQL
                    where_clauses.append(f"{field} = %s")
                params.append(value)
            
            query = f"SELECT COUNT(*) as count FROM {self.table_name} WHERE {' AND '.join(where_clauses)}"
            rows = self.db.execute_query(query, tuple(params))
        else:
            query = f"SELECT COUNT(*) as count FROM {self.table_name}"
            rows = self.db.execute_query(query)
        
        return rows[0]['count'] if rows else 0
    
    def exists(self, record_id: int) -> bool:
        """Check if record exists by ID.
        
        Args:
            record_id: Primary key value
            
        Returns:
            True if record exists, False otherwise
        """
        if self.db.db_type == "sqlite":
            query = f"SELECT 1 FROM {self.table_name} WHERE {self.primary_key} = ? LIMIT 1"
        else:  # PostgreSQL
            query = f"SELECT 1 FROM {self.table_name} WHERE {self.primary_key} = %s LIMIT 1"
        
        rows = self.db.execute_query(query, (record_id,))
        return len(rows) > 0
    
    def find_with_pagination(self, 
                           page: int = 1, 
                           page_size: int = 20,
                           criteria: Optional[Dict[str, Any]] = None,
                           order_by: Optional[str] = None) -> Tuple[List[Dict[str, Any]], int]:
        """Find records with pagination.
        
        Args:
            page: Page number (1-based)
            page_size: Number of records per page
            criteria: Optional filtering criteria
            order_by: Optional field to order by
            
        Returns:
            Tuple of (records, total_count)
        """
        offset = (page - 1) * page_size
        
        # Build WHERE clause
        where_clause = ""
        params = []
        
        if criteria:
            where_conditions = []
            for field, value in criteria.items():
                if self.db.db_type == "sqlite":
                    where_conditions.append(f"{field} = ?")
                else:  # PostgreSQL
                    where_conditions.append(f"{field} = %s")
                params.append(value)
            
            if where_conditions:
                where_clause = f"WHERE {' AND '.join(where_conditions)}"
        
        # Build ORDER BY clause
        order_clause = ""
        if order_by:
            order_clause = f"ORDER BY {order_by}"
        
        # Get records
        query = f"SELECT * FROM {self.table_name} {where_clause} {order_clause} LIMIT {page_size} OFFSET {offset}"
        rows = self.db.execute_query(query, tuple(params))
        records = [dict(row) for row in rows]
        
        # Get total count
        count_query = f"SELECT COUNT(*) as count FROM {self.table_name} {where_clause}"
        count_rows = self.db.execute_query(count_query, tuple(params))
        total_count = count_rows[0]['count'] if count_rows else 0
        
        return records, total_count
    
    def bulk_insert(self, records: List[Dict[str, Any]]) -> List[int]:
        """Insert multiple records efficiently.
        
        Args:
            records: List of record dictionaries
            
        Returns:
            List of created record IDs
        """
        if not records:
            return []
        
        # Use the first record to determine fields
        fields = list(records[0].keys())
        created_ids = []
        
        for record in records:
            # Ensure all records have the same fields
            record_data = {field: record.get(field) for field in fields}
            record_id = self.create(record_data)
            if record_id:
                created_ids.append(record_id)
        
        return created_ids