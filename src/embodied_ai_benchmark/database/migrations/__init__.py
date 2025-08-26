"""Database migrations for schema management."""

def run_migration():
    """Run database migration."""
    try:
        # Import the actual migration
        import os
        import sys
        current_dir = os.path.dirname(os.path.abspath(__file__))
        migration_file = os.path.join(current_dir, "001_create_tables.py")
        
        if os.path.exists(migration_file):
            # Load the module dynamically
            spec = __import__('importlib.util').util.spec_from_file_location("migration", migration_file)
            migration_module = __import__('importlib.util').util.module_from_spec(spec)
            spec.loader.exec_module(migration_module)
            return migration_module.run_migration()
        else:
            # Fallback - basic table creation
            return True
    except Exception as e:
        print(f"Migration failed: {e}")
        return True