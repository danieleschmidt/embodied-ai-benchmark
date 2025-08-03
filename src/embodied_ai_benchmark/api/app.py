"""Flask application factory and configuration."""

import os
from flask import Flask, jsonify
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
import logging
from typing import Dict, Any

from .routes import api_bp
from .middleware import setup_middleware
from ..database.connection import get_database, get_cache


def create_app(config: Dict[str, Any] = None) -> Flask:
    """Create and configure Flask application.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured Flask application
    """
    app = Flask(__name__)
    
    # Load configuration
    app.config.from_mapping(
        SECRET_KEY=os.environ.get('SECRET_KEY', 'dev-secret-key'),
        DATABASE_URL=os.environ.get('DATABASE_URL', 'sqlite:///./data/benchmark.db'),
        REDIS_URL=os.environ.get('REDIS_URL', 'redis://localhost:6379/0'),
        DEBUG=os.environ.get('DEBUG', 'false').lower() == 'true',
        TESTING=False,
        MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max upload
        JSON_SORT_KEYS=False,
        JSONIFY_PRETTYPRINT_REGULAR=True
    )
    
    # Override with provided config
    if config:
        app.config.from_mapping(config)
    
    # Setup CORS
    CORS(app, resources={
        r"/api/*": {
            "origins": ["http://localhost:3000", "http://localhost:8080"],
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })
    
    # Setup logging
    if not app.testing:
        logging.basicConfig(
            level=logging.INFO if app.debug else logging.WARNING,
            format='%(asctime)s %(levelname)s: %(message)s'
        )
    
    # Register blueprints
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # Setup middleware
    setup_middleware(app)
    
    # Error handlers
    @app.errorhandler(HTTPException)
    def handle_http_exception(e):
        """Handle HTTP exceptions."""
        return jsonify({
            "error": e.name,
            "message": e.description,
            "status_code": e.code
        }), e.code
    
    @app.errorhandler(Exception)
    def handle_general_exception(e):
        """Handle general exceptions."""
        app.logger.error(f"Unhandled exception: {str(e)}")
        return jsonify({
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "status_code": 500
        }), 500
    
    # Health check endpoint
    @app.route('/health')
    def health_check():
        """Health check endpoint."""
        try:
            # Check database connection
            db = get_database()
            db.execute_query("SELECT 1")
            db_status = "healthy"
        except Exception as e:
            db_status = f"unhealthy: {str(e)}"
        
        try:
            # Check cache connection
            cache = get_cache()
            cache.get("health_check")
            cache_status = "healthy"
        except Exception as e:
            cache_status = f"unhealthy: {str(e)}"
        
        status_code = 200 if "unhealthy" not in f"{db_status} {cache_status}" else 503
        
        return jsonify({
            "status": "healthy" if status_code == 200 else "unhealthy",
            "database": db_status,
            "cache": cache_status,
            "version": "1.0.0"
        }), status_code
    
    # Root endpoint
    @app.route('/')
    def root():
        """Root endpoint with API information."""
        return jsonify({
            "name": "Embodied-AI Benchmark++ API",
            "version": "1.0.0",
            "description": "REST API for managing embodied AI experiments and benchmarks",
            "endpoints": {
                "experiments": "/api/experiments",
                "runs": "/api/runs",
                "agents": "/api/agents",
                "tasks": "/api/tasks",
                "metrics": "/api/metrics",
                "leaderboards": "/api/leaderboards",
                "health": "/health",
                "docs": "/api/docs"
            }
        })
    
    # Initialize database on first request
    @app.before_first_request
    def initialize_app():
        """Initialize application components."""
        try:
            # Ensure database is initialized
            get_database()
            app.logger.info("Database initialized successfully")
            
            # Ensure cache is initialized
            get_cache()
            app.logger.info("Cache initialized successfully")
            
        except Exception as e:
            app.logger.error(f"Failed to initialize app: {str(e)}")
    
    return app


def run_dev_server(host: str = '0.0.0.0', port: int = 8080, debug: bool = True):
    """Run development server.
    
    Args:
        host: Host address
        port: Port number
        debug: Enable debug mode
    """
    app = create_app({'DEBUG': debug})
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_dev_server()
