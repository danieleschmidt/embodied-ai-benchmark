"""Middleware for Flask application."""

import time
import uuid
from flask import Flask, request, g, current_app, jsonify
from werkzeug.exceptions import TooManyRequests
from typing import Dict, Any
import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta


class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        """Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests per window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(deque)
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed for client.
        
        Args:
            client_id: Client identifier
            
        Returns:
            True if request is allowed
        """
        now = datetime.now()
        window_start = now - timedelta(seconds=self.window_seconds)
        
        # Clean old requests
        client_requests = self.requests[client_id]
        while client_requests and client_requests[0] < window_start:
            client_requests.popleft()
        
        # Check if within limit
        if len(client_requests) >= self.max_requests:
            return False
        
        # Add current request
        client_requests.append(now)
        return True
    
    def get_reset_time(self, client_id: str) -> datetime:
        """Get when rate limit resets for client.
        
        Args:
            client_id: Client identifier
            
        Returns:
            Reset time
        """
        client_requests = self.requests[client_id]
        if client_requests:
            return client_requests[0] + timedelta(seconds=self.window_seconds)
        return datetime.now()


# Global rate limiter instance
rate_limiter = RateLimiter(max_requests=100, window_seconds=60)


def setup_middleware(app: Flask) -> None:
    """Setup middleware for Flask application.
    
    Args:
        app: Flask application instance
    """
    
    @app.before_request
    def before_request():
        """Execute before each request."""
        # Generate request ID
        g.request_id = str(uuid.uuid4())
        g.start_time = time.time()
        
        # Get client IP
        client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
        if client_ip:
            client_ip = client_ip.split(',')[0].strip()
        g.client_ip = client_ip or 'unknown'
        
        # Rate limiting (skip for health checks)
        if not request.path.startswith('/health'):
            if not rate_limiter.is_allowed(g.client_ip):
                reset_time = rate_limiter.get_reset_time(g.client_ip)
                response = jsonify({
                    "error": "Rate limit exceeded",
                    "message": "Too many requests. Please try again later.",
                    "retry_after": int((reset_time - datetime.now()).total_seconds())
                })
                response.status_code = 429
                response.headers['Retry-After'] = str(int((reset_time - datetime.now()).total_seconds()))
                return response
        
        # Log request
        if not app.testing:
            current_app.logger.info(
                f"Request {g.request_id}: {request.method} {request.path} from {g.client_ip}"
            )
    
    @app.after_request
    def after_request(response):
        """Execute after each request.
        
        Args:
            response: Flask response object
            
        Returns:
            Modified response object
        """
        # Add request ID to response headers
        if hasattr(g, 'request_id'):
            response.headers['X-Request-ID'] = g.request_id
        
        # Add CORS headers for API endpoints
        if request.path.startswith('/api/'):
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        
        # Add security headers
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        
        # Log response
        if not app.testing and hasattr(g, 'start_time'):
            duration = time.time() - g.start_time
            current_app.logger.info(
                f"Response {g.request_id}: {response.status_code} in {duration:.3f}s"
            )
        
        return response
    
    @app.errorhandler(429)
    def handle_rate_limit(error):
        """Handle rate limit errors.
        
        Args:
            error: Rate limit error
            
        Returns:
            JSON error response
        """
        return jsonify({
            "error": "Rate limit exceeded",
            "message": "Too many requests. Please try again later.",
            "status_code": 429
        }), 429


class RequestLogger:
    """Custom request logger for detailed logging."""
    
    def __init__(self, app: Flask = None):
        """Initialize request logger.
        
        Args:
            app: Flask application instance
        """
        self.app = app
        if app:
            self.init_app(app)
    
    def init_app(self, app: Flask) -> None:
        """Initialize with Flask app.
        
        Args:
            app: Flask application instance
        """
        app.before_request(self.log_request)
        app.after_request(self.log_response)
    
    def log_request(self) -> None:
        """Log incoming request details."""
        if current_app.testing:
            return
        
        request_data = {
            "request_id": getattr(g, 'request_id', 'unknown'),
            "method": request.method,
            "path": request.path,
            "query_string": request.query_string.decode('utf-8'),
            "client_ip": getattr(g, 'client_ip', 'unknown'),
            "user_agent": request.headers.get('User-Agent', 'unknown'),
            "content_type": request.headers.get('Content-Type', 'unknown'),
            "content_length": request.headers.get('Content-Length', 0)
        }
        
        current_app.logger.info(f"Request details: {request_data}")
    
    def log_response(self, response) -> Any:
        """Log response details.
        
        Args:
            response: Flask response object
            
        Returns:
            Response object
        """
        if current_app.testing:
            return response
        
        if hasattr(g, 'start_time'):
            duration = time.time() - g.start_time
            
            response_data = {
                "request_id": getattr(g, 'request_id', 'unknown'),
                "status_code": response.status_code,
                "content_type": response.headers.get('Content-Type', 'unknown'),
                "content_length": response.headers.get('Content-Length', 0),
                "duration_ms": round(duration * 1000, 2)
            }
            
            current_app.logger.info(f"Response details: {response_data}")
        
        return response


class SecurityHeaders:
    """Middleware for adding security headers."""
    
    def __init__(self, app: Flask = None):
        """Initialize security headers middleware.
        
        Args:
            app: Flask application instance
        """
        self.app = app
        if app:
            self.init_app(app)
    
    def init_app(self, app: Flask) -> None:
        """Initialize with Flask app.
        
        Args:
            app: Flask application instance
        """
        app.after_request(self.add_security_headers)
    
    def add_security_headers(self, response) -> Any:
        """Add security headers to response.
        
        Args:
            response: Flask response object
            
        Returns:
            Response with security headers
        """
        # Prevent MIME type sniffing
        response.headers['X-Content-Type-Options'] = 'nosniff'
        
        # Prevent clickjacking
        response.headers['X-Frame-Options'] = 'DENY'
        
        # Enable XSS protection
        response.headers['X-XSS-Protection'] = '1; mode=block'
        
        # Content Security Policy (basic)
        if request.path.startswith('/api/'):
            response.headers['Content-Security-Policy'] = "default-src 'self'"
        
        # Strict Transport Security (HTTPS only)
        if request.is_secure:
            response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        
        # Referrer Policy
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        
        # Permissions Policy (formerly Feature Policy)
        response.headers['Permissions-Policy'] = 'geolocation=(), microphone=(), camera=()'
        
        return response


class RequestSizeLimit:
    """Middleware for limiting request size."""
    
    def __init__(self, app: Flask = None, max_size: int = 16 * 1024 * 1024):
        """Initialize request size limit middleware.
        
        Args:
            app: Flask application instance
            max_size: Maximum request size in bytes (default 16MB)
        """
        self.max_size = max_size
        self.app = app
        if app:
            self.init_app(app)
    
    def init_app(self, app: Flask) -> None:
        """Initialize with Flask app.
        
        Args:
            app: Flask application instance
        """
        app.before_request(self.check_request_size)
    
    def check_request_size(self) -> Any:
        """Check if request size is within limits.
        
        Returns:
            Error response if request is too large
        """
        content_length = request.headers.get('Content-Length')
        
        if content_length:
            try:
                content_length = int(content_length)
                if content_length > self.max_size:
                    return jsonify({
                        "error": "Request too large",
                        "message": f"Request size ({content_length} bytes) exceeds maximum allowed size ({self.max_size} bytes)",
                        "status_code": 413
                    }), 413
            except ValueError:
                pass


# Convenience function to setup all middleware
def setup_all_middleware(app: Flask) -> None:
    """Setup all middleware for Flask application.
    
    Args:
        app: Flask application instance
    """
    setup_middleware(app)
    RequestLogger(app)
    SecurityHeaders(app)
    RequestSizeLimit(app)
