"""Production security framework with input validation, authentication, and threat detection."""

import hashlib
import secrets
import time
import json
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from functools import wraps
import jwt
from cryptography.fernet import Fernet


@dataclass
class SecurityEvent:
    """Security event record."""
    event_type: str
    severity: str  # "low", "medium", "high", "critical"
    source_ip: str
    user_id: Optional[str]
    timestamp: datetime
    details: Dict[str, Any]
    mitigated: bool = False


@dataclass
class ValidationRule:
    """Input validation rule."""
    name: str
    validator: Callable[[Any], Tuple[bool, str]]
    required: bool = True
    sanitizer: Optional[Callable[[Any], Any]] = None


class SecurityFramework:
    """Comprehensive security framework for production deployment."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize security framework.
        
        Args:
            config: Security configuration
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Security settings
        self.jwt_secret_key = self.config.get("jwt_secret_key", secrets.token_urlsafe(32))
        self.jwt_algorithm = self.config.get("jwt_algorithm", "HS256")
        self.jwt_expiration_hours = self.config.get("jwt_expiration_hours", 24)
        
        # Encryption
        self.encryption_key = self.config.get("encryption_key", Fernet.generate_key())
        self.fernet = Fernet(self.encryption_key)
        
        # Rate limiting
        self.rate_limits = {}
        self.request_counts = {}
        self.blocked_ips = set()
        
        # Security events
        self.security_events = []
        self.max_security_events = self.config.get("max_security_events", 10000)
        
        # Validation rules
        self.validation_rules = {}
        self._setup_default_validation_rules()
        
        # Threat detection
        self.threat_patterns = self._load_threat_patterns()
        self.suspicious_activities = {}
        
        self.logger.info("Security framework initialized")
    
    def _setup_default_validation_rules(self):
        """Setup default input validation rules."""
        
        # String validation
        def validate_string(value, max_length=1000, min_length=0):
            def validator(val):
                if not isinstance(val, str):
                    return False, "Must be a string"
                if len(val) < min_length:
                    return False, f"Must be at least {min_length} characters"
                if len(val) > max_length:
                    return False, f"Must be at most {max_length} characters"
                return True, "Valid"
            return validator
        
        # Email validation
        def validate_email(value):
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not isinstance(value, str):
                return False, "Must be a string"
            if not re.match(email_pattern, value):
                return False, "Invalid email format"
            return True, "Valid email"
        
        # Numeric validation
        def validate_number(value, min_val=None, max_val=None):
            def validator(val):
                if not isinstance(val, (int, float)):
                    return False, "Must be a number"
                if min_val is not None and val < min_val:
                    return False, f"Must be at least {min_val}"
                if max_val is not None and val > max_val:
                    return False, f"Must be at most {max_val}"
                return True, "Valid"
            return validator
        
        # Array validation
        def validate_array(value, max_items=100, item_validator=None):
            def validator(val):
                if not isinstance(val, list):
                    return False, "Must be an array"
                if len(val) > max_items:
                    return False, f"Must have at most {max_items} items"
                if item_validator:
                    for i, item in enumerate(val):
                        valid, msg = item_validator(item)
                        if not valid:
                            return False, f"Item {i}: {msg}"
                return True, "Valid"
            return validator
        
        # Register default rules
        self.register_validation_rule("string", ValidationRule(
            "string", validate_string(), sanitizer=self._sanitize_string
        ))
        
        self.register_validation_rule("email", ValidationRule(
            "email", validate_email, sanitizer=self._sanitize_email
        ))
        
        self.register_validation_rule("number", ValidationRule(
            "number", validate_number()
        ))
        
        self.register_validation_rule("array", ValidationRule(
            "array", validate_array()
        ))
    
    def register_validation_rule(self, name: str, rule: ValidationRule):
        """Register a validation rule."""
        self.validation_rules[name] = rule
        self.logger.info(f"Registered validation rule: {name}")
    
    def validate_input(self, 
                      data: Dict[str, Any], 
                      schema: Dict[str, str],
                      sanitize: bool = True) -> Tuple[bool, Dict[str, Any], List[str]]:
        """Validate input data against schema.
        
        Args:
            data: Input data to validate
            schema: Validation schema (field -> rule_name)
            sanitize: Whether to sanitize input
            
        Returns:
            Tuple of (is_valid, sanitized_data, error_messages)
        """
        errors = []
        sanitized_data = {}
        
        # Check for required fields
        for field, rule_name in schema.items():
            if rule_name not in self.validation_rules:
                errors.append(f"Unknown validation rule: {rule_name}")
                continue
            
            rule = self.validation_rules[rule_name]
            
            # Check if field is present
            if field not in data:
                if rule.required:
                    errors.append(f"Required field missing: {field}")
                continue
            
            value = data[field]
            
            # Validate
            is_valid, message = rule.validator(value)
            if not is_valid:
                errors.append(f"{field}: {message}")
                continue
            
            # Sanitize if requested and sanitizer available
            if sanitize and rule.sanitizer:
                sanitized_value = rule.sanitizer(value)
            else:
                sanitized_value = value
            
            sanitized_data[field] = sanitized_value
        
        return len(errors) == 0, sanitized_data, errors
    
    def _sanitize_string(self, value: str) -> str:
        """Sanitize string input."""
        # Remove potential XSS characters
        value = value.replace('<', '&lt;').replace('>', '&gt;')
        value = value.replace('"', '&quot;').replace("'", '&#x27;')
        
        # Remove null bytes
        value = value.replace('\x00', '')
        
        # Limit length
        if len(value) > 10000:
            value = value[:10000]
        
        return value.strip()
    
    def _sanitize_email(self, value: str) -> str:
        """Sanitize email input."""
        return value.lower().strip()
    
    def generate_jwt_token(self, 
                          user_id: str, 
                          additional_claims: Dict[str, Any] = None) -> str:
        """Generate JWT token for user.
        
        Args:
            user_id: User identifier
            additional_claims: Additional claims to include
            
        Returns:
            JWT token string
        """
        now = datetime.now(timezone.utc)
        expiration = now + timedelta(hours=self.jwt_expiration_hours)
        
        payload = {
            "user_id": user_id,
            "iat": now.timestamp(),
            "exp": expiration.timestamp(),
            "jti": secrets.token_urlsafe(16)  # JWT ID
        }
        
        if additional_claims:
            payload.update(additional_claims)
        
        token = jwt.encode(payload, self.jwt_secret_key, algorithm=self.jwt_algorithm)
        
        self.logger.info(f"JWT token generated for user: {user_id}")
        return token
    
    def verify_jwt_token(self, token: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Verify JWT token.
        
        Args:
            token: JWT token to verify
            
        Returns:
            Tuple of (is_valid, payload)
        """
        try:
            payload = jwt.decode(
                token, 
                self.jwt_secret_key, 
                algorithms=[self.jwt_algorithm]
            )
            return True, payload
            
        except jwt.ExpiredSignatureError:
            self.log_security_event("token_expired", "low", None, None, {
                "token": token[:20] + "..."
            })
            return False, None
            
        except jwt.InvalidTokenError as e:
            self.log_security_event("invalid_token", "medium", None, None, {
                "error": str(e),
                "token": token[:20] + "..."
            })
            return False, None
    
    def encrypt_sensitive_data(self, data: Union[str, Dict[str, Any]]) -> str:
        """Encrypt sensitive data.
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data as base64 string
        """
        if isinstance(data, dict):
            data_str = json.dumps(data)
        else:
            data_str = str(data)
        
        encrypted = self.fernet.encrypt(data_str.encode())
        return encrypted.decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> Union[str, Dict[str, Any]]:
        """Decrypt sensitive data.
        
        Args:
            encrypted_data: Encrypted data as base64 string
            
        Returns:
            Decrypted data
        """
        try:
            decrypted = self.fernet.decrypt(encrypted_data.encode())
            data_str = decrypted.decode()
            
            # Try to parse as JSON
            try:
                return json.loads(data_str)
            except json.JSONDecodeError:
                return data_str
                
        except Exception as e:
            self.logger.error(f"Failed to decrypt data: {e}")
            raise ValueError("Failed to decrypt data")
    
    def hash_password(self, password: str) -> str:
        """Hash password securely.
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password
        """
        salt = secrets.token_urlsafe(16)
        pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return f"{salt}${pwd_hash.hex()}"
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash.
        
        Args:
            password: Plain text password
            hashed: Hashed password from storage
            
        Returns:
            Whether password matches
        """
        try:
            salt, stored_hash = hashed.split('$')
            pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
            return pwd_hash.hex() == stored_hash
        except Exception:
            return False
    
    def check_rate_limit(self, 
                        identifier: str, 
                        max_requests: int = 100, 
                        window_seconds: int = 3600) -> Tuple[bool, Dict[str, Any]]:
        """Check rate limit for identifier.
        
        Args:
            identifier: Identifier to check (IP, user ID, etc.)
            max_requests: Maximum requests allowed
            window_seconds: Time window in seconds
            
        Returns:
            Tuple of (allowed, info)
        """
        current_time = time.time()
        window_key = f"{identifier}:{int(current_time // window_seconds)}"
        
        # Initialize request count if not exists
        if window_key not in self.request_counts:
            self.request_counts[window_key] = 0
        
        self.request_counts[window_key] += 1
        
        # Clean up old windows
        self._cleanup_rate_limit_data(current_time, window_seconds)
        
        allowed = self.request_counts[window_key] <= max_requests
        
        info = {
            "current_count": self.request_counts[window_key],
            "max_requests": max_requests,
            "window_seconds": window_seconds,
            "reset_time": (int(current_time // window_seconds) + 1) * window_seconds
        }
        
        if not allowed:
            self.log_security_event("rate_limit_exceeded", "medium", identifier, None, info)
        
        return allowed, info
    
    def _cleanup_rate_limit_data(self, current_time: float, window_seconds: int):
        """Clean up old rate limit data."""
        current_window = int(current_time // window_seconds)
        
        # Remove data older than 2 windows
        keys_to_remove = [
            key for key in self.request_counts.keys()
            if int(key.split(':')[1]) < current_window - 1
        ]
        
        for key in keys_to_remove:
            del self.request_counts[key]
    
    def block_ip(self, ip_address: str, reason: str = "Security violation"):
        """Block IP address.
        
        Args:
            ip_address: IP address to block
            reason: Reason for blocking
        """
        self.blocked_ips.add(ip_address)
        self.log_security_event("ip_blocked", "high", ip_address, None, {
            "reason": reason
        })
        self.logger.warning(f"Blocked IP address {ip_address}: {reason}")
    
    def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP address is blocked.
        
        Args:
            ip_address: IP address to check
            
        Returns:
            Whether IP is blocked
        """
        return ip_address in self.blocked_ips
    
    def unblock_ip(self, ip_address: str):
        """Unblock IP address.
        
        Args:
            ip_address: IP address to unblock
        """
        self.blocked_ips.discard(ip_address)
        self.logger.info(f"Unblocked IP address: {ip_address}")
    
    def log_security_event(self, 
                          event_type: str,
                          severity: str,
                          source_ip: Optional[str],
                          user_id: Optional[str],
                          details: Dict[str, Any]):
        """Log security event.
        
        Args:
            event_type: Type of security event
            severity: Event severity (low, medium, high, critical)
            source_ip: Source IP address
            user_id: User ID if applicable
            details: Additional event details
        """
        event = SecurityEvent(
            event_type=event_type,
            severity=severity,
            source_ip=source_ip or "unknown",
            user_id=user_id,
            timestamp=datetime.now(timezone.utc),
            details=details
        )
        
        self.security_events.append(event)
        
        # Limit event history
        if len(self.security_events) > self.max_security_events:
            self.security_events = self.security_events[-self.max_security_events:]
        
        # Log based on severity
        log_level = {
            "low": logging.INFO,
            "medium": logging.WARNING,
            "high": logging.ERROR,
            "critical": logging.CRITICAL
        }.get(severity, logging.INFO)
        
        self.logger.log(log_level, f"Security event: {event_type} - {details}")
    
    def _load_threat_patterns(self) -> Dict[str, List[str]]:
        """Load threat detection patterns."""
        return {
            "sql_injection": [
                r"union\s+select",
                r"drop\s+table",
                r"insert\s+into",
                r"delete\s+from",
                r"update\s+set",
                r"exec\s*\(",
                r"script\s*>",
                r"--\s*$"
            ],
            "xss": [
                r"<script[^>]*>",
                r"javascript:",
                r"on\w+\s*=",
                r"<iframe[^>]*>",
                r"<object[^>]*>",
                r"<embed[^>]*>"
            ],
            "command_injection": [
                r";\s*rm\s+",
                r";\s*cat\s+",
                r";\s*ls\s+",
                r"\|\s*nc\s+",
                r"&&\s*wget",
                r";\s*curl\s+"
            ]
        }
    
    def detect_threats(self, input_data: str) -> List[Tuple[str, str]]:
        """Detect security threats in input data.
        
        Args:
            input_data: Input data to analyze
            
        Returns:
            List of (threat_type, matched_pattern) tuples
        """
        threats = []
        input_lower = input_data.lower()
        
        for threat_type, patterns in self.threat_patterns.items():
            for pattern in patterns:
                if re.search(pattern, input_lower, re.IGNORECASE):
                    threats.append((threat_type, pattern))
        
        return threats
    
    def analyze_request_security(self, 
                               request_data: Dict[str, Any],
                               source_ip: str,
                               user_id: Optional[str] = None) -> Dict[str, Any]:
        """Analyze request for security threats.
        
        Args:
            request_data: Request data to analyze
            source_ip: Source IP address
            user_id: User ID if authenticated
            
        Returns:
            Security analysis results
        """
        analysis = {
            "safe": True,
            "threats": [],
            "blocked": False,
            "recommendations": []
        }
        
        # Check if IP is blocked
        if self.is_ip_blocked(source_ip):
            analysis["safe"] = False
            analysis["blocked"] = True
            analysis["recommendations"].append("IP address is blocked")
            return analysis
        
        # Check rate limits
        allowed, rate_info = self.check_rate_limit(source_ip)
        if not allowed:
            analysis["safe"] = False
            analysis["recommendations"].append("Rate limit exceeded")
        
        # Analyze all string values for threats
        def analyze_value(value, path=""):
            if isinstance(value, str):
                threats = self.detect_threats(value)
                for threat_type, pattern in threats:
                    analysis["threats"].append({
                        "type": threat_type,
                        "pattern": pattern,
                        "path": path,
                        "value_preview": value[:50] + "..." if len(value) > 50 else value
                    })
                    analysis["safe"] = False
            elif isinstance(value, dict):
                for key, sub_value in value.items():
                    analyze_value(sub_value, f"{path}.{key}" if path else key)
            elif isinstance(value, list):
                for i, sub_value in enumerate(value):
                    analyze_value(sub_value, f"{path}[{i}]" if path else f"[{i}]")
        
        analyze_value(request_data)
        
        # Log threats if found
        if analysis["threats"]:
            self.log_security_event(
                "threats_detected",
                "high",
                source_ip,
                user_id,
                {
                    "threat_count": len(analysis["threats"]),
                    "threat_types": list(set(t["type"] for t in analysis["threats"])),
                    "request_data": str(request_data)[:200] + "..."
                }
            )
        
        return analysis
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security summary and statistics."""
        now = datetime.now(timezone.utc)
        last_24h = now - timedelta(days=1)
        
        recent_events = [e for e in self.security_events if e.timestamp >= last_24h]
        
        return {
            "total_security_events": len(self.security_events),
            "recent_events_24h": len(recent_events),
            "blocked_ips_count": len(self.blocked_ips),
            "blocked_ips": list(self.blocked_ips),
            "event_types": {
                event_type: len([e for e in recent_events if e.event_type == event_type])
                for event_type in set(e.event_type for e in recent_events)
            },
            "severity_distribution": {
                severity: len([e for e in recent_events if e.severity == severity])
                for severity in ["low", "medium", "high", "critical"]
            },
            "validation_rules_count": len(self.validation_rules),
            "threat_patterns_count": sum(len(patterns) for patterns in self.threat_patterns.values())
        }


def secure_endpoint(security_framework: SecurityFramework = None,
                   require_auth: bool = True,
                   rate_limit_requests: int = 100,
                   rate_limit_window: int = 3600,
                   validate_input: Dict[str, str] = None):
    """Decorator to secure API endpoints.
    
    Args:
        security_framework: Security framework instance
        require_auth: Whether authentication is required
        rate_limit_requests: Rate limit max requests
        rate_limit_window: Rate limit window in seconds
        validate_input: Input validation schema
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # This would need to be integrated with actual web framework
            # For now, just demonstrate the concept
            
            # Extract request info (would come from Flask/FastAPI/etc.)
            request_data = kwargs.get('request_data', {})
            source_ip = kwargs.get('source_ip', '127.0.0.1')
            auth_token = kwargs.get('auth_token')
            
            sf = security_framework or SecurityFramework()
            
            # Check rate limit
            allowed, _ = sf.check_rate_limit(source_ip, rate_limit_requests, rate_limit_window)
            if not allowed:
                raise ValueError("Rate limit exceeded")
            
            # Check authentication
            if require_auth:
                if not auth_token:
                    raise ValueError("Authentication required")
                
                valid, payload = sf.verify_jwt_token(auth_token)
                if not valid:
                    raise ValueError("Invalid authentication token")
                
                kwargs['user_id'] = payload.get('user_id')
            
            # Validate input
            if validate_input and request_data:
                valid, sanitized_data, errors = sf.validate_input(request_data, validate_input)
                if not valid:
                    raise ValueError(f"Input validation failed: {'; '.join(errors)}")
                kwargs['request_data'] = sanitized_data
            
            # Analyze security
            security_analysis = sf.analyze_request_security(request_data, source_ip, kwargs.get('user_id'))
            if not security_analysis['safe']:
                sf.log_security_event("unsafe_request_blocked", "high", source_ip, kwargs.get('user_id'), security_analysis)
                raise ValueError("Request blocked due to security concerns")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator