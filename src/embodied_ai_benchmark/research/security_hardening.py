"""Security Hardening for Research Components."""

import hashlib
import hmac
import secrets
import time
import os
import json
import logging
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import threading
import functools
from datetime import datetime, timedelta

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class SecurityConfig:
    """Configuration for security hardening."""
    enable_encryption: bool = True
    enable_access_control: bool = True
    enable_audit_logging: bool = True
    enable_input_validation: bool = True
    session_timeout_minutes: int = 60
    max_failed_attempts: int = 5
    lockout_duration_minutes: int = 15
    password_min_length: int = 12
    require_strong_passwords: bool = True
    audit_log_path: str = "security_audit.log"
    encryption_key_path: str = "encryption.key"


@dataclass
class SecurityContext:
    """Security context for requests/operations."""
    user_id: str
    session_id: str
    permissions: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


@dataclass
class AuditEvent:
    """Security audit event."""
    timestamp: float
    event_type: str
    user_id: str
    resource: str
    action: str
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)
    risk_level: str = "low"  # low, medium, high, critical


class EncryptionManager:
    """Manages encryption/decryption operations."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.fernet = None
        self.private_key = None
        self.public_key = None
        self._initialize_encryption()
    
    def _initialize_encryption(self):
        """Initialize encryption keys."""
        key_path = Path(self.config.encryption_key_path)
        
        if key_path.exists():
            # Load existing key
            with open(key_path, 'rb') as f:
                key_data = f.read()
            
            try:
                # Try to load as Fernet key
                self.fernet = Fernet(key_data)
                logger.info("Loaded existing encryption key")
            except Exception:
                logger.warning("Failed to load encryption key, generating new one")
                self._generate_new_key()
        else:
            self._generate_new_key()
        
        # Generate RSA key pair for asymmetric encryption
        self._generate_rsa_keys()
    
    def _generate_new_key(self):
        """Generate new encryption key."""
        key = Fernet.generate_key()
        self.fernet = Fernet(key)
        
        # Save key securely
        key_path = Path(self.config.encryption_key_path)
        key_path.parent.mkdir(exist_ok=True)
        
        with open(key_path, 'wb') as f:
            f.write(key)
        
        # Set restrictive permissions
        os.chmod(key_path, 0o600)
        
        logger.info("Generated new encryption key")
    
    def _generate_rsa_keys(self):
        """Generate RSA key pair for asymmetric encryption."""
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.public_key = self.private_key.public_key()
    
    def encrypt_data(self, data: Union[str, bytes]) -> str:
        """Encrypt data using symmetric encryption."""
        if not self.config.enable_encryption:
            return data if isinstance(data, str) else data.decode()
        
        if isinstance(data, str):
            data = data.encode()
        
        encrypted = self.fernet.encrypt(data)
        return base64.b64encode(encrypted).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt data using symmetric encryption."""
        if not self.config.enable_encryption:
            return encrypted_data
        
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode())
            decrypted = self.fernet.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise SecurityError("Failed to decrypt data")
    
    def encrypt_with_public_key(self, data: Union[str, bytes]) -> str:
        """Encrypt data using RSA public key."""
        if isinstance(data, str):
            data = data.encode()
        
        encrypted = self.public_key.encrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return base64.b64encode(encrypted).decode()
    
    def decrypt_with_private_key(self, encrypted_data: str) -> str:
        """Decrypt data using RSA private key."""
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode())
            decrypted = self.private_key.decrypt(
                encrypted_bytes,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            return decrypted.decode()
        except Exception as e:
            logger.error(f"RSA decryption failed: {e}")
            raise SecurityError("Failed to decrypt data with private key")
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure random token."""
        return secrets.token_urlsafe(length)
    
    def hash_password(self, password: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """Hash password using PBKDF2."""
        if salt is None:
            salt = secrets.token_hex(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt.encode(),
            iterations=100000,
        )
        
        key = kdf.derive(password.encode())
        password_hash = base64.b64encode(key).decode()
        
        return password_hash, salt
    
    def verify_password(self, password: str, password_hash: str, salt: str) -> bool:
        """Verify password against hash."""
        computed_hash, _ = self.hash_password(password, salt)
        return hmac.compare_digest(password_hash, computed_hash)


class AccessControlManager:
    """Manages access control and permissions."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.users: Dict[str, Dict[str, Any]] = {}
        self.sessions: Dict[str, SecurityContext] = {}
        self.failed_attempts: Dict[str, List[float]] = {}
        self.locked_accounts: Dict[str, float] = {}
        self._lock = threading.RLock()
    
    def create_user(self, user_id: str, password: str, permissions: List[str]) -> bool:
        """Create a new user account."""
        with self._lock:
            if user_id in self.users:
                logger.warning(f"User {user_id} already exists")
                return False
            
            if not self._validate_password(password):
                logger.warning(f"Password validation failed for user {user_id}")
                return False
            
            # Hash password
            encryption_manager = EncryptionManager(self.config)
            password_hash, salt = encryption_manager.hash_password(password)
            
            self.users[user_id] = {
                "password_hash": password_hash,
                "salt": salt,
                "permissions": permissions,
                "created_at": time.time(),
                "last_login": None,
                "login_count": 0
            }
            
            logger.info(f"Created user: {user_id}")
            return True
    
    def authenticate_user(self, user_id: str, password: str, ip_address: Optional[str] = None) -> Optional[str]:
        """Authenticate user and return session ID."""
        with self._lock:
            # Check if account is locked
            if self._is_account_locked(user_id):
                logger.warning(f"Authentication failed - account locked: {user_id}")
                return None
            
            # Check if user exists
            if user_id not in self.users:
                self._record_failed_attempt(user_id)
                logger.warning(f"Authentication failed - user not found: {user_id}")
                return None
            
            user_data = self.users[user_id]
            
            # Verify password
            encryption_manager = EncryptionManager(self.config)
            if not encryption_manager.verify_password(password, user_data["password_hash"], user_data["salt"]):
                self._record_failed_attempt(user_id)
                logger.warning(f"Authentication failed - invalid password: {user_id}")
                return None
            
            # Clear failed attempts on successful login
            if user_id in self.failed_attempts:
                del self.failed_attempts[user_id]
            
            # Create session
            session_id = encryption_manager.generate_secure_token()
            
            security_context = SecurityContext(
                user_id=user_id,
                session_id=session_id,
                permissions=user_data["permissions"].copy(),
                timestamp=time.time(),
                ip_address=ip_address
            )
            
            self.sessions[session_id] = security_context
            
            # Update user login info
            user_data["last_login"] = time.time()
            user_data["login_count"] += 1
            
            logger.info(f"User authenticated: {user_id}")
            return session_id
    
    def validate_session(self, session_id: str) -> Optional[SecurityContext]:
        """Validate session and return security context."""
        with self._lock:
            if session_id not in self.sessions:
                return None
            
            context = self.sessions[session_id]
            
            # Check session timeout
            if time.time() - context.timestamp > (self.config.session_timeout_minutes * 60):
                del self.sessions[session_id]
                logger.info(f"Session expired: {session_id}")
                return None
            
            # Update timestamp
            context.timestamp = time.time()
            
            return context
    
    def logout_user(self, session_id: str) -> bool:
        """Logout user and invalidate session."""
        with self._lock:
            if session_id in self.sessions:
                user_id = self.sessions[session_id].user_id
                del self.sessions[session_id]
                logger.info(f"User logged out: {user_id}")
                return True
            return False
    
    def check_permission(self, session_id: str, required_permission: str) -> bool:
        """Check if user has required permission."""
        context = self.validate_session(session_id)
        if not context:
            return False
        
        return required_permission in context.permissions or "admin" in context.permissions
    
    def _validate_password(self, password: str) -> bool:
        """Validate password strength."""
        if not self.config.require_strong_passwords:
            return len(password) >= self.config.password_min_length
        
        if len(password) < self.config.password_min_length:
            return False
        
        # Check for uppercase, lowercase, digits, and special characters
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
        
        return has_upper and has_lower and has_digit and has_special
    
    def _record_failed_attempt(self, user_id: str):
        """Record failed authentication attempt."""
        current_time = time.time()
        
        if user_id not in self.failed_attempts:
            self.failed_attempts[user_id] = []
        
        self.failed_attempts[user_id].append(current_time)
        
        # Remove old attempts (older than lockout duration)
        cutoff_time = current_time - (self.config.lockout_duration_minutes * 60)
        self.failed_attempts[user_id] = [
            t for t in self.failed_attempts[user_id] if t > cutoff_time
        ]
        
        # Lock account if too many failed attempts
        if len(self.failed_attempts[user_id]) >= self.config.max_failed_attempts:
            self.locked_accounts[user_id] = current_time
            logger.warning(f"Account locked due to failed attempts: {user_id}")
    
    def _is_account_locked(self, user_id: str) -> bool:
        """Check if account is currently locked."""
        if user_id not in self.locked_accounts:
            return False
        
        lock_time = self.locked_accounts[user_id]
        unlock_time = lock_time + (self.config.lockout_duration_minutes * 60)
        
        if time.time() > unlock_time:
            # Unlock account
            del self.locked_accounts[user_id]
            logger.info(f"Account automatically unlocked: {user_id}")
            return False
        
        return True


class AuditLogger:
    """Logs security-related events for auditing."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.audit_log_path = Path(config.audit_log_path)
        self.audit_log_path.parent.mkdir(exist_ok=True)
        
        # Setup audit logger
        self.audit_logger = logging.getLogger("security_audit")
        self.audit_logger.setLevel(logging.INFO)
        
        # Create file handler if not exists
        if not self.audit_logger.handlers:
            handler = logging.FileHandler(self.audit_log_path)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.audit_logger.addHandler(handler)
    
    def log_event(self, event: AuditEvent):
        """Log security audit event."""
        if not self.config.enable_audit_logging:
            return
        
        event_data = {
            "timestamp": datetime.fromtimestamp(event.timestamp).isoformat(),
            "event_type": event.event_type,
            "user_id": event.user_id,
            "resource": event.resource,
            "action": event.action,
            "success": event.success,
            "risk_level": event.risk_level,
            "details": event.details
        }
        
        log_message = json.dumps(event_data)
        
        if event.risk_level in ["high", "critical"]:
            self.audit_logger.error(log_message)
        elif event.risk_level == "medium":
            self.audit_logger.warning(log_message)
        else:
            self.audit_logger.info(log_message)
    
    def log_authentication(self, user_id: str, success: bool, ip_address: Optional[str] = None):
        """Log authentication event."""
        event = AuditEvent(
            timestamp=time.time(),
            event_type="authentication",
            user_id=user_id,
            resource="auth_system",
            action="login",
            success=success,
            details={"ip_address": ip_address},
            risk_level="medium" if not success else "low"
        )
        self.log_event(event)
    
    def log_permission_check(self, user_id: str, resource: str, permission: str, granted: bool):
        """Log permission check event."""
        event = AuditEvent(
            timestamp=time.time(),
            event_type="permission_check",
            user_id=user_id,
            resource=resource,
            action=permission,
            success=granted,
            risk_level="medium" if not granted else "low"
        )
        self.log_event(event)
    
    def log_data_access(self, user_id: str, resource: str, action: str, success: bool):
        """Log data access event."""
        event = AuditEvent(
            timestamp=time.time(),
            event_type="data_access",
            user_id=user_id,
            resource=resource,
            action=action,
            success=success,
            risk_level="low"
        )
        self.log_event(event)
    
    def log_security_incident(self, user_id: str, incident_type: str, details: Dict[str, Any]):
        """Log security incident."""
        event = AuditEvent(
            timestamp=time.time(),
            event_type="security_incident",
            user_id=user_id,
            resource="security_system",
            action=incident_type,
            success=False,
            details=details,
            risk_level="high"
        )
        self.log_event(event)


class InputValidator:
    """Validates and sanitizes input data."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
    
    def validate_string(self, value: str, max_length: int = 1000, 
                       allowed_chars: Optional[str] = None) -> str:
        """Validate and sanitize string input."""
        if not self.config.enable_input_validation:
            return value
        
        if not isinstance(value, str):
            raise SecurityError("Input must be a string")
        
        if len(value) > max_length:
            raise SecurityError(f"Input too long (max {max_length} characters)")
        
        # Remove null bytes and control characters
        sanitized = value.replace('\x00', '').replace('\r', '').replace('\n', ' ')
        
        # Check allowed characters
        if allowed_chars:
            if not all(c in allowed_chars for c in sanitized):
                raise SecurityError("Input contains disallowed characters")
        
        # Check for common injection patterns
        dangerous_patterns = [
            '<script', 'javascript:', 'vbscript:', 'onload=', 'onerror=',
            'DROP TABLE', 'DELETE FROM', 'INSERT INTO', 'UPDATE SET',
            '../', '..\\', '/etc/passwd', '/bin/sh'
        ]
        
        sanitized_lower = sanitized.lower()
        for pattern in dangerous_patterns:
            if pattern.lower() in sanitized_lower:
                raise SecurityError(f"Input contains potentially dangerous pattern: {pattern}")
        
        return sanitized
    
    def validate_number(self, value: Union[int, float], min_val: Optional[float] = None,
                       max_val: Optional[float] = None) -> Union[int, float]:
        """Validate numeric input."""
        if not self.config.enable_input_validation:
            return value
        
        if not isinstance(value, (int, float)):
            raise SecurityError("Input must be a number")
        
        if min_val is not None and value < min_val:
            raise SecurityError(f"Value too small (min {min_val})")
        
        if max_val is not None and value > max_val:
            raise SecurityError(f"Value too large (max {max_val})")
        
        return value
    
    def validate_list(self, value: List[Any], max_length: int = 1000) -> List[Any]:
        """Validate list input."""
        if not self.config.enable_input_validation:
            return value
        
        if not isinstance(value, list):
            raise SecurityError("Input must be a list")
        
        if len(value) > max_length:
            raise SecurityError(f"List too long (max {max_length} items)")
        
        return value
    
    def validate_dict(self, value: Dict[str, Any], max_keys: int = 100) -> Dict[str, Any]:
        """Validate dictionary input."""
        if not self.config.enable_input_validation:
            return value
        
        if not isinstance(value, dict):
            raise SecurityError("Input must be a dictionary")
        
        if len(value) > max_keys:
            raise SecurityError(f"Dictionary too large (max {max_keys} keys)")
        
        # Validate keys
        for key in value.keys():
            if not isinstance(key, str):
                raise SecurityError("Dictionary keys must be strings")
            
            self.validate_string(key, max_length=100)
        
        return value


class SecurityError(Exception):
    """Custom exception for security-related errors."""
    pass


def require_permission(permission: str):
    """Decorator to require specific permission for function access."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Try to find security context in args/kwargs
            security_context = None
            
            for arg in args:
                if isinstance(arg, SecurityContext):
                    security_context = arg
                    break
            
            if not security_context:
                for value in kwargs.values():
                    if isinstance(value, SecurityContext):
                        security_context = value
                        break
            
            if not security_context:
                raise SecurityError("No security context provided")
            
            if permission not in security_context.permissions and "admin" not in security_context.permissions:
                raise SecurityError(f"Insufficient permissions: {permission} required")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def secure_function(audit_resource: str, audit_action: str, 
                   required_permission: Optional[str] = None):
    """Decorator to add security to functions."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get security hardening instance
            security_hardening = None
            for arg in args:
                if isinstance(arg, SecurityHardening):
                    security_hardening = arg
                    break
            
            if not security_hardening:
                # Try to get from global context or create default
                security_hardening = SecurityHardening(SecurityConfig())
            
            # Get security context
            security_context = kwargs.get('security_context')
            if not security_context:
                for arg in args:
                    if isinstance(arg, SecurityContext):
                        security_context = arg
                        break
            
            if not security_context:
                raise SecurityError("No security context provided")
            
            # Check permission if required
            if required_permission:
                if not security_hardening.check_permission(security_context.session_id, required_permission):
                    security_hardening.audit_logger.log_permission_check(
                        security_context.user_id, audit_resource, required_permission, False
                    )
                    raise SecurityError(f"Permission denied: {required_permission}")
            
            # Execute function
            try:
                result = func(*args, **kwargs)
                
                # Log successful access
                security_hardening.audit_logger.log_data_access(
                    security_context.user_id, audit_resource, audit_action, True
                )
                
                return result
                
            except Exception as e:
                # Log failed access
                security_hardening.audit_logger.log_data_access(
                    security_context.user_id, audit_resource, audit_action, False
                )
                
                # Log security incident if it's a security error
                if isinstance(e, SecurityError):
                    security_hardening.audit_logger.log_security_incident(
                        security_context.user_id, "access_violation", 
                        {"function": func.__name__, "error": str(e)}
                    )
                
                raise
        
        return wrapper
    return decorator


class SecurityHardening:
    """Main security hardening system."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.encryption_manager = EncryptionManager(config)
        self.access_control = AccessControlManager(config)
        self.audit_logger = AuditLogger(config)
        self.input_validator = InputValidator(config)
        
        logger.info("Security hardening initialized")
    
    def create_user(self, user_id: str, password: str, permissions: List[str]) -> bool:
        """Create a new user with security validation."""
        # Validate inputs
        user_id = self.input_validator.validate_string(user_id, max_length=50, 
                                                       allowed_chars="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
        permissions = self.input_validator.validate_list(permissions, max_length=20)
        
        success = self.access_control.create_user(user_id, password, permissions)
        
        # Log user creation
        self.audit_logger.log_event(AuditEvent(
            timestamp=time.time(),
            event_type="user_management",
            user_id="system",
            resource="user_accounts",
            action="create_user",
            success=success,
            details={"target_user": user_id, "permissions": permissions}
        ))
        
        return success
    
    def authenticate(self, user_id: str, password: str, ip_address: Optional[str] = None) -> Optional[str]:
        """Authenticate user with security logging."""
        # Validate inputs
        user_id = self.input_validator.validate_string(user_id, max_length=50)
        
        session_id = self.access_control.authenticate_user(user_id, password, ip_address)
        
        # Log authentication attempt
        self.audit_logger.log_authentication(user_id, session_id is not None, ip_address)
        
        return session_id
    
    def check_permission(self, session_id: str, permission: str) -> bool:
        """Check user permission with audit logging."""
        context = self.access_control.validate_session(session_id)
        if not context:
            return False
        
        granted = self.access_control.check_permission(session_id, permission)
        
        # Log permission check
        self.audit_logger.log_permission_check(context.user_id, "system", permission, granted)
        
        return granted
    
    def logout(self, session_id: str) -> bool:
        """Logout user with audit logging."""
        context = self.access_control.validate_session(session_id)
        success = self.access_control.logout_user(session_id)
        
        if context:
            self.audit_logger.log_event(AuditEvent(
                timestamp=time.time(),
                event_type="authentication",
                user_id=context.user_id,
                resource="auth_system",
                action="logout",
                success=success
            ))
        
        return success
    
    def encrypt_sensitive_data(self, data: Union[str, Dict[str, Any]]) -> str:
        """Encrypt sensitive data."""
        if isinstance(data, dict):
            data = json.dumps(data)
        
        return self.encryption_manager.encrypt_data(data)
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        return self.encryption_manager.decrypt_data(encrypted_data)
    
    def validate_and_sanitize_input(self, data: Any, data_type: str = "string", **validation_params) -> Any:
        """Validate and sanitize input data."""
        if data_type == "string":
            return self.input_validator.validate_string(data, **validation_params)
        elif data_type == "number":
            return self.input_validator.validate_number(data, **validation_params)
        elif data_type == "list":
            return self.input_validator.validate_list(data, **validation_params)
        elif data_type == "dict":
            return self.input_validator.validate_dict(data, **validation_params)
        else:
            raise SecurityError(f"Unsupported data type for validation: {data_type}")
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure token."""
        return self.encryption_manager.generate_secure_token(length)
    
    def get_security_context(self, session_id: str) -> Optional[SecurityContext]:
        """Get security context for session."""
        return self.access_control.validate_session(session_id)
    
    def get_security_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate security report."""
        end_time = time.time()
        start_time = end_time - (hours * 3600)
        
        # Read audit log
        audit_events = []
        try:
            with open(self.audit_logger.audit_log_path, 'r') as f:
                for line in f:
                    try:
                        event_data = json.loads(line.split(' - ', 3)[-1])
                        event_timestamp = datetime.fromisoformat(event_data['timestamp']).timestamp()
                        
                        if event_timestamp >= start_time:
                            audit_events.append(event_data)
                    except (json.JSONDecodeError, ValueError, KeyError):
                        continue
        except FileNotFoundError:
            pass
        
        # Analyze events
        total_events = len(audit_events)
        failed_logins = sum(1 for e in audit_events 
                           if e.get('event_type') == 'authentication' and not e.get('success', True))
        permission_denials = sum(1 for e in audit_events 
                                if e.get('event_type') == 'permission_check' and not e.get('success', True))
        security_incidents = sum(1 for e in audit_events 
                                if e.get('event_type') == 'security_incident')
        
        # Active sessions
        active_sessions = len(self.access_control.sessions)
        locked_accounts = len(self.access_control.locked_accounts)
        
        return {
            "report_period_hours": hours,
            "total_audit_events": total_events,
            "failed_login_attempts": failed_logins,
            "permission_denials": permission_denials,
            "security_incidents": security_incidents,
            "active_sessions": active_sessions,
            "locked_accounts": locked_accounts,
            "users_created": len(self.access_control.users),
            "encryption_enabled": self.config.enable_encryption,
            "access_control_enabled": self.config.enable_access_control,
            "audit_logging_enabled": self.config.enable_audit_logging
        }


def create_security_hardening(config: Optional[SecurityConfig] = None) -> SecurityHardening:
    """Factory function to create security hardening system."""
    if config is None:
        config = SecurityConfig()
    
    security = SecurityHardening(config)
    
    logger.info("Security hardening system created")
    logger.info(f"Encryption: {config.enable_encryption}, Access Control: {config.enable_access_control}")
    logger.info(f"Audit Logging: {config.enable_audit_logging}, Input Validation: {config.enable_input_validation}")
    
    return security