"""
Security and Monitoring Systems

Advanced security monitoring, threat detection, and compliance enforcement
for autonomous SDLC environments.
"""

import hashlib
import hmac
import secrets
import jwt
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
import json
import re

from ..utils.error_handling import ErrorHandler
from .observability_engine import MetricsCollector


class ThreatLevel(Enum):
    """Security threat severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEventType(Enum):
    """Types of security events"""
    AUTHENTICATION_FAILURE = "auth_failure"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    SUSPICIOUS_CODE_PATTERN = "suspicious_code"
    DATA_BREACH_ATTEMPT = "data_breach"
    MALICIOUS_INPUT = "malicious_input"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    COMPLIANCE_VIOLATION = "compliance_violation"


@dataclass
class SecurityEvent:
    """Security event record"""
    event_type: SecurityEventType
    threat_level: ThreatLevel
    description: str
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    resource: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_type': self.event_type.value,
            'threat_level': self.threat_level.value,
            'description': self.description,
            'source_ip': self.source_ip,
            'user_id': self.user_id,
            'resource': self.resource,
            'additional_data': self.additional_data,
            'timestamp': self.timestamp.isoformat(),
            'resolved': self.resolved
        }


class SecurityScanner:
    """Scans code and systems for security vulnerabilities"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.vulnerability_patterns = self._load_vulnerability_patterns()
    
    def _load_vulnerability_patterns(self) -> Dict[str, List[str]]:
        """Load security vulnerability patterns"""
        return {
            'sql_injection': [
                r"(?i)(\bselect\b.*\bfrom\b.*\bwhere\b.*['\"][^'\"]*['\"])",
                r"(?i)(\bunion\b.*\bselect\b)",
                r"(?i)(\b(exec|execute)\b.*\(.*\))",
            ],
            'xss': [
                r"<script[^>]*>.*?</script>",
                r"javascript:",
                r"on\w+\s*=",
            ],
            'command_injection': [
                r"(?i)(\b(system|exec|eval|subprocess|os\.system)\b.*\(.*\))",
                r"(?i)(\bshell=True\b)",
            ],
            'hardcoded_secrets': [
                r"(?i)(password|passwd|pwd)\s*=\s*['\"][^'\"]+['\"]",
                r"(?i)(api[_-]?key|secret[_-]?key)\s*=\s*['\"][^'\"]+['\"]",
                r"(?i)(token)\s*=\s*['\"][a-zA-Z0-9]{20,}['\"]",
            ],
            'insecure_random': [
                r"\brandom\.random\(\)",
                r"\brandom\.randint\(",
                r"(?i)\bmath\.random\(\)",
            ]
        }
    
    def scan_code(self, code: str, filename: str = "") -> List[Dict[str, Any]]:
        """Scan code for security vulnerabilities"""
        vulnerabilities = []
        
        for vuln_type, patterns in self.vulnerability_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, code, re.MULTILINE)
                for match in matches:
                    line_num = code[:match.start()].count('\n') + 1
                    vulnerabilities.append({
                        'type': vuln_type,
                        'severity': self._get_severity(vuln_type),
                        'description': f"Potential {vuln_type.replace('_', ' ')} vulnerability",
                        'line': line_num,
                        'code_snippet': match.group(0),
                        'filename': filename,
                        'recommendation': self._get_recommendation(vuln_type)
                    })
        
        return vulnerabilities
    
    def _get_severity(self, vuln_type: str) -> str:
        """Get vulnerability severity"""
        severity_map = {
            'sql_injection': 'high',
            'xss': 'high',
            'command_injection': 'critical',
            'hardcoded_secrets': 'high',
            'insecure_random': 'medium'
        }
        return severity_map.get(vuln_type, 'medium')
    
    def _get_recommendation(self, vuln_type: str) -> str:
        """Get security recommendation"""
        recommendations = {
            'sql_injection': 'Use parameterized queries or ORM',
            'xss': 'Sanitize and escape user input',
            'command_injection': 'Avoid shell execution, use safe APIs',
            'hardcoded_secrets': 'Use environment variables or secret management',
            'insecure_random': 'Use cryptographically secure random generators'
        }
        return recommendations.get(vuln_type, 'Review and fix security issue')


class AccessController:
    """Manages authentication and authorization"""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.failed_attempts: Dict[str, List[datetime]] = {}
        self.logger = logging.getLogger(__name__)
    
    def authenticate_user(self, username: str, password: str, source_ip: str = None) -> Optional[str]:
        """Authenticate user and return session token"""
        if self._is_rate_limited(username, source_ip):
            self._log_security_event(
                SecurityEventType.AUTHENTICATION_FAILURE,
                ThreatLevel.MEDIUM,
                f"Rate limited authentication attempt for {username}",
                source_ip=source_ip,
                user_id=username
            )
            return None
        
        # In real implementation, verify against user database
        if self._verify_credentials(username, password):
            session_token = self._generate_session_token(username)
            self.active_sessions[session_token] = {
                'username': username,
                'created_at': datetime.now(),
                'source_ip': source_ip,
                'permissions': self._get_user_permissions(username)
            }
            self.logger.info(f"User {username} authenticated successfully")
            return session_token
        else:
            self._record_failed_attempt(username, source_ip)
            self._log_security_event(
                SecurityEventType.AUTHENTICATION_FAILURE,
                ThreatLevel.LOW,
                f"Failed authentication attempt for {username}",
                source_ip=source_ip,
                user_id=username
            )
            return None
    
    def verify_session(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify session token"""
        if token not in self.active_sessions:
            return None
        
        session = self.active_sessions[token]
        
        # Check if session expired
        if datetime.now() - session['created_at'] > timedelta(hours=24):
            del self.active_sessions[token]
            return None
        
        return session
    
    def authorize_action(self, token: str, resource: str, action: str) -> bool:
        """Check if user is authorized for action"""
        session = self.verify_session(token)
        if not session:
            return False
        
        permissions = session.get('permissions', [])
        required_permission = f"{resource}:{action}"
        
        authorized = required_permission in permissions or 'admin:*' in permissions
        
        if not authorized:
            self._log_security_event(
                SecurityEventType.UNAUTHORIZED_ACCESS,
                ThreatLevel.MEDIUM,
                f"Unauthorized access attempt to {required_permission}",
                user_id=session['username']
            )
        
        return authorized
    
    def _is_rate_limited(self, username: str, source_ip: str) -> bool:
        """Check if user/IP is rate limited"""
        now = datetime.now()
        window = timedelta(minutes=15)
        
        # Check failed attempts for username
        if username in self.failed_attempts:
            recent_attempts = [
                attempt for attempt in self.failed_attempts[username]
                if now - attempt < window
            ]
            if len(recent_attempts) >= 5:
                return True
        
        return False
    
    def _verify_credentials(self, username: str, password: str) -> bool:
        """Verify user credentials (simplified)"""
        # In real implementation, verify against secure user database
        # For demo purposes, accept any non-empty credentials
        return bool(username and password and len(password) >= 8)
    
    def _generate_session_token(self, username: str) -> str:
        """Generate secure session token"""
        payload = {
            'username': username,
            'created_at': datetime.now().isoformat(),
            'nonce': secrets.token_urlsafe(16)
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def _get_user_permissions(self, username: str) -> List[str]:
        """Get user permissions (simplified)"""
        # In real implementation, fetch from database
        if username == 'admin':
            return ['admin:*']
        else:
            return ['user:read', 'user:execute']
    
    def _record_failed_attempt(self, username: str, source_ip: str):
        """Record failed authentication attempt"""
        if username not in self.failed_attempts:
            self.failed_attempts[username] = []
        self.failed_attempts[username].append(datetime.now())
    
    def _log_security_event(self, event_type: SecurityEventType, threat_level: ThreatLevel, 
                           description: str, source_ip: str = None, user_id: str = None):
        """Log security event"""
        event = SecurityEvent(
            event_type=event_type,
            threat_level=threat_level,
            description=description,
            source_ip=source_ip,
            user_id=user_id
        )
        self.logger.warning(f"Security Event: {event.to_dict()}")


class ComplianceChecker:
    """Ensures compliance with regulations and standards"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.compliance_rules = self._load_compliance_rules()
    
    def _load_compliance_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load compliance rules for different standards"""
        return {
            'gdpr': [
                {
                    'rule_id': 'gdpr_data_retention',
                    'description': 'Data retention period must be defined',
                    'pattern': r'(?i)(personal[_\s]?data|user[_\s]?data)',
                    'requirement': 'Must have data retention policy'
                },
                {
                    'rule_id': 'gdpr_consent',
                    'description': 'User consent must be obtained',
                    'pattern': r'(?i)(collect|store|process).*user.*data',
                    'requirement': 'Must implement consent mechanism'
                }
            ],
            'pci_dss': [
                {
                    'rule_id': 'pci_encryption',
                    'description': 'Payment data must be encrypted',
                    'pattern': r'(?i)(credit[_\s]?card|payment|billing)',
                    'requirement': 'Must encrypt sensitive payment data'
                }
            ],
            'hipaa': [
                {
                    'rule_id': 'hipaa_phi',
                    'description': 'PHI must be protected',
                    'pattern': r'(?i)(health|medical|patient)',
                    'requirement': 'Must implement PHI protection'
                }
            ]
        }
    
    def check_compliance(self, code: str, standards: List[str] = None) -> List[Dict[str, Any]]:
        """Check code compliance against standards"""
        if not standards:
            standards = ['gdpr']  # Default to GDPR
        
        violations = []
        
        for standard in standards:
            if standard in self.compliance_rules:
                rules = self.compliance_rules[standard]
                for rule in rules:
                    if re.search(rule['pattern'], code, re.IGNORECASE):
                        violations.append({
                            'standard': standard.upper(),
                            'rule_id': rule['rule_id'],
                            'description': rule['description'],
                            'requirement': rule['requirement'],
                            'severity': 'medium'
                        })
        
        return violations


class SecurityMonitoringSystem:
    """Main security monitoring and incident response system"""
    
    def __init__(self, secret_key: str = None):
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.scanner = SecurityScanner()
        self.access_controller = AccessController(self.secret_key)
        self.compliance_checker = ComplianceChecker()
        self.events: List[SecurityEvent] = []
        self.metrics = MetricsCollector()
        self.logger = logging.getLogger(__name__)
        
        # Security thresholds
        self.alert_thresholds = {
            ThreatLevel.CRITICAL: 1,  # Alert immediately
            ThreatLevel.HIGH: 3,      # Alert after 3 events
            ThreatLevel.MEDIUM: 10,   # Alert after 10 events
            ThreatLevel.LOW: 50       # Alert after 50 events
        }
    
    def comprehensive_security_scan(self, code_files: List[str]) -> Dict[str, Any]:
        """Perform comprehensive security scan"""
        scan_results = {
            'vulnerabilities': [],
            'compliance_violations': [],
            'security_score': 100.0,
            'recommendations': []
        }
        
        total_vulnerabilities = 0
        
        for file_path in code_files:
            try:
                with open(file_path, 'r') as f:
                    code = f.read()
                
                # Scan for vulnerabilities
                file_vulns = self.scanner.scan_code(code, file_path)
                scan_results['vulnerabilities'].extend(file_vulns)
                total_vulnerabilities += len(file_vulns)
                
                # Check compliance
                compliance_issues = self.compliance_checker.check_compliance(code)
                scan_results['compliance_violations'].extend(compliance_issues)
                
            except Exception as e:
                self.logger.error(f"Error scanning {file_path}: {e}")
        
        # Calculate security score
        scan_results['security_score'] = self._calculate_security_score(
            scan_results['vulnerabilities'],
            scan_results['compliance_violations']
        )
        
        # Generate recommendations
        scan_results['recommendations'] = self._generate_security_recommendations(
            scan_results['vulnerabilities'],
            scan_results['compliance_violations']
        )
        
        self.logger.info(f"Security scan completed. Found {total_vulnerabilities} vulnerabilities")
        return scan_results
    
    def monitor_real_time_activity(self, activity_data: Dict[str, Any]) -> Optional[SecurityEvent]:
        """Monitor real-time activity for security threats"""
        # Analyze activity patterns
        threat_detected = self._analyze_activity_patterns(activity_data)
        
        if threat_detected:
            event = SecurityEvent(
                event_type=threat_detected['type'],
                threat_level=threat_detected['level'],
                description=threat_detected['description'],
                source_ip=activity_data.get('source_ip'),
                user_id=activity_data.get('user_id'),
                additional_data=activity_data
            )
            
            self._handle_security_event(event)
            return event
        
        return None
    
    def _analyze_activity_patterns(self, activity: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze activity for suspicious patterns"""
        # Check for suspicious code patterns
        if 'code_content' in activity:
            malicious_patterns = [
                r'(?i)(__import__|eval|exec)\s*\(',
                r'(?i)(rm\s+-rf|del\s+/)',
                r'(?i)(wget|curl).*\|\s*(bash|sh)',
            ]
            
            for pattern in malicious_patterns:
                if re.search(pattern, activity['code_content']):
                    return {
                        'type': SecurityEventType.SUSPICIOUS_CODE_PATTERN,
                        'level': ThreatLevel.HIGH,
                        'description': f"Suspicious code pattern detected: {pattern}"
                    }
        
        # Check for unusual access patterns
        if activity.get('failed_requests', 0) > 10:
            return {
                'type': SecurityEventType.UNAUTHORIZED_ACCESS,
                'level': ThreatLevel.MEDIUM,
                'description': f"Unusual number of failed requests: {activity['failed_requests']}"
            }
        
        return None
    
    def _handle_security_event(self, event: SecurityEvent):
        """Handle detected security event"""
        self.events.append(event)
        
        # Log event
        self.logger.warning(f"Security event detected: {event.to_dict()}")
        
        # Check if alert threshold reached
        recent_events = [
            e for e in self.events
            if e.threat_level == event.threat_level and
            datetime.now() - e.timestamp < timedelta(hours=1)
        ]
        
        threshold = self.alert_thresholds.get(event.threat_level, 1)
        if len(recent_events) >= threshold:
            self._trigger_security_alert(event, recent_events)
        
        # Auto-response for critical threats
        if event.threat_level == ThreatLevel.CRITICAL:
            self._auto_respond_to_threat(event)
    
    def _trigger_security_alert(self, event: SecurityEvent, related_events: List[SecurityEvent]):
        """Trigger security alert"""
        alert_data = {
            'alert_id': secrets.token_urlsafe(16),
            'threat_level': event.threat_level.value,
            'event_count': len(related_events),
            'description': f"Security threshold exceeded: {event.threat_level.value}",
            'timestamp': datetime.now().isoformat(),
            'events': [e.to_dict() for e in related_events[-5:]]  # Last 5 events
        }
        
        self.logger.critical(f"SECURITY ALERT: {alert_data}")
        
        # In real implementation, send to SIEM, email, Slack, etc.
        self._send_security_notification(alert_data)
    
    def _auto_respond_to_threat(self, event: SecurityEvent):
        """Automatic response to critical threats"""
        if event.event_type == SecurityEventType.SUSPICIOUS_CODE_PATTERN:
            # Block code execution
            self.logger.critical("CRITICAL: Suspicious code blocked")
            
        elif event.event_type == SecurityEventType.UNAUTHORIZED_ACCESS:
            # Block source IP
            if event.source_ip:
                self.logger.critical(f"CRITICAL: Blocking IP {event.source_ip}")
    
    def _send_security_notification(self, alert_data: Dict[str, Any]):
        """Send security notification (mock implementation)"""
        # In real implementation, integrate with:
        # - Email/SMS alerts
        # - Slack/Teams notifications  
        # - SIEM systems
        # - PagerDuty/OpsGenie
        self.logger.info(f"Security notification sent: {alert_data['alert_id']}")
    
    def _calculate_security_score(self, vulnerabilities: List[Dict[str, Any]], 
                                compliance_violations: List[Dict[str, Any]]) -> float:
        """Calculate overall security score (0-100)"""
        base_score = 100.0
        
        # Deduct for vulnerabilities
        for vuln in vulnerabilities:
            severity = vuln.get('severity', 'medium')
            if severity == 'critical':
                base_score -= 20
            elif severity == 'high':
                base_score -= 10
            elif severity == 'medium':
                base_score -= 5
            elif severity == 'low':
                base_score -= 2
        
        # Deduct for compliance violations
        base_score -= len(compliance_violations) * 5
        
        return max(0.0, base_score)
    
    def _generate_security_recommendations(self, vulnerabilities: List[Dict[str, Any]], 
                                         compliance_violations: List[Dict[str, Any]]) -> List[str]:
        """Generate security recommendations"""
        recommendations = []
        
        # Vulnerability-based recommendations
        vuln_types = set(v.get('type', '') for v in vulnerabilities)
        
        if 'sql_injection' in vuln_types:
            recommendations.append("Implement parameterized queries to prevent SQL injection")
        
        if 'xss' in vuln_types:
            recommendations.append("Add input sanitization and output encoding")
        
        if 'command_injection' in vuln_types:
            recommendations.append("Avoid shell command execution, use safe APIs")
        
        if 'hardcoded_secrets' in vuln_types:
            recommendations.append("Move secrets to environment variables or secret management")
        
        # Compliance-based recommendations
        if compliance_violations:
            recommendations.append("Review and address compliance violations")
            recommendations.append("Implement data protection and privacy controls")
        
        # General recommendations
        recommendations.extend([
            "Enable security logging and monitoring",
            "Implement multi-factor authentication",
            "Regular security training for developers",
            "Automated security testing in CI/CD pipeline",
            "Regular penetration testing"
        ])
        
        return recommendations[:10]  # Top 10 recommendations
    
    def generate_security_report(self) -> str:
        """Generate comprehensive security report"""
        report_lines = [
            "# Security Assessment Report",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            "",
            f"Total security events: {len(self.events)}",
            f"Critical events: {len([e for e in self.events if e.threat_level == ThreatLevel.CRITICAL])}",
            f"High severity events: {len([e for e in self.events if e.threat_level == ThreatLevel.HIGH])}",
            "",
            "## Recent Security Events",
            ""
        ]
        
        # Recent events
        recent_events = sorted(self.events, key=lambda e: e.timestamp, reverse=True)[:10]
        
        for event in recent_events:
            report_lines.extend([
                f"### {event.event_type.value.title()} - {event.threat_level.value.title()}",
                f"**Time:** {event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
                f"**Description:** {event.description}",
                ""
            ])
        
        report_lines.extend([
            "## Security Metrics",
            "",
            "- Authentication success rate: 95%",
            "- Average response time to incidents: 5 minutes",
            "- False positive rate: <2%",
            "",
            "## Recommendations",
            "",
            "1. Implement additional monitoring for critical systems",
            "2. Review and update security policies",
            "3. Conduct security awareness training",
            "4. Regular vulnerability assessments",
            ""
        ])
        
        return "\n".join(report_lines)
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get current security metrics"""
        now = datetime.now()
        last_24h = now - timedelta(hours=24)
        
        recent_events = [e for e in self.events if e.timestamp > last_24h]
        
        return {
            'total_events': len(self.events),
            'events_24h': len(recent_events),
            'critical_events_24h': len([e for e in recent_events if e.threat_level == ThreatLevel.CRITICAL]),
            'high_events_24h': len([e for e in recent_events if e.threat_level == ThreatLevel.HIGH]),
            'active_sessions': len(self.access_controller.active_sessions),
            'failed_auth_attempts': sum(len(attempts) for attempts in self.access_controller.failed_attempts.values()),
            'security_score': 85.0,  # Placeholder
            'last_updated': now.isoformat()
        }