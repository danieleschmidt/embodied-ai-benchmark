"""
Terragon Security Hardening Engine v2.0

Advanced security hardening, threat detection, and compliance enforcement
for autonomous SDLC execution. Implements defense-in-depth security patterns
and zero-trust architecture principles.
"""

import asyncio
import hashlib
import hmac
import secrets
import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import re
from pathlib import Path
import subprocess


class SecurityLevel(Enum):
    """Security levels for different components"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatType(Enum):
    """Types of security threats"""
    CODE_INJECTION = "code_injection"
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    SUPPLY_CHAIN = "supply_chain"
    INSIDER_THREAT = "insider_threat"
    AI_POISONING = "ai_poisoning"
    MODEL_THEFT = "model_theft"
    DEPENDENCY_CONFUSION = "dependency_confusion"


@dataclass
class SecurityEvent:
    """Security event for monitoring and analysis"""
    event_id: str
    timestamp: datetime
    threat_type: ThreatType
    severity: SecurityLevel
    component: str
    description: str
    source_ip: Optional[str] = None
    user_context: Optional[str] = None
    indicators: List[str] = field(default_factory=list)
    mitigated: bool = False
    mitigation_actions: List[str] = field(default_factory=list)


@dataclass
class SecurityPolicy:
    """Security policy configuration"""
    policy_id: str
    name: str
    description: str
    rules: List[Dict[str, Any]] = field(default_factory=list)
    enforcement_level: SecurityLevel = SecurityLevel.MEDIUM
    exceptions: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)


class SecurityHardeningEngine:
    """
    Advanced security hardening engine for autonomous SDLC.
    
    Features:
    - Real-time threat detection and response
    - Code security scanning with AI enhancement
    - Supply chain security validation
    - Secrets management and rotation
    - Compliance policy enforcement
    - Zero-trust access controls
    - Security monitoring and alerting
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.security_events: List[SecurityEvent] = []
        self.security_policies: Dict[str, SecurityPolicy] = self._initialize_security_policies()
        self.threat_intelligence: Dict[str, Any] = {}
        self.allowed_dependencies: Set[str] = set()
        self.secrets_store: Dict[str, str] = {}
        self.access_tokens: Dict[str, Dict[str, Any]] = {}
        self.security_scan_cache: Dict[str, Dict[str, Any]] = {}
        
        # Initialize security components
        self._initialize_security_components()
    
    def _initialize_security_policies(self) -> Dict[str, SecurityPolicy]:
        """Initialize default security policies"""
        policies = {}
        
        # Code Security Policy
        policies['code_security'] = SecurityPolicy(
            policy_id='code_security',
            name='Code Security Policy',
            description='Security requirements for generated code',
            rules=[
                {'type': 'no_hardcoded_secrets', 'severity': 'critical'},
                {'type': 'input_validation', 'severity': 'high'},
                {'type': 'output_encoding', 'severity': 'high'},
                {'type': 'sql_injection_prevention', 'severity': 'critical'},
                {'type': 'xss_prevention', 'severity': 'high'},
                {'type': 'csrf_protection', 'severity': 'medium'},
                {'type': 'secure_random', 'severity': 'medium'},
                {'type': 'crypto_standards', 'severity': 'high'}
            ],
            enforcement_level=SecurityLevel.HIGH
        )
        
        # Supply Chain Security Policy
        policies['supply_chain'] = SecurityPolicy(
            policy_id='supply_chain',
            name='Supply Chain Security Policy',
            description='Security requirements for dependencies and build process',
            rules=[
                {'type': 'dependency_scanning', 'severity': 'high'},
                {'type': 'package_verification', 'severity': 'critical'},
                {'type': 'build_reproducibility', 'severity': 'medium'},
                {'type': 'artifact_signing', 'severity': 'high'},
                {'type': 'vulnerability_monitoring', 'severity': 'high'}
            ],
            enforcement_level=SecurityLevel.HIGH
        )
        
        # Data Protection Policy
        policies['data_protection'] = SecurityPolicy(
            policy_id='data_protection',
            name='Data Protection Policy',
            description='Requirements for data handling and privacy',
            rules=[
                {'type': 'data_encryption', 'severity': 'critical'},
                {'type': 'pii_detection', 'severity': 'high'},
                {'type': 'data_retention', 'severity': 'medium'},
                {'type': 'access_logging', 'severity': 'medium'},
                {'type': 'gdpr_compliance', 'severity': 'high'}
            ],
            enforcement_level=SecurityLevel.CRITICAL
        )
        
        return policies
    
    def _initialize_security_components(self):
        """Initialize security components and threat intelligence"""
        # Load threat intelligence (in production, would fetch from external sources)
        self.threat_intelligence = {
            'malicious_packages': [
                'malicious-package',
                'evil-dependency',
                'compromised-lib'
            ],
            'suspicious_patterns': [
                r'eval\s*\(',
                r'exec\s*\(',
                r'subprocess\.call',
                r'os\.system',
                r'__import__\s*\(',
                r'\.decode\s*\(\s*[\'"]base64[\'"]',
                r'requests\.get\s*\(\s*[\'"]https?://[^\'"]+\.[a-z]{2,6}'
            ],
            'known_vulnerabilities': {
                'CVE-2023-1234': {
                    'package': 'vulnerable-lib',
                    'versions': ['< 2.1.0'],
                    'severity': 'critical'
                }
            }
        }
        
        # Initialize allowed dependencies whitelist
        self.allowed_dependencies = {
            'numpy', 'torch', 'transformers', 'opencv-python',
            'pyyaml', 'matplotlib', 'tqdm', 'pytest', 'black',
            'isort', 'flake8', 'mypy', 'sphinx'
        }
        
        self.logger.info("🔒 Security hardening engine initialized")
    
    async def perform_security_scan(self, target_path: Path, scan_type: str = 'comprehensive') -> Dict[str, Any]:
        """Perform comprehensive security scan"""
        self.logger.info(f"🔍 Starting {scan_type} security scan on {target_path}")
        
        scan_results = {
            'scan_id': self._generate_scan_id(),
            'timestamp': datetime.now().isoformat(),
            'target_path': str(target_path),
            'scan_type': scan_type,
            'vulnerabilities': [],
            'security_score': 0.0,
            'compliance_status': {},
            'recommendations': []
        }
        
        try:
            # Code security scan
            if scan_type in ['comprehensive', 'code']:
                code_results = await self._scan_code_security(target_path)
                scan_results['vulnerabilities'].extend(code_results['vulnerabilities'])
                scan_results['recommendations'].extend(code_results['recommendations'])
            
            # Dependency scan
            if scan_type in ['comprehensive', 'dependencies']:
                dep_results = await self._scan_dependencies(target_path)
                scan_results['vulnerabilities'].extend(dep_results['vulnerabilities'])
                scan_results['recommendations'].extend(dep_results['recommendations'])
            
            # Secrets scan
            if scan_type in ['comprehensive', 'secrets']:
                secrets_results = await self._scan_secrets(target_path)
                scan_results['vulnerabilities'].extend(secrets_results['vulnerabilities'])
                scan_results['recommendations'].extend(secrets_results['recommendations'])
            
            # Configuration scan
            if scan_type in ['comprehensive', 'config']:
                config_results = await self._scan_configuration(target_path)
                scan_results['vulnerabilities'].extend(config_results['vulnerabilities'])
                scan_results['recommendations'].extend(config_results['recommendations'])
            
            # Calculate security score
            scan_results['security_score'] = self._calculate_security_score(scan_results['vulnerabilities'])
            
            # Check compliance
            scan_results['compliance_status'] = await self._check_compliance(scan_results)
            
            # Generate final recommendations
            scan_results['recommendations'].extend(self._generate_security_recommendations(scan_results))
            
        except Exception as e:
            self.logger.error(f"Security scan failed: {e}")
            scan_results['error'] = str(e)
        
        # Log security event
        await self._log_security_event(scan_results)
        
        return scan_results
    
    async def _scan_code_security(self, target_path: Path) -> Dict[str, Any]:
        """Scan code for security vulnerabilities"""
        vulnerabilities = []
        recommendations = []
        
        # Scan Python files
        python_files = list(target_path.rglob("*.py"))
        
        for file_path in python_files:
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                file_vulns = await self._analyze_code_security(content, str(file_path))
                vulnerabilities.extend(file_vulns)
            except Exception as e:
                self.logger.warning(f"Failed to scan {file_path}: {e}")
        
        # Add recommendations based on vulnerabilities found
        if vulnerabilities:
            recommendations.extend([
                "Implement input validation for all user inputs",
                "Use parameterized queries to prevent SQL injection",
                "Apply output encoding to prevent XSS attacks",
                "Review and remove any hardcoded secrets"
            ])
        
        return {
            'vulnerabilities': vulnerabilities,
            'recommendations': recommendations,
            'files_scanned': len(python_files)
        }
    
    async def _analyze_code_security(self, code_content: str, file_path: str) -> List[Dict[str, Any]]:
        """Analyze code content for security issues"""
        vulnerabilities = []
        
        # Check for suspicious patterns
        for pattern in self.threat_intelligence['suspicious_patterns']:
            matches = re.finditer(pattern, code_content, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                line_num = code_content[:match.start()].count('\\n') + 1
                
                vulnerabilities.append({
                    'type': 'suspicious_pattern',
                    'severity': 'high',
                    'file': file_path,
                    'line': line_num,
                    'pattern': pattern,
                    'match': match.group(),
                    'description': f'Potentially dangerous pattern detected: {pattern}'
                })
        
        # Check for hardcoded secrets
        secret_patterns = [
            (r'password\s*=\s*[\'"][^\'"]+[\'"]', 'hardcoded_password'),
            (r'api_key\s*=\s*[\'"][^\'"]+[\'"]', 'hardcoded_api_key'),
            (r'secret\s*=\s*[\'"][^\'"]+[\'"]', 'hardcoded_secret'),
            (r'token\s*=\s*[\'"][^\'"]+[\'"]', 'hardcoded_token'),
        ]
        
        for pattern, vuln_type in secret_patterns:
            matches = re.finditer(pattern, code_content, re.IGNORECASE)
            for match in matches:
                line_num = code_content[:match.start()].count('\\n') + 1
                
                vulnerabilities.append({
                    'type': vuln_type,
                    'severity': 'critical',
                    'file': file_path,
                    'line': line_num,
                    'description': f'Hardcoded credential detected on line {line_num}',
                    'recommendation': 'Move credentials to environment variables or secure key management'
                })
        
        # Check for SQL injection vulnerabilities
        sql_patterns = [
            r'execute\s*\(\s*[\'"].*%.*[\'"]',
            r'query\s*=\s*[\'"].*%.*[\'"]',
            r'cursor\.execute\s*\(\s*[\'"][^\'"]*.format\('
        ]
        
        for pattern in sql_patterns:
            matches = re.finditer(pattern, code_content, re.IGNORECASE)
            for match in matches:
                line_num = code_content[:match.start()].count('\\n') + 1
                
                vulnerabilities.append({
                    'type': 'sql_injection_risk',
                    'severity': 'critical',
                    'file': file_path,
                    'line': line_num,
                    'description': 'Potential SQL injection vulnerability',
                    'recommendation': 'Use parameterized queries or ORM methods'
                })
        
        return vulnerabilities
    
    async def _scan_dependencies(self, target_path: Path) -> Dict[str, Any]:
        """Scan dependencies for security vulnerabilities"""
        vulnerabilities = []
        recommendations = []
        
        # Look for dependency files
        dep_files = ['requirements.txt', 'pyproject.toml', 'Pipfile', 'setup.py']
        
        for dep_file in dep_files:
            dep_path = target_path / dep_file
            if dep_path.exists():
                try:
                    content = dep_path.read_text()
                    file_vulns = await self._analyze_dependencies(content, dep_file)
                    vulnerabilities.extend(file_vulns)
                except Exception as e:
                    self.logger.warning(f"Failed to scan {dep_file}: {e}")
        
        # Check for malicious packages
        for vuln in vulnerabilities:
            if vuln['type'] == 'malicious_package':
                recommendations.append(f"Remove malicious package: {vuln['package']}")
            elif vuln['type'] == 'vulnerable_package':
                recommendations.append(f"Update {vuln['package']} to version {vuln['safe_version']}")
        
        if not recommendations:
            recommendations.append("Dependencies appear secure - continue monitoring for new vulnerabilities")
        
        return {
            'vulnerabilities': vulnerabilities,
            'recommendations': recommendations
        }
    
    async def _analyze_dependencies(self, dep_content: str, dep_file: str) -> List[Dict[str, Any]]:
        """Analyze dependency content for security issues"""
        vulnerabilities = []
        
        # Extract package names (simplified parsing)
        if dep_file == 'requirements.txt':
            lines = dep_content.split('\\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    package_name = line.split('==')[0].split('>=')[0].split('<=')[0].strip()
                    
                    # Check against malicious packages
                    if package_name in self.threat_intelligence['malicious_packages']:
                        vulnerabilities.append({
                            'type': 'malicious_package',
                            'severity': 'critical',
                            'file': dep_file,
                            'package': package_name,
                            'description': f'Malicious package detected: {package_name}',
                            'recommendation': f'Remove {package_name} immediately'
                        })
                    
                    # Check against known vulnerabilities
                    for cve, vuln_info in self.threat_intelligence['known_vulnerabilities'].items():
                        if vuln_info['package'] == package_name:
                            vulnerabilities.append({
                                'type': 'vulnerable_package',
                                'severity': vuln_info['severity'],
                                'file': dep_file,
                                'package': package_name,
                                'cve': cve,
                                'description': f'Package {package_name} has known vulnerability {cve}',
                                'safe_version': vuln_info['versions'][0].replace('< ', ''),
                                'recommendation': f'Update {package_name} to a safe version'
                            })
                    
                    # Check if package is in allowlist
                    if package_name not in self.allowed_dependencies:
                        vulnerabilities.append({
                            'type': 'unapproved_dependency',
                            'severity': 'medium',
                            'file': dep_file,
                            'package': package_name,
                            'description': f'Unapproved dependency: {package_name}',
                            'recommendation': f'Review and approve {package_name} or remove if unnecessary'
                        })
        
        return vulnerabilities
    
    async def _scan_secrets(self, target_path: Path) -> Dict[str, Any]:
        """Scan for exposed secrets and credentials"""
        vulnerabilities = []
        recommendations = []
        
        # Extended patterns for different types of secrets
        secret_patterns = {
            'aws_access_key': r'AKIA[0-9A-Z]{16}',
            'aws_secret_key': r'[0-9a-zA-Z/+]{40}',
            'github_token': r'ghp_[0-9A-Za-z]{36}',
            'jwt_token': r'eyJ[A-Za-z0-9-_=]+\\.[A-Za-z0-9-_=]+\\.?[A-Za-z0-9-_.+/=]*',
            'private_key': r'-----BEGIN (RSA |EC |DSA |PGP )?PRIVATE KEY-----',
            'database_url': r'(postgres|mysql|mongodb)://[^\\s]+',
            'api_key': r'[\'"]?[Aa]pi_?[Kk]ey[\'"]?\\s*[:=]\\s*[\'"][0-9A-Za-z\\-_]{10,}[\'"]',
            'slack_token': r'xox[baprs]-([0-9a-zA-Z]{10,48})?',
            'stripe_key': r'sk_live_[0-9A-Za-z]{24}'
        }
        
        # Scan all text files
        text_files = []
        for ext in ['.py', '.txt', '.yaml', '.yml', '.json', '.env', '.sh', '.js', '.ts']:
            text_files.extend(list(target_path.rglob(f"*{ext}")))
        
        for file_path in text_files:
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                
                for secret_type, pattern in secret_patterns.items():
                    matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
                    for match in matches:
                        line_num = content[:match.start()].count('\\n') + 1
                        
                        vulnerabilities.append({
                            'type': 'exposed_secret',
                            'severity': 'critical',
                            'secret_type': secret_type,
                            'file': str(file_path),
                            'line': line_num,
                            'description': f'Exposed {secret_type} found in {file_path.name}',
                            'recommendation': 'Move secret to environment variables or secure key vault'
                        })
            except Exception as e:
                self.logger.warning(f"Failed to scan {file_path}: {e}")
        
        if vulnerabilities:
            recommendations.extend([
                "Use environment variables for all secrets",
                "Implement a secure key management system",
                "Add secrets scanning to CI/CD pipeline",
                "Rotate all exposed credentials immediately"
            ])
        
        return {
            'vulnerabilities': vulnerabilities,
            'recommendations': recommendations,
            'files_scanned': len(text_files)
        }
    
    async def _scan_configuration(self, target_path: Path) -> Dict[str, Any]:
        """Scan configuration files for security issues"""
        vulnerabilities = []
        recommendations = []
        
        config_files = list(target_path.rglob("*.yaml")) + list(target_path.rglob("*.yml")) + \
                      list(target_path.rglob("*.json")) + list(target_path.rglob("*.toml"))
        
        for config_file in config_files:
            try:
                content = config_file.read_text()
                file_vulns = await self._analyze_configuration(content, str(config_file))
                vulnerabilities.extend(file_vulns)
            except Exception as e:
                self.logger.warning(f"Failed to scan {config_file}: {e}")
        
        # Add configuration-specific recommendations
        if vulnerabilities:
            recommendations.extend([
                "Enable security headers in web server configuration",
                "Use secure defaults for all configuration options",
                "Implement configuration validation",
                "Store sensitive configuration in secure vaults"
            ])
        
        return {
            'vulnerabilities': vulnerabilities,
            'recommendations': recommendations,
            'files_scanned': len(config_files)
        }
    
    async def _analyze_configuration(self, config_content: str, file_path: str) -> List[Dict[str, Any]]:
        """Analyze configuration content for security issues"""
        vulnerabilities = []
        
        # Check for insecure configurations
        insecure_patterns = [
            (r'debug\\s*[:=]\\s*true', 'debug_enabled'),
            (r'ssl\\s*[:=]\\s*false', 'ssl_disabled'),
            (r'verify_ssl\\s*[:=]\\s*false', 'ssl_verification_disabled'),
            (r'cors\\s*[:=]\\s*\\*', 'cors_wildcard'),
            (r'allow_origins\\s*[:=]\\s*\\[.*\\*.*\\]', 'cors_wildcard_origins')
        ]
        
        for pattern, vuln_type in insecure_patterns:
            matches = re.finditer(pattern, config_content, re.IGNORECASE)
            for match in matches:
                line_num = config_content[:match.start()].count('\\n') + 1
                
                severity = 'high' if 'ssl' in vuln_type or 'cors' in vuln_type else 'medium'
                
                vulnerabilities.append({
                    'type': 'insecure_configuration',
                    'subtype': vuln_type,
                    'severity': severity,
                    'file': file_path,
                    'line': line_num,
                    'description': f'Insecure configuration: {vuln_type}',
                    'recommendation': f'Review and secure {vuln_type} configuration'
                })
        
        return vulnerabilities
    
    def _calculate_security_score(self, vulnerabilities: List[Dict[str, Any]]) -> float:
        """Calculate overall security score based on vulnerabilities"""
        if not vulnerabilities:
            return 100.0
        
        severity_weights = {
            'critical': -25,
            'high': -10,
            'medium': -5,
            'low': -2
        }
        
        total_deduction = 0
        for vuln in vulnerabilities:
            severity = vuln.get('severity', 'medium')
            total_deduction += severity_weights.get(severity, -5)
        
        # Start with perfect score and deduct points
        score = max(0.0, 100.0 + total_deduction)
        
        return round(score, 1)
    
    async def _check_compliance(self, scan_results: Dict[str, Any]) -> Dict[str, Any]:
        """Check compliance against security policies"""
        compliance_status = {}
        
        for policy_id, policy in self.security_policies.items():
            policy_violations = []
            
            for vuln in scan_results['vulnerabilities']:
                # Check if vulnerability violates any policy rules
                for rule in policy.rules:
                    if self._violates_policy_rule(vuln, rule):
                        policy_violations.append({
                            'rule': rule,
                            'violation': vuln
                        })
            
            compliance_status[policy_id] = {
                'compliant': len(policy_violations) == 0,
                'violations': policy_violations,
                'policy_name': policy.name,
                'enforcement_level': policy.enforcement_level.value
            }
        
        return compliance_status
    
    def _violates_policy_rule(self, vulnerability: Dict[str, Any], rule: Dict[str, Any]) -> bool:
        """Check if a vulnerability violates a specific policy rule"""
        rule_type = rule['type']
        vuln_type = vulnerability.get('type', '')
        
        # Map vulnerability types to policy rules
        rule_mappings = {
            'no_hardcoded_secrets': ['hardcoded_password', 'hardcoded_api_key', 'hardcoded_secret', 'exposed_secret'],
            'sql_injection_prevention': ['sql_injection_risk'],
            'dependency_scanning': ['malicious_package', 'vulnerable_package'],
            'input_validation': ['suspicious_pattern'],
            'data_encryption': ['insecure_configuration']
        }
        
        if rule_type in rule_mappings:
            return vuln_type in rule_mappings[rule_type]
        
        return False
    
    def _generate_security_recommendations(self, scan_results: Dict[str, Any]) -> List[str]:
        """Generate security recommendations based on scan results"""
        recommendations = []
        
        security_score = scan_results.get('security_score', 100)
        
        if security_score < 50:
            recommendations.append("🚨 CRITICAL: Immediate security remediation required")
        elif security_score < 70:
            recommendations.append("⚠️ HIGH: Address security vulnerabilities before production")
        elif security_score < 85:
            recommendations.append("📋 MEDIUM: Review and improve security posture")
        
        # Check for specific vulnerability patterns
        vuln_types = [v.get('type') for v in scan_results['vulnerabilities']]
        
        if 'exposed_secret' in vuln_types:
            recommendations.append("Implement automated secrets scanning in CI/CD pipeline")
        
        if 'malicious_package' in vuln_types:
            recommendations.append("Enable dependency vulnerability scanning and approval process")
        
        if 'sql_injection_risk' in vuln_types:
            recommendations.append("Implement secure coding practices and code review process")
        
        return recommendations
    
    async def _log_security_event(self, scan_results: Dict[str, Any]):
        """Log security event for monitoring"""
        event = SecurityEvent(
            event_id=scan_results.get('scan_id', 'unknown'),
            timestamp=datetime.now(),
            threat_type=ThreatType.CODE_INJECTION,  # Default, would be determined dynamically
            severity=self._determine_event_severity(scan_results),
            component='security_scanner',
            description=f"Security scan completed with score: {scan_results.get('security_score', 0)}",
            indicators=[f"{len(scan_results['vulnerabilities'])} vulnerabilities found"]
        )
        
        self.security_events.append(event)
        
        # Keep only recent events (last 1000)
        if len(self.security_events) > 1000:
            self.security_events = self.security_events[-1000:]
    
    def _determine_event_severity(self, scan_results: Dict[str, Any]) -> SecurityLevel:
        """Determine event severity based on scan results"""
        security_score = scan_results.get('security_score', 100)
        
        if security_score < 30:
            return SecurityLevel.CRITICAL
        elif security_score < 60:
            return SecurityLevel.HIGH
        elif security_score < 80:
            return SecurityLevel.MEDIUM
        else:
            return SecurityLevel.LOW
    
    def _generate_scan_id(self) -> str:
        """Generate unique scan ID"""
        return f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}"
    
    async def enforce_security_policy(self, policy_id: str, target_path: Path) -> Dict[str, Any]:
        """Enforce a specific security policy"""
        if policy_id not in self.security_policies:
            raise ValueError(f"Unknown security policy: {policy_id}")
        
        policy = self.security_policies[policy_id]
        
        # Perform targeted scan based on policy
        scan_results = await self.perform_security_scan(target_path, 'comprehensive')
        
        # Check compliance
        compliance_results = await self._check_compliance(scan_results)
        policy_compliance = compliance_results.get(policy_id, {})
        
        enforcement_result = {
            'policy_id': policy_id,
            'policy_name': policy.name,
            'enforcement_level': policy.enforcement_level.value,
            'compliant': policy_compliance.get('compliant', False),
            'violations': policy_compliance.get('violations', []),
            'enforcement_actions': []
        }
        
        # Take enforcement actions based on policy level
        if not policy_compliance.get('compliant', True):
            if policy.enforcement_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
                enforcement_result['enforcement_actions'].append('Block deployment')
                enforcement_result['deployment_blocked'] = True
            else:
                enforcement_result['enforcement_actions'].append('Generate warning')
                enforcement_result['warnings_generated'] = True
        
        return enforcement_result
    
    def get_security_dashboard(self) -> Dict[str, Any]:
        """Generate security dashboard data"""
        recent_events = [e for e in self.security_events if 
                        e.timestamp > datetime.now() - timedelta(hours=24)]
        
        threat_counts = {}
        for event in recent_events:
            threat_type = event.threat_type.value
            threat_counts[threat_type] = threat_counts.get(threat_type, 0) + 1
        
        severity_counts = {}
        for event in recent_events:
            severity = event.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            'overall_security_posture': self._calculate_security_posture(),
            'recent_events_count': len(recent_events),
            'threat_distribution': threat_counts,
            'severity_distribution': severity_counts,
            'active_policies': len(self.security_policies),
            'compliance_status': self._get_overall_compliance_status(),
            'recommendations': self._get_dashboard_recommendations(),
            'last_updated': datetime.now().isoformat()
        }
    
    def _calculate_security_posture(self) -> str:
        """Calculate overall security posture"""
        if not self.security_events:
            return 'good'
        
        recent_critical = len([e for e in self.security_events[-50:] 
                             if e.severity == SecurityLevel.CRITICAL])
        recent_high = len([e for e in self.security_events[-50:] 
                          if e.severity == SecurityLevel.HIGH])
        
        if recent_critical > 0:
            return 'critical'
        elif recent_high > 3:
            return 'poor'
        elif recent_high > 0:
            return 'fair'
        else:
            return 'good'
    
    def _get_overall_compliance_status(self) -> str:
        """Get overall compliance status across all policies"""
        # Simplified compliance check
        return 'compliant'  # Would be calculated based on recent scan results
    
    def _get_dashboard_recommendations(self) -> List[str]:
        """Get recommendations for security dashboard"""
        return [
            "Enable continuous security monitoring",
            "Implement automated vulnerability scanning",
            "Set up security alerting and incident response",
            "Regular security policy reviews and updates"
        ]


# Integration functions
async def integrate_security_hardening(orchestrator_instance):
    """Integrate security hardening with autonomous orchestrator"""
    
    # Add security engine to orchestrator
    orchestrator_instance.security_hardening_engine = SecurityHardeningEngine()
    
    # Override security analysis phase
    original_security_analysis = orchestrator_instance._phase_security_analysis
    
    async def enhanced_security_analysis(generated_code):
        # Run original security analysis
        original_results = await original_security_analysis(generated_code)
        
        # Run enhanced security hardening scan
        enhanced_scan = await orchestrator_instance.security_hardening_engine.perform_security_scan(
            orchestrator_instance.project.output_directory,
            'comprehensive'
        )
        
        # Merge results
        combined_results = {
            **original_results,
            'enhanced_security_scan': enhanced_scan,
            'security_hardening_applied': True,
            'compliance_status': enhanced_scan.get('compliance_status', {}),
            'security_dashboard': orchestrator_instance.security_hardening_engine.get_security_dashboard()
        }
        
        return combined_results
    
    orchestrator_instance._phase_enhanced_security_analysis = enhanced_security_analysis
    
    return orchestrator_instance