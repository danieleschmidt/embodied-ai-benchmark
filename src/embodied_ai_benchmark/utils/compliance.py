"""Compliance and regulatory utilities for the embodied AI benchmark."""

import json
import logging
import hashlib
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class ComplianceLevel(Enum):
    """Compliance levels for different regulatory frameworks."""
    BASIC = "basic"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOX = "sox"
    ISO27001 = "iso27001"
    NIST = "nist"


class DataClassification(Enum):
    """Data classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


@dataclass
class AuditLogEntry:
    """Audit log entry for compliance tracking."""
    timestamp: datetime
    user_id: str
    action: str
    resource: str
    outcome: str
    details: Dict[str, Any] = field(default_factory=dict)
    risk_level: str = "low"
    compliance_tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "action": self.action,
            "resource": self.resource,
            "outcome": self.outcome,
            "details": self.details,
            "risk_level": self.risk_level,
            "compliance_tags": self.compliance_tags
        }


@dataclass
class DataRetentionPolicy:
    """Data retention policy configuration."""
    data_type: str
    retention_days: int
    archive_after_days: Optional[int] = None
    encryption_required: bool = True
    anonymization_required: bool = False
    deletion_method: str = "secure_wipe"
    compliance_requirements: List[str] = field(default_factory=list)


class ComplianceManager:
    """Manages compliance requirements and audit logging."""
    
    def __init__(self, 
                 compliance_level: ComplianceLevel = ComplianceLevel.BASIC,
                 audit_log_path: Optional[str] = None):
        """Initialize compliance manager.
        
        Args:
            compliance_level: Required compliance level
            audit_log_path: Path to store audit logs
        """
        self.compliance_level = compliance_level
        self.audit_log_path = Path(audit_log_path) if audit_log_path else Path("audit.log")
        
        self.data_retention_policies: Dict[str, DataRetentionPolicy] = {}
        self.consent_records: Dict[str, Dict[str, Any]] = {}
        self.data_processing_records: Dict[str, Dict[str, Any]] = {}
        
        self._setup_default_policies()
        logger.info(f"Compliance manager initialized with level: {compliance_level.value}")
    
    def _setup_default_policies(self):
        """Set up default retention policies based on compliance level."""
        if self.compliance_level == ComplianceLevel.GDPR:
            self.data_retention_policies.update({
                "personal_data": DataRetentionPolicy(
                    data_type="personal_data",
                    retention_days=2555,  # 7 years
                    archive_after_days=1095,  # 3 years
                    encryption_required=True,
                    anonymization_required=True,
                    compliance_requirements=["GDPR Article 5", "GDPR Article 17"]
                ),
                "benchmark_results": DataRetentionPolicy(
                    data_type="benchmark_results",
                    retention_days=1825,  # 5 years
                    encryption_required=True,
                    compliance_requirements=["Data retention for research"]
                )
            })
        elif self.compliance_level == ComplianceLevel.HIPAA:
            self.data_retention_policies.update({
                "health_data": DataRetentionPolicy(
                    data_type="health_data",
                    retention_days=2190,  # 6 years
                    encryption_required=True,
                    anonymization_required=True,
                    compliance_requirements=["HIPAA 164.530(j)(2)"]
                )
            })
        else:
            # Basic compliance
            self.data_retention_policies.update({
                "experiment_data": DataRetentionPolicy(
                    data_type="experiment_data",
                    retention_days=365,  # 1 year
                    encryption_required=False
                )
            })
    
    def log_audit_event(self, 
                       user_id: str,
                       action: str,
                       resource: str,
                       outcome: str,
                       details: Optional[Dict[str, Any]] = None,
                       risk_level: str = "low") -> str:
        """Log an audit event.
        
        Args:
            user_id: ID of user performing action
            action: Action being performed
            resource: Resource being acted upon
            outcome: Outcome of the action
            details: Additional details
            risk_level: Risk level (low, medium, high, critical)
            
        Returns:
            Audit event ID
        """
        audit_id = str(uuid.uuid4())
        
        # Determine compliance tags based on action and compliance level
        compliance_tags = self._get_compliance_tags(action, resource)
        
        entry = AuditLogEntry(
            timestamp=datetime.utcnow(),
            user_id=user_id,
            action=action,
            resource=resource,
            outcome=outcome,
            details=details or {},
            risk_level=risk_level,
            compliance_tags=compliance_tags
        )
        
        # Write to audit log
        try:
            with open(self.audit_log_path, 'a', encoding='utf-8') as f:
                log_line = json.dumps(entry.to_dict()) + '\n'
                f.write(log_line)
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
        
        # Log high-risk events
        if risk_level in ["high", "critical"]:
            logger.warning(f"High-risk audit event: {action} on {resource} by {user_id}")
        
        return audit_id
    
    def _get_compliance_tags(self, action: str, resource: str) -> List[str]:
        """Get compliance tags for an action."""
        tags = []
        
        if self.compliance_level == ComplianceLevel.GDPR:
            if "personal" in resource.lower() or "user" in resource.lower():
                tags.append("GDPR")
            if action.lower() in ["delete", "anonymize", "export"]:
                tags.append("GDPR_DATA_RIGHTS")
        
        if self.compliance_level == ComplianceLevel.HIPAA:
            if "health" in resource.lower() or "medical" in resource.lower():
                tags.append("HIPAA")
        
        if action.lower() in ["access", "view", "download"]:
            tags.append("DATA_ACCESS")
        elif action.lower() in ["create", "update", "modify"]:
            tags.append("DATA_MODIFICATION")
        elif action.lower() in ["delete", "destroy"]:
            tags.append("DATA_DELETION")
        
        return tags
    
    def record_consent(self, 
                      user_id: str,
                      purpose: str,
                      data_types: List[str],
                      consent_given: bool,
                      expiry_date: Optional[datetime] = None) -> str:
        """Record user consent for data processing.
        
        Args:
            user_id: User identifier
            purpose: Purpose of data processing
            data_types: Types of data being processed
            consent_given: Whether consent was given
            expiry_date: When consent expires
            
        Returns:
            Consent record ID
        """
        consent_id = str(uuid.uuid4())
        
        consent_record = {
            "consent_id": consent_id,
            "user_id": user_id,
            "purpose": purpose,
            "data_types": data_types,
            "consent_given": consent_given,
            "timestamp": datetime.utcnow().isoformat(),
            "expiry_date": expiry_date.isoformat() if expiry_date else None,
            "compliance_level": self.compliance_level.value
        }
        
        self.consent_records[consent_id] = consent_record
        
        # Log audit event
        self.log_audit_event(
            user_id=user_id,
            action="record_consent",
            resource=f"consent/{consent_id}",
            outcome="success" if consent_given else "denied",
            details={"purpose": purpose, "data_types": data_types},
            risk_level="medium"
        )
        
        return consent_id
    
    def check_consent_valid(self, user_id: str, purpose: str) -> bool:
        """Check if user has valid consent for a purpose.
        
        Args:
            user_id: User identifier
            purpose: Purpose to check consent for
            
        Returns:
            True if consent is valid
        """
        current_time = datetime.utcnow()
        
        for consent_record in self.consent_records.values():
            if (consent_record["user_id"] == user_id and 
                consent_record["purpose"] == purpose and
                consent_record["consent_given"]):
                
                # Check expiry
                if consent_record["expiry_date"]:
                    expiry = datetime.fromisoformat(consent_record["expiry_date"])
                    if current_time > expiry:
                        continue
                
                return True
        
        return False
    
    def classify_data(self, data: Dict[str, Any]) -> DataClassification:
        """Classify data based on content and compliance requirements.
        
        Args:
            data: Data to classify
            
        Returns:
            Data classification level
        """
        # Simple classification logic - can be extended
        sensitive_keys = [
            "password", "ssn", "social_security", "credit_card", "medical",
            "health", "biometric", "genetic", "location", "ip_address"
        ]
        
        pii_keys = [
            "name", "email", "phone", "address", "user_id", "username"
        ]
        
        data_str = json.dumps(data).lower()
        
        # Check for restricted data
        if any(key in data_str for key in sensitive_keys):
            return DataClassification.RESTRICTED
        
        # Check for confidential data (PII)
        if any(key in data_str for key in pii_keys):
            return DataClassification.CONFIDENTIAL
        
        # Check for internal data patterns
        if any(key in data_str for key in ["internal", "proprietary", "benchmark"]):
            return DataClassification.INTERNAL
        
        return DataClassification.PUBLIC
    
    def should_encrypt_data(self, data_classification: DataClassification) -> bool:
        """Determine if data should be encrypted based on classification.
        
        Args:
            data_classification: Data classification level
            
        Returns:
            True if encryption is required
        """
        if self.compliance_level in [ComplianceLevel.GDPR, ComplianceLevel.HIPAA]:
            return data_classification in [DataClassification.CONFIDENTIAL, DataClassification.RESTRICTED]
        elif self.compliance_level == ComplianceLevel.ISO27001:
            return data_classification != DataClassification.PUBLIC
        else:
            return data_classification == DataClassification.RESTRICTED
    
    def check_data_retention(self, data_type: str, creation_date: datetime) -> Dict[str, Any]:
        """Check data retention status.
        
        Args:
            data_type: Type of data to check
            creation_date: When data was created
            
        Returns:
            Retention status information
        """
        policy = self.data_retention_policies.get(data_type)
        if not policy:
            return {"status": "no_policy", "action_required": False}
        
        current_time = datetime.utcnow()
        age_days = (current_time - creation_date).days
        
        status = {
            "policy": policy,
            "age_days": age_days,
            "retention_days": policy.retention_days,
            "status": "active",
            "action_required": False,
            "recommended_action": None
        }
        
        if age_days >= policy.retention_days:
            status.update({
                "status": "expired",
                "action_required": True,
                "recommended_action": "delete"
            })
        elif policy.archive_after_days and age_days >= policy.archive_after_days:
            status.update({
                "status": "archive_due",
                "action_required": True,
                "recommended_action": "archive"
            })
        elif policy.anonymization_required and age_days >= (policy.retention_days * 0.8):
            status.update({
                "status": "anonymization_due",
                "action_required": True,
                "recommended_action": "anonymize"
            })
        
        return status
    
    def generate_compliance_report(self, 
                                 start_date: datetime,
                                 end_date: datetime) -> Dict[str, Any]:
        """Generate compliance report for a date range.
        
        Args:
            start_date: Report start date
            end_date: Report end date
            
        Returns:
            Compliance report data
        """
        report = {
            "report_id": str(uuid.uuid4()),
            "generated_at": datetime.utcnow().isoformat(),
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "compliance_level": self.compliance_level.value,
            "audit_events": [],
            "consent_records": [],
            "data_retention_status": [],
            "risk_summary": {
                "low": 0, "medium": 0, "high": 0, "critical": 0
            }
        }
        
        # Parse audit log for the period
        try:
            with open(self.audit_log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        entry_time = datetime.fromisoformat(entry["timestamp"])
                        
                        if start_date <= entry_time <= end_date:
                            report["audit_events"].append(entry)
                            report["risk_summary"][entry["risk_level"]] += 1
                    except:
                        continue
        except FileNotFoundError:
            logger.warning("Audit log file not found")
        
        # Include consent records from the period
        for consent_record in self.consent_records.values():
            consent_time = datetime.fromisoformat(consent_record["timestamp"])
            if start_date <= consent_time <= end_date:
                report["consent_records"].append(consent_record)
        
        # Check data retention compliance
        for data_type, policy in self.data_retention_policies.items():
            report["data_retention_status"].append({
                "data_type": data_type,
                "policy": {
                    "retention_days": policy.retention_days,
                    "encryption_required": policy.encryption_required,
                    "compliance_requirements": policy.compliance_requirements
                }
            })
        
        return report
    
    def validate_data_processing(self, 
                               user_id: str,
                               data: Dict[str, Any],
                               purpose: str) -> Tuple[bool, str]:
        """Validate if data processing is compliant.
        
        Args:
            user_id: User performing the processing
            data: Data to be processed
            purpose: Purpose of processing
            
        Returns:
            Tuple of (is_valid, reason)
        """
        # Check consent if required
        if self.compliance_level in [ComplianceLevel.GDPR, ComplianceLevel.HIPAA]:
            data_classification = self.classify_data(data)
            
            if data_classification in [DataClassification.CONFIDENTIAL, DataClassification.RESTRICTED]:
                if not self.check_consent_valid(user_id, purpose):
                    return False, "No valid consent for data processing"
        
        # Check data classification and encryption requirements
        data_classification = self.classify_data(data)
        if self.should_encrypt_data(data_classification):
            # In a real implementation, you would check if data is encrypted
            pass
        
        # Log the validation attempt
        self.log_audit_event(
            user_id=user_id,
            action="validate_processing",
            resource=f"data/{data_classification.value}",
            outcome="approved",
            details={"purpose": purpose, "classification": data_classification.value}
        )
        
        return True, "Data processing approved"


# Global compliance manager instance
global_compliance_manager: Optional[ComplianceManager] = None


def init_compliance(compliance_level: ComplianceLevel = ComplianceLevel.BASIC,
                   audit_log_path: Optional[str] = None) -> ComplianceManager:
    """Initialize global compliance manager.
    
    Args:
        compliance_level: Required compliance level
        audit_log_path: Path for audit logs
        
    Returns:
        Initialized compliance manager
    """
    global global_compliance_manager
    global_compliance_manager = ComplianceManager(compliance_level, audit_log_path)
    return global_compliance_manager


def get_compliance_manager() -> Optional[ComplianceManager]:
    """Get global compliance manager instance."""
    return global_compliance_manager


def audit_log(user_id: str, action: str, resource: str, outcome: str, **kwargs) -> str:
    """Convenience function for audit logging.
    
    Args:
        user_id: User ID
        action: Action performed
        resource: Resource acted upon
        outcome: Outcome of action
        **kwargs: Additional details
        
    Returns:
        Audit event ID
    """
    if global_compliance_manager:
        return global_compliance_manager.log_audit_event(
            user_id, action, resource, outcome, kwargs
        )
    else:
        logger.warning("Compliance manager not initialized")
        return ""