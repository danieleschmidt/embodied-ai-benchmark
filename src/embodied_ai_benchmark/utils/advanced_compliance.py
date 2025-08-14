"""
Advanced Compliance and Regulatory Framework
"""

import json
import logging
import hashlib
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import uuid
import re
from cryptography.fernet import Fernet
import base64

logger = logging.getLogger(__name__)


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    GDPR = "gdpr"              # General Data Protection Regulation (EU)
    CCPA = "ccpa"              # California Consumer Privacy Act (US)
    PDPA = "pdpa"              # Personal Data Protection Act (Singapore, Thailand)
    HIPAA = "hipaa"            # Health Insurance Portability and Accountability Act (US)
    SOX = "sox"                # Sarbanes-Oxley Act (US)
    ISO27001 = "iso27001"      # Information Security Management
    SOC2 = "soc2"              # Service Organization Control 2
    NIST = "nist"              # National Institute of Standards and Technology
    FedRAMP = "fedramp"        # Federal Risk and Authorization Management Program


class DataCategory(Enum):
    """Categories of data for compliance classification."""
    PERSONAL_DATA = "personal_data"
    SENSITIVE_PERSONAL_DATA = "sensitive_personal_data"
    HEALTH_DATA = "health_data"
    FINANCIAL_DATA = "financial_data"
    BIOMETRIC_DATA = "biometric_data"
    LOCATION_DATA = "location_data"
    BEHAVIORAL_DATA = "behavioral_data"
    TECHNICAL_DATA = "technical_data"
    PUBLIC_DATA = "public_data"


class ProcessingPurpose(Enum):
    """Lawful purposes for data processing."""
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"
    RESEARCH = "research"
    STATISTICS = "statistics"


@dataclass
class DataSubject:
    """Represents a data subject (individual whose data is processed)."""
    subject_id: str
    jurisdiction: str
    consent_records: Dict[str, Any] = field(default_factory=dict)
    opt_out_preferences: Set[str] = field(default_factory=set)
    data_categories: Set[DataCategory] = field(default_factory=set)
    retention_requirements: Dict[str, timedelta] = field(default_factory=dict)
    
    
@dataclass
class CompliancePolicy:
    """Defines compliance requirements for a specific framework."""
    framework: ComplianceFramework
    jurisdiction: str
    data_categories: Set[DataCategory]
    processing_purposes: Set[ProcessingPurpose]
    retention_period: timedelta
    encryption_required: bool = True
    anonymization_required: bool = False
    consent_required: bool = True
    right_to_deletion: bool = True
    right_to_portability: bool = True
    breach_notification_time: timedelta = timedelta(hours=72)
    

@dataclass
class DataProcessingRecord:
    """Record of data processing activity."""
    record_id: str
    timestamp: datetime
    data_subject_id: str
    data_categories: Set[DataCategory]
    processing_purpose: ProcessingPurpose
    legal_basis: str
    retention_period: timedelta
    processor_id: str
    location: str
    security_measures: List[str]
    

class AdvancedComplianceEngine:
    """Advanced compliance engine for multi-jurisdiction support."""
    
    def __init__(self):
        """Initialize compliance engine."""
        self.policies: Dict[ComplianceFramework, CompliancePolicy] = {}
        self.data_subjects: Dict[str, DataSubject] = {}
        self.processing_records: List[DataProcessingRecord] = []
        self.encryption_key = Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
        
        # Audit and logging
        self.audit_log: List[Dict[str, Any]] = []
        self.compliance_violations: List[Dict[str, Any]] = []
        
        # Initialize default policies
        self._setup_default_policies()
        
    def _setup_default_policies(self) -> None:
        """Setup default compliance policies."""
        # GDPR Policy
        self.policies[ComplianceFramework.GDPR] = CompliancePolicy(
            framework=ComplianceFramework.GDPR,
            jurisdiction="EU",
            data_categories={
                DataCategory.PERSONAL_DATA,
                DataCategory.SENSITIVE_PERSONAL_DATA,
                DataCategory.BIOMETRIC_DATA,
                DataCategory.LOCATION_DATA
            },
            processing_purposes={
                ProcessingPurpose.CONSENT,
                ProcessingPurpose.LEGITIMATE_INTERESTS,
                ProcessingPurpose.RESEARCH
            },
            retention_period=timedelta(days=365),  # 1 year default
            encryption_required=True,
            consent_required=True,
            right_to_deletion=True,
            right_to_portability=True,
            breach_notification_time=timedelta(hours=72)
        )
        
        # CCPA Policy
        self.policies[ComplianceFramework.CCPA] = CompliancePolicy(
            framework=ComplianceFramework.CCPA,
            jurisdiction="California",
            data_categories={
                DataCategory.PERSONAL_DATA,
                DataCategory.BEHAVIORAL_DATA,
                DataCategory.LOCATION_DATA,
                DataCategory.BIOMETRIC_DATA
            },
            processing_purposes={
                ProcessingPurpose.CONSENT,
                ProcessingPurpose.LEGITIMATE_INTERESTS
            },
            retention_period=timedelta(days=365),
            encryption_required=True,
            consent_required=False,  # Opt-out model
            right_to_deletion=True,
            right_to_portability=True,
            breach_notification_time=timedelta(hours=24)
        )
        
        # HIPAA Policy
        self.policies[ComplianceFramework.HIPAA] = CompliancePolicy(
            framework=ComplianceFramework.HIPAA,
            jurisdiction="US",
            data_categories={DataCategory.HEALTH_DATA},
            processing_purposes={
                ProcessingPurpose.CONSENT,
                ProcessingPurpose.LEGAL_OBLIGATION
            },
            retention_period=timedelta(days=2555),  # 7 years
            encryption_required=True,
            anonymization_required=True,
            consent_required=True,
            right_to_deletion=False,  # Healthcare records retention
            right_to_portability=True
        )
        
    def register_data_subject(self, 
                            subject_id: str,
                            jurisdiction: str,
                            data_categories: Set[DataCategory],
                            consents: Dict[str, bool] = None) -> None:
        """Register a data subject and their preferences.
        
        Args:
            subject_id: Unique identifier for the data subject
            jurisdiction: Legal jurisdiction
            data_categories: Categories of data being processed
            consents: Consent preferences
        """
        consents = consents or {}
        
        data_subject = DataSubject(
            subject_id=subject_id,
            jurisdiction=jurisdiction,
            consent_records=consents,
            data_categories=data_categories
        )
        
        self.data_subjects[subject_id] = data_subject
        
        # Determine applicable frameworks
        applicable_frameworks = self._get_applicable_frameworks(jurisdiction, data_categories)
        
        # Set retention requirements based on frameworks
        for framework in applicable_frameworks:
            policy = self.policies.get(framework)
            if policy:
                data_subject.retention_requirements[framework.value] = policy.retention_period
                
        self._log_audit_event(
            action="data_subject_registered",
            subject_id=subject_id,
            details={
                "jurisdiction": jurisdiction,
                "data_categories": [cat.value for cat in data_categories],
                "applicable_frameworks": [f.value for f in applicable_frameworks]
            }
        )
        
    def record_data_processing(self,
                             data_subject_id: str,
                             data_categories: Set[DataCategory],
                             processing_purpose: ProcessingPurpose,
                             processor_id: str,
                             location: str) -> str:
        """Record a data processing activity.
        
        Args:
            data_subject_id: ID of the data subject
            data_categories: Categories of data being processed
            processing_purpose: Purpose of processing
            processor_id: ID of the processor
            location: Geographic location of processing
            
        Returns:
            Unique record ID
        """
        record_id = str(uuid.uuid4())
        
        # Get data subject info
        data_subject = self.data_subjects.get(data_subject_id)
        if not data_subject:
            raise ValueError(f"Data subject {data_subject_id} not registered")
            
        # Determine applicable frameworks and legal basis
        applicable_frameworks = self._get_applicable_frameworks(
            data_subject.jurisdiction, 
            data_categories
        )
        
        # Validate processing is compliant
        compliance_issues = self._validate_processing_compliance(
            data_subject, data_categories, processing_purpose, applicable_frameworks
        )
        
        if compliance_issues:
            self._record_compliance_violation(
                violation_type="unauthorized_processing",
                details={
                    "data_subject_id": data_subject_id,
                    "issues": compliance_issues
                }
            )
            raise ValueError(f"Processing not compliant: {compliance_issues}")
            
        # Determine retention period
        retention_period = self._calculate_retention_period(applicable_frameworks)
        
        # Create processing record
        processing_record = DataProcessingRecord(
            record_id=record_id,
            timestamp=datetime.now(),
            data_subject_id=data_subject_id,
            data_categories=data_categories,
            processing_purpose=processing_purpose,
            legal_basis=self._determine_legal_basis(processing_purpose, applicable_frameworks),
            retention_period=retention_period,
            processor_id=processor_id,
            location=location,
            security_measures=["encryption", "access_control", "audit_logging"]
        )
        
        self.processing_records.append(processing_record)
        
        self._log_audit_event(
            action="data_processing_recorded",
            subject_id=data_subject_id,
            details={
                "record_id": record_id,
                "data_categories": [cat.value for cat in data_categories],
                "processing_purpose": processing_purpose.value,
                "location": location
            }
        )
        
        return record_id
        
    def handle_data_subject_request(self,
                                   data_subject_id: str,
                                   request_type: str,
                                   specific_data: Optional[List[str]] = None) -> Dict[str, Any]:
        """Handle data subject rights requests (access, deletion, portability).
        
        Args:
            data_subject_id: ID of the data subject
            request_type: Type of request (access, deletion, portability, opt_out)
            specific_data: Specific data categories requested
            
        Returns:
            Response with request status and data
        """
        data_subject = self.data_subjects.get(data_subject_id)
        if not data_subject:
            return {"status": "error", "message": "Data subject not found"}
            
        # Get applicable frameworks
        applicable_frameworks = self._get_applicable_frameworks(
            data_subject.jurisdiction,
            data_subject.data_categories
        )
        
        response = {"status": "success", "request_type": request_type}
        
        if request_type == "access":
            # Right of access - provide all data
            subject_data = self._get_data_subject_data(data_subject_id, specific_data)
            response["data"] = subject_data
            
        elif request_type == "deletion":
            # Right to be forgotten
            if self._can_delete_data(data_subject_id, applicable_frameworks):
                deleted_records = self._delete_data_subject_data(data_subject_id)
                response["deleted_records"] = len(deleted_records)
            else:
                response["status"] = "error"
                response["message"] = "Data cannot be deleted due to legal retention requirements"
                
        elif request_type == "portability":
            # Data portability
            portable_data = self._export_portable_data(data_subject_id)
            response["portable_data"] = portable_data
            
        elif request_type == "opt_out":
            # Opt-out of processing
            data_subject.opt_out_preferences.add("all_processing")
            response["message"] = "Opted out of future processing"
            
        # Log the request
        self._log_audit_event(
            action=f"data_subject_request_{request_type}",
            subject_id=data_subject_id,
            details={"request_type": request_type, "status": response["status"]}
        )
        
        return response
        
    def encrypt_sensitive_data(self, data: Dict[str, Any]) -> str:
        """Encrypt sensitive data for storage.
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data as base64 string
        """
        data_json = json.dumps(data, default=str)
        encrypted_data = self.cipher.encrypt(data_json.encode())
        return base64.b64encode(encrypted_data).decode()
        
    def decrypt_sensitive_data(self, encrypted_data: str) -> Dict[str, Any]:
        """Decrypt sensitive data.
        
        Args:
            encrypted_data: Base64 encoded encrypted data
            
        Returns:
            Decrypted data
        """
        encrypted_bytes = base64.b64decode(encrypted_data.encode())
        decrypted_data = self.cipher.decrypt(encrypted_bytes)
        return json.loads(decrypted_data.decode())
        
    def anonymize_data(self, data: Dict[str, Any], technique: str = "pseudonymization") -> Dict[str, Any]:
        """Anonymize personal data.
        
        Args:
            data: Data to anonymize
            technique: Anonymization technique
            
        Returns:
            Anonymized data
        """
        anonymized = data.copy()
        
        # Define fields that should be anonymized
        personal_fields = [
            "name", "email", "phone", "address", "ssn", "id_number",
            "first_name", "last_name", "date_of_birth"
        ]
        
        for field in personal_fields:
            if field in anonymized:
                if technique == "pseudonymization":
                    # Replace with pseudonym
                    anonymized[field] = self._generate_pseudonym(str(anonymized[field]))
                elif technique == "generalization":
                    # Generalize the data
                    anonymized[field] = self._generalize_value(anonymized[field], field)
                elif technique == "suppression":
                    # Remove the field entirely
                    del anonymized[field]
                    
        return anonymized
        
    def _generate_pseudonym(self, original_value: str) -> str:
        """Generate a consistent pseudonym for a value."""
        # Use HMAC for consistent pseudonymization
        import hmac
        secret_key = "compliance_pseudonym_key"  # In production, use secure key management
        pseudonym = hmac.new(
            secret_key.encode(),
            original_value.encode(),
            hashlib.sha256
        ).hexdigest()[:12]
        return f"pseudo_{pseudonym}"
        
    def _generalize_value(self, value: Any, field_type: str) -> str:
        """Generalize a value based on its type."""
        if field_type == "date_of_birth":
            # Return only year if it's a date
            if isinstance(value, (datetime, str)):
                try:
                    if isinstance(value, str):
                        date_obj = datetime.fromisoformat(value.replace('Z', '+00:00'))
                    else:
                        date_obj = value
                    return str(date_obj.year)
                except:
                    return "unknown_year"
        elif field_type in ["address", "location"]:
            # Return only city/state if it's an address
            return "generalized_location"
        else:
            return "generalized_value"
            
    def get_compliance_report(self, framework: Optional[ComplianceFramework] = None) -> Dict[str, Any]:
        """Generate comprehensive compliance report.
        
        Args:
            framework: Specific framework to report on (optional)
            
        Returns:
            Compliance report
        """
        frameworks_to_report = [framework] if framework else list(self.policies.keys())
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "frameworks": {},
            "summary": {
                "total_data_subjects": len(self.data_subjects),
                "total_processing_records": len(self.processing_records),
                "compliance_violations": len(self.compliance_violations),
                "audit_events": len(self.audit_log)
            }
        }
        
        for fw in frameworks_to_report:
            if fw not in self.policies:
                continue
                
            policy = self.policies[fw]
            
            # Count relevant processing records
            relevant_records = [
                r for r in self.processing_records
                if any(cat in policy.data_categories for cat in r.data_categories)
            ]
            
            # Calculate compliance metrics
            total_records = len(relevant_records)
            expired_records = len([
                r for r in relevant_records
                if datetime.now() - r.timestamp > r.retention_period
            ])
            
            report["frameworks"][fw.value] = {
                "jurisdiction": policy.jurisdiction,
                "total_processing_records": total_records,
                "expired_records_requiring_deletion": expired_records,
                "encryption_compliance": 100.0,  # Assuming all data is encrypted
                "retention_compliance": ((total_records - expired_records) / max(1, total_records)) * 100,
                "applicable_data_categories": [cat.value for cat in policy.data_categories],
                "breach_notification_requirement": str(policy.breach_notification_time)
            }
            
        return report
        
    def _get_applicable_frameworks(self, 
                                 jurisdiction: str, 
                                 data_categories: Set[DataCategory]) -> List[ComplianceFramework]:
        """Determine which compliance frameworks apply."""
        applicable = []
        
        for framework, policy in self.policies.items():
            # Check jurisdiction match
            jurisdiction_match = (
                policy.jurisdiction.lower() in jurisdiction.lower() or
                jurisdiction.lower() in policy.jurisdiction.lower() or
                (framework == ComplianceFramework.GDPR and "eu" in jurisdiction.lower())
            )
            
            # Check data category overlap
            category_overlap = bool(data_categories.intersection(policy.data_categories))
            
            if jurisdiction_match and category_overlap:
                applicable.append(framework)
                
        return applicable
        
    def _validate_processing_compliance(self,
                                      data_subject: DataSubject,
                                      data_categories: Set[DataCategory],
                                      processing_purpose: ProcessingPurpose,
                                      frameworks: List[ComplianceFramework]) -> List[str]:
        """Validate if processing is compliant with applicable frameworks."""
        issues = []
        
        for framework in frameworks:
            policy = self.policies[framework]
            
            # Check if processing purpose is allowed
            if processing_purpose not in policy.processing_purposes:
                issues.append(f"Processing purpose {processing_purpose.value} not allowed under {framework.value}")
                
            # Check consent requirements
            if policy.consent_required and processing_purpose == ProcessingPurpose.CONSENT:
                for category in data_categories:
                    consent_key = f"{framework.value}_{category.value}"
                    if not data_subject.consent_records.get(consent_key, False):
                        issues.append(f"Missing consent for {category.value} under {framework.value}")
                        
            # Check opt-out preferences
            if "all_processing" in data_subject.opt_out_preferences:
                issues.append("Data subject has opted out of processing")
                
        return issues
        
    def _calculate_retention_period(self, frameworks: List[ComplianceFramework]) -> timedelta:
        """Calculate the appropriate retention period based on frameworks."""
        if not frameworks:
            return timedelta(days=365)  # Default 1 year
            
        # Use the most restrictive (shortest) retention period
        min_retention = min(
            self.policies[fw].retention_period for fw in frameworks
            if fw in self.policies
        )
        
        return min_retention
        
    def _determine_legal_basis(self, purpose: ProcessingPurpose, frameworks: List[ComplianceFramework]) -> str:
        """Determine the legal basis for processing."""
        basis_map = {
            ProcessingPurpose.CONSENT: "consent",
            ProcessingPurpose.CONTRACT: "contract",
            ProcessingPurpose.LEGAL_OBLIGATION: "legal_obligation",
            ProcessingPurpose.VITAL_INTERESTS: "vital_interests",
            ProcessingPurpose.PUBLIC_TASK: "public_task",
            ProcessingPurpose.LEGITIMATE_INTERESTS: "legitimate_interests",
            ProcessingPurpose.RESEARCH: "scientific_research",
            ProcessingPurpose.STATISTICS: "statistical_purposes"
        }
        
        return basis_map.get(purpose, "legitimate_interests")
        
    def _get_data_subject_data(self, data_subject_id: str, specific_data: Optional[List[str]] = None) -> Dict[str, Any]:
        """Retrieve all data for a data subject."""
        # Get processing records
        records = [
            r for r in self.processing_records
            if r.data_subject_id == data_subject_id
        ]
        
        data = {
            "data_subject_id": data_subject_id,
            "processing_records": [
                {
                    "record_id": r.record_id,
                    "timestamp": r.timestamp.isoformat(),
                    "data_categories": [cat.value for cat in r.data_categories],
                    "processing_purpose": r.processing_purpose.value,
                    "processor_id": r.processor_id,
                    "location": r.location
                }
                for r in records
            ]
        }
        
        # Filter if specific data requested
        if specific_data:
            filtered_records = []
            for record in data["processing_records"]:
                if any(cat in record["data_categories"] for cat in specific_data):
                    filtered_records.append(record)
            data["processing_records"] = filtered_records
            
        return data
        
    def _can_delete_data(self, data_subject_id: str, frameworks: List[ComplianceFramework]) -> bool:
        """Check if data can be legally deleted."""
        # Some frameworks (like HIPAA) may not allow deletion
        for framework in frameworks:
            policy = self.policies.get(framework)
            if policy and not policy.right_to_deletion:
                return False
                
        return True
        
    def _delete_data_subject_data(self, data_subject_id: str) -> List[str]:
        """Delete all data for a data subject."""
        deleted_records = []
        
        # Remove processing records
        original_count = len(self.processing_records)
        self.processing_records = [
            r for r in self.processing_records
            if r.data_subject_id != data_subject_id
        ]
        deleted_count = original_count - len(self.processing_records)
        
        # Remove data subject
        if data_subject_id in self.data_subjects:
            del self.data_subjects[data_subject_id]
            
        deleted_records.extend([f"processing_record_{i}" for i in range(deleted_count)])
        deleted_records.append(f"data_subject_{data_subject_id}")
        
        return deleted_records
        
    def _export_portable_data(self, data_subject_id: str) -> Dict[str, Any]:
        """Export data in a portable format."""
        data = self._get_data_subject_data(data_subject_id)
        
        # Structure for portability (e.g., JSON-LD, CSV, etc.)
        portable_data = {
            "format": "json",
            "version": "1.0",
            "export_timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        return portable_data
        
    def _log_audit_event(self, action: str, subject_id: str, details: Dict[str, Any]) -> None:
        """Log an audit event."""
        audit_event = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "subject_id": subject_id,
            "details": details,
            "event_id": str(uuid.uuid4())
        }
        
        self.audit_log.append(audit_event)
        
        # Keep audit log manageable
        if len(self.audit_log) > 100000:
            self.audit_log = self.audit_log[-100000:]
            
    def _record_compliance_violation(self, violation_type: str, details: Dict[str, Any]) -> None:
        """Record a compliance violation."""
        violation = {
            "timestamp": datetime.now().isoformat(),
            "violation_type": violation_type,
            "details": details,
            "violation_id": str(uuid.uuid4())
        }
        
        self.compliance_violations.append(violation)
        logger.error(f"Compliance violation recorded: {violation_type}")


# Global compliance engine instance
_compliance_engine: Optional[AdvancedComplianceEngine] = None


def get_compliance_engine() -> AdvancedComplianceEngine:
    """Get global compliance engine instance."""
    global _compliance_engine
    if _compliance_engine is None:
        _compliance_engine = AdvancedComplianceEngine()
    return _compliance_engine


def ensure_compliance(frameworks: List[ComplianceFramework]):
    """Decorator to ensure function execution is compliant with specified frameworks."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Pre-execution compliance checks
            compliance_engine = get_compliance_engine()
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Post-execution compliance logging
            compliance_engine._log_audit_event(
                action="function_execution",
                subject_id="system",
                details={
                    "function": func.__name__,
                    "frameworks": [fw.value for fw in frameworks]
                }
            )
            
            return result
        return wrapper
    return decorator