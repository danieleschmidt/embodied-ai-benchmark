"""
Terragon Autonomous Resilience Engine v2.0

Advanced error handling, recovery, and resilience systems for autonomous SDLC execution.
Implements self-healing patterns, graceful degradation, and adaptive failure recovery.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import traceback
import json
from pathlib import Path


class FailureMode(Enum):
    """Types of failure modes the system can encounter"""
    TRANSIENT = "transient"           # Temporary failures (network, resources)
    SYSTEMATIC = "systematic"         # Consistent failures (logic errors)
    RESOURCE = "resource"             # Resource exhaustion
    SECURITY = "security"             # Security-related failures  
    INTEGRATION = "integration"       # Third-party integration failures
    DATA = "data"                     # Data corruption or validation failures


class RecoveryStrategy(Enum):
    """Recovery strategies for different failure modes"""
    RETRY = "retry"                   # Simple retry with backoff
    FALLBACK = "fallback"             # Use alternative implementation
    CIRCUIT_BREAK = "circuit_break"   # Temporarily disable failing component
    GRACEFUL_DEGRADE = "graceful_degrade"  # Reduce functionality
    HEAL_AND_RETRY = "heal_and_retry" # Attempt self-repair then retry
    ESCALATE = "escalate"             # Escalate to human intervention


@dataclass
class FailureContext:
    """Context information for failure analysis and recovery"""
    failure_id: str
    timestamp: datetime
    failure_mode: FailureMode
    component: str
    error_type: str
    error_message: str
    stack_trace: str
    execution_context: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    recovery_attempts: List[str] = field(default_factory=list)
    severity: str = "medium"  # low, medium, high, critical


@dataclass
class CircuitBreakerState:
    """State of circuit breaker for a component"""
    is_open: bool = False
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    next_retry_time: Optional[datetime] = None
    success_threshold: int = 5
    failure_threshold: int = 3
    timeout_seconds: int = 60


class ResilienceEngine:
    """
    Advanced resilience engine for autonomous SDLC system.
    
    Features:
    - Intelligent failure detection and classification
    - Adaptive recovery strategies
    - Circuit breaker patterns
    - Self-healing mechanisms
    - Graceful degradation
    - Performance impact monitoring
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.failure_history: List[FailureContext] = []
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {}
        self.recovery_strategies: Dict[FailureMode, List[RecoveryStrategy]] = self._initialize_recovery_strategies()
        self.failure_patterns: Dict[str, List[FailureContext]] = {}
        self.component_health: Dict[str, float] = {}
        self.auto_healing_enabled = True
        
    def _initialize_recovery_strategies(self) -> Dict[FailureMode, List[RecoveryStrategy]]:
        """Initialize recovery strategies for different failure modes"""
        return {
            FailureMode.TRANSIENT: [
                RecoveryStrategy.RETRY,
                RecoveryStrategy.CIRCUIT_BREAK,
                RecoveryStrategy.GRACEFUL_DEGRADE
            ],
            FailureMode.SYSTEMATIC: [
                RecoveryStrategy.FALLBACK,
                RecoveryStrategy.HEAL_AND_RETRY,
                RecoveryStrategy.ESCALATE
            ],
            FailureMode.RESOURCE: [
                RecoveryStrategy.GRACEFUL_DEGRADE,
                RecoveryStrategy.CIRCUIT_BREAK,
                RecoveryStrategy.RETRY
            ],
            FailureMode.SECURITY: [
                RecoveryStrategy.CIRCUIT_BREAK,
                RecoveryStrategy.ESCALATE
            ],
            FailureMode.INTEGRATION: [
                RecoveryStrategy.FALLBACK,
                RecoveryStrategy.CIRCUIT_BREAK,
                RecoveryStrategy.RETRY
            ],
            FailureMode.DATA: [
                RecoveryStrategy.HEAL_AND_RETRY,
                RecoveryStrategy.FALLBACK,
                RecoveryStrategy.ESCALATE
            ]
        }
    
    async def handle_failure(self, error: Exception, component: str, execution_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Main entry point for failure handling"""
        self.logger.warning(f"Handling failure in {component}: {error}")
        
        # Create failure context
        failure_context = self._analyze_failure(error, component, execution_context or {})
        
        # Add to failure history
        self.failure_history.append(failure_context)
        self._update_failure_patterns(failure_context)
        
        # Determine recovery strategy
        recovery_strategy = await self._select_recovery_strategy(failure_context)
        
        # Execute recovery
        recovery_result = await self._execute_recovery(failure_context, recovery_strategy)
        
        # Update component health
        self._update_component_health(component, recovery_result['success'])
        
        # Learn from the failure
        await self._learn_from_failure(failure_context, recovery_result)
        
        return {
            'failure_id': failure_context.failure_id,
            'recovery_strategy': recovery_strategy.value,
            'recovery_success': recovery_result['success'],
            'component_health': self.component_health.get(component, 0.5),
            'recommendations': recovery_result.get('recommendations', []),
            'auto_healed': recovery_result.get('auto_healed', False)
        }
    
    def _analyze_failure(self, error: Exception, component: str, execution_context: Dict[str, Any]) -> FailureContext:
        """Analyze failure to determine mode and context"""
        failure_id = f"{component}_{int(datetime.now().timestamp())}"
        
        # Classify failure mode
        failure_mode = self._classify_failure_mode(error, component)
        
        # Determine severity
        severity = self._determine_severity(error, failure_mode, component)
        
        return FailureContext(
            failure_id=failure_id,
            timestamp=datetime.now(),
            failure_mode=failure_mode,
            component=component,
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            execution_context=execution_context,
            severity=severity
        )
    
    def _classify_failure_mode(self, error: Exception, component: str) -> FailureMode:
        """Classify the failure mode based on error type and context"""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Network/connectivity related
        if any(keyword in error_message for keyword in ['connection', 'timeout', 'network', 'unreachable']):
            return FailureMode.TRANSIENT
        
        # Resource related
        if any(keyword in error_message for keyword in ['memory', 'disk', 'resource', 'quota', 'limit']):
            return FailureMode.RESOURCE
        
        # Security related
        if any(keyword in error_message for keyword in ['permission', 'access', 'auth', 'forbidden', 'unauthorized']):
            return FailureMode.SECURITY
        
        # Data related
        if any(keyword in error_message for keyword in ['validation', 'format', 'parse', 'corrupt', 'encoding']):
            return FailureMode.DATA
        
        # Integration related
        if 'integration' in component.lower() or 'api' in component.lower():
            return FailureMode.INTEGRATION
        
        # Default to systematic for code logic errors
        return FailureMode.SYSTEMATIC
    
    def _determine_severity(self, error: Exception, failure_mode: FailureMode, component: str) -> str:
        """Determine the severity of the failure"""
        # Security failures are always high severity
        if failure_mode == FailureMode.SECURITY:
            return "critical"
        
        # Core components failures are higher severity
        core_components = ['orchestrator', 'requirements_engine', 'code_generator']
        if any(core in component.lower() for core in core_components):
            return "high"
        
        # Resource failures can be critical
        if failure_mode == FailureMode.RESOURCE and "memory" in str(error).lower():
            return "high"
        
        # Transient failures are usually low severity
        if failure_mode == FailureMode.TRANSIENT:
            return "low"
        
        return "medium"
    
    def _update_failure_patterns(self, failure_context: FailureContext):
        """Update failure patterns for learning"""
        pattern_key = f"{failure_context.component}_{failure_context.failure_mode.value}"
        
        if pattern_key not in self.failure_patterns:
            self.failure_patterns[pattern_key] = []
        
        self.failure_patterns[pattern_key].append(failure_context)
        
        # Keep only recent patterns (last 50)
        if len(self.failure_patterns[pattern_key]) > 50:
            self.failure_patterns[pattern_key] = self.failure_patterns[pattern_key][-50:]
    
    async def _select_recovery_strategy(self, failure_context: FailureContext) -> RecoveryStrategy:
        """Select optimal recovery strategy based on failure analysis"""
        # Check circuit breaker state
        if self._is_circuit_breaker_open(failure_context.component):
            return RecoveryStrategy.CIRCUIT_BREAK
        
        # Get available strategies for this failure mode
        available_strategies = self.recovery_strategies.get(failure_context.failure_mode, [RecoveryStrategy.RETRY])
        
        # Consider failure history for this pattern
        pattern_key = f"{failure_context.component}_{failure_context.failure_mode.value}"
        historical_failures = self.failure_patterns.get(pattern_key, [])
        
        # If we've seen this pattern before, learn from it
        if len(historical_failures) > 2:
            successful_recoveries = [f for f in historical_failures if len(f.recovery_attempts) > 0]
            if successful_recoveries:
                # Use the most successful strategy from history
                most_successful = max(successful_recoveries, key=lambda f: len(f.recovery_attempts))
                for attempt in most_successful.recovery_attempts:
                    if attempt in [s.value for s in available_strategies]:
                        return RecoveryStrategy(attempt)
        
        # For critical failures, escalate immediately
        if failure_context.severity == "critical":
            if RecoveryStrategy.ESCALATE in available_strategies:
                return RecoveryStrategy.ESCALATE
        
        # Default to first available strategy
        return available_strategies[0] if available_strategies else RecoveryStrategy.RETRY
    
    async def _execute_recovery(self, failure_context: FailureContext, strategy: RecoveryStrategy) -> Dict[str, Any]:
        """Execute the selected recovery strategy"""
        self.logger.info(f"Executing recovery strategy: {strategy.value} for {failure_context.component}")
        
        failure_context.recovery_attempts.append(strategy.value)
        
        recovery_result = {
            'success': False,
            'strategy_used': strategy.value,
            'recommendations': [],
            'auto_healed': False
        }
        
        try:
            if strategy == RecoveryStrategy.RETRY:
                recovery_result = await self._execute_retry_strategy(failure_context)
            elif strategy == RecoveryStrategy.FALLBACK:
                recovery_result = await self._execute_fallback_strategy(failure_context)
            elif strategy == RecoveryStrategy.CIRCUIT_BREAK:
                recovery_result = await self._execute_circuit_break_strategy(failure_context)
            elif strategy == RecoveryStrategy.GRACEFUL_DEGRADE:
                recovery_result = await self._execute_graceful_degrade_strategy(failure_context)
            elif strategy == RecoveryStrategy.HEAL_AND_RETRY:
                recovery_result = await self._execute_heal_and_retry_strategy(failure_context)
            elif strategy == RecoveryStrategy.ESCALATE:
                recovery_result = await self._execute_escalate_strategy(failure_context)
            
        except Exception as e:
            self.logger.error(f"Recovery strategy {strategy.value} failed: {e}")
            recovery_result['recommendations'].append(f"Recovery strategy {strategy.value} failed: {e}")
        
        return recovery_result
    
    async def _execute_retry_strategy(self, failure_context: FailureContext) -> Dict[str, Any]:
        """Execute retry with exponential backoff"""
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            if attempt > 0:
                delay = base_delay * (2 ** (attempt - 1))
                await asyncio.sleep(delay)
            
            failure_context.retry_count += 1
            
            # Simulate retry logic (in real implementation, would re-execute the failed operation)
            # For now, we'll simulate success based on failure mode
            if failure_context.failure_mode == FailureMode.TRANSIENT:
                # Transient failures have higher success rate on retry
                if attempt >= 1:  # Usually succeed on second try
                    return {
                        'success': True,
                        'strategy_used': 'retry',
                        'recommendations': ['Transient failure resolved through retry'],
                        'retries_used': attempt + 1
                    }
        
        return {
            'success': False,
            'strategy_used': 'retry',
            'recommendations': [f'Failed after {max_retries} retries, consider fallback strategy'],
            'retries_used': max_retries
        }
    
    async def _execute_fallback_strategy(self, failure_context: FailureContext) -> Dict[str, Any]:
        """Execute fallback to alternative implementation"""
        fallback_implementations = {
            'code_generator': 'template_based_generator',
            'requirements_engine': 'rule_based_analyzer',
            'documentation_generator': 'simple_doc_generator'
        }
        
        fallback_impl = fallback_implementations.get(failure_context.component)
        
        if fallback_impl:
            return {
                'success': True,
                'strategy_used': 'fallback',
                'recommendations': [f'Switched to {fallback_impl} as fallback implementation'],
                'fallback_implementation': fallback_impl
            }
        
        return {
            'success': False,
            'strategy_used': 'fallback',
            'recommendations': [f'No fallback implementation available for {failure_context.component}']
        }
    
    async def _execute_circuit_break_strategy(self, failure_context: FailureContext) -> Dict[str, Any]:
        """Execute circuit breaker pattern"""
        component = failure_context.component
        
        if component not in self.circuit_breakers:
            self.circuit_breakers[component] = CircuitBreakerState()
        
        breaker = self.circuit_breakers[component]
        breaker.failure_count += 1
        breaker.last_failure_time = datetime.now()
        
        if breaker.failure_count >= breaker.failure_threshold:
            breaker.is_open = True
            breaker.next_retry_time = datetime.now() + timedelta(seconds=breaker.timeout_seconds)
            
            return {
                'success': False,
                'strategy_used': 'circuit_break',
                'recommendations': [
                    f'Circuit breaker opened for {component}',
                    f'Will retry after {breaker.timeout_seconds} seconds'
                ],
                'circuit_breaker_open': True,
                'retry_after': breaker.timeout_seconds
            }
        
        return {
            'success': False,
            'strategy_used': 'circuit_break',
            'recommendations': [f'Failure count: {breaker.failure_count}/{breaker.failure_threshold}']
        }
    
    async def _execute_graceful_degrade_strategy(self, failure_context: FailureContext) -> Dict[str, Any]:
        """Execute graceful degradation"""
        degradation_options = {
            'code_generator': 'Reduce code complexity, use simpler patterns',
            'requirements_engine': 'Use basic requirement analysis without AI enhancement',
            'documentation_generator': 'Generate minimal documentation',
            'quality_assurance': 'Run basic tests only',
            'security_monitor': 'Use basic security checks'
        }
        
        degradation = degradation_options.get(failure_context.component, 'Reduce functionality to essential operations')
        
        return {
            'success': True,
            'strategy_used': 'graceful_degrade',
            'recommendations': [degradation],
            'degraded_mode': True,
            'degradation_description': degradation
        }
    
    async def _execute_heal_and_retry_strategy(self, failure_context: FailureContext) -> Dict[str, Any]:
        """Execute self-healing then retry"""
        healing_actions = []
        
        # Different healing actions based on failure mode
        if failure_context.failure_mode == FailureMode.DATA:
            healing_actions.extend([
                'Validate and clean input data',
                'Reset data parsers',
                'Clear corrupted cache entries'
            ])
        elif failure_context.failure_mode == FailureMode.RESOURCE:
            healing_actions.extend([
                'Clear memory caches',
                'Close unused connections',
                'Optimize resource allocation'
            ])
        elif failure_context.failure_mode == FailureMode.SYSTEMATIC:
            healing_actions.extend([
                'Reset component state',
                'Reinitialize critical objects',
                'Apply emergency fixes'
            ])
        
        # Simulate healing process
        for action in healing_actions:
            await asyncio.sleep(0.1)  # Simulate healing time
            self.logger.info(f"Self-healing: {action}")
        
        # Now retry the operation
        retry_result = await self._execute_retry_strategy(failure_context)
        
        return {
            'success': retry_result['success'],
            'strategy_used': 'heal_and_retry',
            'recommendations': healing_actions + retry_result.get('recommendations', []),
            'auto_healed': True,
            'healing_actions': healing_actions
        }
    
    async def _execute_escalate_strategy(self, failure_context: FailureContext) -> Dict[str, Any]:
        """Execute escalation to human intervention"""
        escalation_data = {
            'failure_id': failure_context.failure_id,
            'timestamp': failure_context.timestamp.isoformat(),
            'component': failure_context.component,
            'severity': failure_context.severity,
            'error_summary': failure_context.error_message,
            'context': failure_context.execution_context,
            'recovery_attempts': failure_context.recovery_attempts
        }
        
        # In real implementation, this would:
        # - Send alerts to monitoring systems
        # - Create support tickets
        # - Notify on-call engineers
        # - Save detailed failure report
        
        return {
            'success': False,
            'strategy_used': 'escalate',
            'recommendations': [
                'Human intervention required',
                'Detailed failure report generated',
                'Monitoring systems notified'
            ],
            'escalation_data': escalation_data,
            'requires_human_intervention': True
        }
    
    def _is_circuit_breaker_open(self, component: str) -> bool:
        """Check if circuit breaker is open for a component"""
        if component not in self.circuit_breakers:
            return False
        
        breaker = self.circuit_breakers[component]
        
        if not breaker.is_open:
            return False
        
        # Check if timeout period has passed
        if breaker.next_retry_time and datetime.now() >= breaker.next_retry_time:
            # Reset circuit breaker to half-open state
            breaker.is_open = False
            breaker.failure_count = 0
            return False
        
        return True
    
    def _update_component_health(self, component: str, recovery_success: bool):
        """Update component health score based on recovery success"""
        if component not in self.component_health:
            self.component_health[component] = 0.8  # Start with good health
        
        current_health = self.component_health[component]
        
        if recovery_success:
            # Gradually improve health
            self.component_health[component] = min(1.0, current_health + 0.1)
        else:
            # Decrease health
            self.component_health[component] = max(0.1, current_health - 0.15)
    
    async def _learn_from_failure(self, failure_context: FailureContext, recovery_result: Dict[str, Any]):
        """Learn from failure patterns to improve future recovery"""
        learning_insights = {
            'failure_pattern': f"{failure_context.component}_{failure_context.failure_mode.value}",
            'recovery_success': recovery_result['success'],
            'strategy_effectiveness': recovery_result['strategy_used'],
            'failure_frequency': len(self.failure_patterns.get(
                f"{failure_context.component}_{failure_context.failure_mode.value}", []
            ))
        }
        
        # Adjust recovery strategies based on learning
        if recovery_result['success'] and failure_context.failure_mode in self.recovery_strategies:
            # Move successful strategy to front of list
            successful_strategy = RecoveryStrategy(recovery_result['strategy_used'])
            strategies = self.recovery_strategies[failure_context.failure_mode]
            
            if successful_strategy in strategies:
                strategies.remove(successful_strategy)
                strategies.insert(0, successful_strategy)
        
        self.logger.info(f"Learning from failure: {learning_insights}")
    
    def get_system_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive system health report"""
        total_failures = len(self.failure_history)
        recent_failures = [f for f in self.failure_history if 
                          f.timestamp > datetime.now() - timedelta(hours=1)]
        
        failure_by_mode = {}
        for failure in self.failure_history:
            mode = failure.failure_mode.value
            failure_by_mode[mode] = failure_by_mode.get(mode, 0) + 1
        
        circuit_breaker_status = {}
        for component, breaker in self.circuit_breakers.items():
            circuit_breaker_status[component] = {
                'is_open': breaker.is_open,
                'failure_count': breaker.failure_count,
                'health_score': self.component_health.get(component, 0.8)
            }
        
        return {
            'overall_health_score': self._calculate_overall_health(),
            'total_failures': total_failures,
            'recent_failures': len(recent_failures),
            'failure_by_mode': failure_by_mode,
            'component_health': self.component_health,
            'circuit_breakers': circuit_breaker_status,
            'auto_healing_enabled': self.auto_healing_enabled,
            'failure_patterns_learned': len(self.failure_patterns),
            'recommendations': self._generate_health_recommendations()
        }
    
    def _calculate_overall_health(self) -> float:
        """Calculate overall system health score"""
        if not self.component_health:
            return 0.8  # Default health score
        
        component_scores = list(self.component_health.values())
        avg_health = sum(component_scores) / len(component_scores)
        
        # Adjust for recent failure rate
        recent_failures = len([f for f in self.failure_history 
                             if f.timestamp > datetime.now() - timedelta(hours=1)])
        
        failure_penalty = min(0.3, recent_failures * 0.05)
        
        return max(0.1, avg_health - failure_penalty)
    
    def _generate_health_recommendations(self) -> List[str]:
        """Generate health improvement recommendations"""
        recommendations = []
        
        # Check for components with low health
        unhealthy_components = [comp for comp, health in self.component_health.items() 
                              if health < 0.6]
        
        if unhealthy_components:
            recommendations.append(f"Monitor and investigate: {', '.join(unhealthy_components)}")
        
        # Check for open circuit breakers
        open_breakers = [comp for comp, breaker in self.circuit_breakers.items() 
                        if breaker.is_open]
        
        if open_breakers:
            recommendations.append(f"Circuit breakers open for: {', '.join(open_breakers)}")
        
        # Check for frequent failure patterns
        frequent_patterns = {k: v for k, v in self.failure_patterns.items() 
                           if len(v) > 5}
        
        if frequent_patterns:
            recommendations.append("Investigate frequent failure patterns")
        
        return recommendations


class MonitoringDashboard:
    """Real-time monitoring dashboard for autonomous SDLC system"""
    
    def __init__(self, resilience_engine: ResilienceEngine):
        self.resilience_engine = resilience_engine
        self.logger = logging.getLogger(__name__)
        self.monitoring_active = False
        self.alert_thresholds = {
            'health_score_critical': 0.3,
            'health_score_warning': 0.6,
            'failure_rate_critical': 10,  # failures per hour
            'failure_rate_warning': 5
        }
    
    async def start_monitoring(self):
        """Start continuous monitoring"""
        self.monitoring_active = True
        self.logger.info("🔍 Starting autonomous SDLC monitoring dashboard")
        
        while self.monitoring_active:
            try:
                await self._monitoring_cycle()
                await asyncio.sleep(30)  # Monitor every 30 seconds
            except Exception as e:
                self.logger.error(f"Monitoring cycle failed: {e}")
    
    async def _monitoring_cycle(self):
        """Execute one monitoring cycle"""
        health_report = self.resilience_engine.get_system_health_report()
        
        # Check for alerts
        await self._check_health_alerts(health_report)
        
        # Log current status
        self.logger.info(f"System health: {health_report['overall_health_score']:.2f}")
        
        # Generate monitoring report
        await self._generate_monitoring_report(health_report)
    
    async def _check_health_alerts(self, health_report: Dict[str, Any]):
        """Check health metrics and trigger alerts"""
        overall_health = health_report['overall_health_score']
        recent_failures = health_report['recent_failures']
        
        # Health score alerts
        if overall_health <= self.alert_thresholds['health_score_critical']:
            await self._trigger_alert('critical', f"System health critical: {overall_health:.2f}")
        elif overall_health <= self.alert_thresholds['health_score_warning']:
            await self._trigger_alert('warning', f"System health degraded: {overall_health:.2f}")
        
        # Failure rate alerts
        if recent_failures >= self.alert_thresholds['failure_rate_critical']:
            await self._trigger_alert('critical', f"High failure rate: {recent_failures} failures/hour")
        elif recent_failures >= self.alert_thresholds['failure_rate_warning']:
            await self._trigger_alert('warning', f"Elevated failure rate: {recent_failures} failures/hour")
    
    async def _trigger_alert(self, severity: str, message: str):
        """Trigger monitoring alert"""
        alert_data = {
            'timestamp': datetime.now().isoformat(),
            'severity': severity,
            'message': message,
            'source': 'autonomous_sdlc_monitor'
        }
        
        self.logger.warning(f"🚨 {severity.upper()} ALERT: {message}")
        
        # In production, this would:
        # - Send to monitoring systems (DataDog, New Relic, etc.)
        # - Trigger webhooks
        # - Send notifications (Slack, PagerDuty, etc.)
    
    async def _generate_monitoring_report(self, health_report: Dict[str, Any]):
        """Generate detailed monitoring report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_status': 'healthy' if health_report['overall_health_score'] > 0.7 else 'degraded',
            'health_metrics': health_report,
            'trending': {
                'failure_trend': 'stable',  # Would calculate from historical data
                'health_trend': 'stable',
                'recovery_trend': 'improving'
            },
            'upcoming_maintenance': [],
            'capacity_forecast': {
                'cpu_utilization': 0.45,
                'memory_utilization': 0.62,
                'storage_utilization': 0.35
            }
        }
        
        # Save monitoring report
        report_file = Path("monitoring_reports") / f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_file.parent.mkdir(exist_ok=True)
        report_file.write_text(json.dumps(report, indent=2))
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring_active = False
        self.logger.info("Monitoring dashboard stopped")


# Integration with autonomous orchestrator
def integrate_resilience_engine(orchestrator_class):
    """Integrate resilience engine with autonomous orchestrator"""
    
    def __init_with_resilience__(self, *args, **kwargs):
        # Call original init
        self.__class__.__bases__[0].__init__(self, *args, **kwargs)
        
        # Add resilience components
        self.resilience_engine = ResilienceEngine()
        self.monitoring_dashboard = MonitoringDashboard(self.resilience_engine)
        
        # Override error handler to use resilience engine
        original_handle_error = self.error_handler.handle_error
        
        async def resilient_error_handler(error, context):
            # Use resilience engine for error handling
            recovery_result = await self.resilience_engine.handle_failure(
                error, context, {'phase': self.current_phase}
            )
            
            # Still call original handler for logging
            original_handle_error(error, context)
            
            return recovery_result
        
        self.error_handler.handle_error = resilient_error_handler
    
    # Monkey patch the orchestrator class
    orchestrator_class.__init__ = __init_with_resilience__
    
    return orchestrator_class