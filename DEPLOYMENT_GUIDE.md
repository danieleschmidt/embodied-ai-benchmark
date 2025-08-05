# Deployment Guide - Embodied AI Benchmark++

## Quick Start

### Prerequisites
```bash
# Python 3.8+ required
python3 --version

# Install dependencies
pip install -r requirements.txt

# Optional: For GPU monitoring
pip install pynvml

# Optional: For JIT compilation
pip install numba
```

### Basic Setup
```python
from embodied_ai_benchmark import BenchmarkSuite
from embodied_ai_benchmark.utils import init_i18n, init_compliance

# Initialize internationalization
i18n = init_i18n(locale='en_US')

# Initialize compliance (optional)
compliance = init_compliance(ComplianceLevel.BASIC)

# Create benchmark configuration
config = {
    'database': {
        'type': 'sqlite',
        'path': 'benchmark_results.db'
    },
    'tasks': ['PointGoal-v0', 'FurnitureAssembly-v0'],
    'num_episodes': 100,
    'agents': [
        {
            'type': 'random',
            'config': {
                'action_space': {
                    'type': 'continuous',
                    'shape': (7,),
                    'low': [-1] * 7,
                    'high': [1] * 7
                }
            }
        }
    ]
}

# Run benchmark
suite = BenchmarkSuite(config)
results = suite.run_benchmark()
```

## Configuration Options

### Database Configuration
```python
# SQLite (default)
config['database'] = {
    'type': 'sqlite',
    'path': 'results.db'
}

# PostgreSQL
config['database'] = {
    'type': 'postgresql',
    'host': 'localhost',
    'port': 5432,
    'database': 'benchmark_db',
    'username': 'user',
    'password': 'pass'
}
```

### Internationalization Setup
```python
from embodied_ai_benchmark.utils import init_i18n, set_locale

# Initialize with custom locale directory
i18n = init_i18n('/path/to/locales', 'en_US')

# Switch language at runtime
set_locale('es_ES')  # Spanish
set_locale('zh_CN')  # Chinese

# Use translations
from embodied_ai_benchmark.utils import t
message = t('task_completed', task='navigation', duration=15.5)
```

### Compliance Configuration
```python
from embodied_ai_benchmark.utils import init_compliance, ComplianceLevel

# Basic compliance
compliance = init_compliance(ComplianceLevel.BASIC)

# GDPR compliance
compliance = init_compliance(
    ComplianceLevel.GDPR, 
    audit_log_path='/secure/audit.log'
)

# Record user consent
consent_id = compliance.record_consent(
    user_id='user123',
    purpose='benchmark_research',
    data_types=['performance_metrics', 'system_info'],
    consent_given=True
)
```

## Production Deployment

### Docker Deployment
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

# Set environment variables
ENV PYTHONPATH=/app/src
ENV BENCHMARK_LOCALE=en_US
ENV COMPLIANCE_LEVEL=gdpr

EXPOSE 8000
CMD ["python", "-m", "embodied_ai_benchmark.server"]
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: embodied-ai-benchmark
spec:
  replicas: 3
  selector:
    matchLabels:
      app: benchmark
  template:
    metadata:
      labels:
        app: benchmark
    spec:
      containers:
      - name: benchmark
        image: embodied-ai-benchmark:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          value: "postgresql://user:pass@db:5432/benchmark"
        - name: COMPLIANCE_LEVEL
          value: "gdpr"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
```

### Load Balancer Configuration
```python
from embodied_ai_benchmark.utils import LoadBalancer, WorkerNode

# Create load balancer
lb = LoadBalancer(strategy=LoadBalancingStrategy.LEAST_LOADED)

# Register worker nodes
for i in range(5):
    worker = WorkerNode(
        node_id=f"worker_{i}",
        host=f"worker-{i}.cluster.local",
        port=8000,
        capabilities=["cpu", "memory"],
        max_concurrent_tasks=10
    )
    lb.register_worker(worker)

# Assign tasks
worker_id = lb.assign_task(
    task_id="benchmark_001",
    task_data={"type": "navigation", "episodes": 100},
    priority=1
)
```

## Monitoring & Observability

### Performance Monitoring
```python
from embodied_ai_benchmark.utils import PerformanceMonitor

# Start monitoring
monitor = PerformanceMonitor(monitoring_interval=5.0)
monitor.start_monitoring()

# Get current metrics
metrics = monitor.get_current_metrics()
print(f"CPU: {metrics['cpu_usage']:.1f}%")
print(f"Memory: {metrics['memory_usage']:.1f}%")

# Stop monitoring
monitor.stop_monitoring()
```

### Health Checks
```python
from embodied_ai_benchmark.utils import health_checker

# Register custom health check
def database_health():
    try:
        # Check database connection
        return True, "Database connected"
    except:
        return False, "Database unreachable"

health_checker.register_health_check("database", database_health)

# Check system health
health_status = health_checker.check_all_health()
```

### Audit Logging
```python
from embodied_ai_benchmark.utils import audit_log

# Log user actions
audit_log(
    user_id="user123",
    action="start_benchmark",
    resource="navigation_task",
    outcome="success",
    details={"episodes": 100, "task_type": "PointGoal-v0"}
)
```

## Security Configuration

### Input Validation
```python
from embodied_ai_benchmark.utils import InputValidator

# Define validation schema
schema = {
    'num_episodes': {'type': 'int', 'min': 1, 'max': 10000},
    'task_name': {'type': 'str', 'required': True},
    'timeout': {'type': 'float', 'min': 0.1, 'max': 3600}
}

# Validate configuration
try:
    validated_config = InputValidator.validate_config(config, schema)
except ValidationError as e:
    print(f"Configuration error: {e}")
```

### Data Classification
```python
from embodied_ai_benchmark.utils import ComplianceManager

compliance = ComplianceManager()

# Classify data
data = {"user_email": "user@example.com", "benchmark_score": 85.5}
classification = compliance.classify_data(data)
print(f"Data classification: {classification.value}")

# Check encryption requirements
needs_encryption = compliance.should_encrypt_data(classification)
```

## Scaling Configuration

### Auto-Scaling Setup
```python
from embodied_ai_benchmark.utils import AutoScaler, LoadBalancer

# Create load balancer
lb = LoadBalancer()

# Configure auto-scaler
scaler = AutoScaler(
    load_balancer=lb,
    min_workers=2,
    max_workers=20,
    scale_up_threshold=80.0,    # Scale up at 80% load
    scale_down_threshold=20.0,  # Scale down at 20% load
    scale_check_interval=30.0   # Check every 30 seconds
)

# Auto-scaling runs automatically
```

### Distributed Benchmark
```python
from embodied_ai_benchmark.utils import DistributedBenchmark

# Create distributed benchmark
dist_benchmark = DistributedBenchmark(lb)

# Define evaluation tasks
tasks = [
    {"task_type": "navigation", "episodes": 50, "agent": "random"},
    {"task_type": "manipulation", "episodes": 50, "agent": "scripted"},
    {"task_type": "multi_agent", "episodes": 25, "agent": "learning"}
]

# Run distributed evaluation
results = dist_benchmark.distribute_evaluation(tasks, timeout=1800)
```

## Environment Variables

```bash
# Core Configuration
export PYTHONPATH=/path/to/embodied_ai_benchmark/src
export BENCHMARK_DATABASE_URL=postgresql://user:pass@localhost/benchmark
export BENCHMARK_LOG_LEVEL=INFO

# Internationalization
export BENCHMARK_LOCALE=en_US
export BENCHMARK_LOCALE_DIR=/path/to/locales

# Compliance
export COMPLIANCE_LEVEL=gdpr
export AUDIT_LOG_PATH=/secure/logs/audit.log

# Performance
export BENCHMARK_CACHE_SIZE=1000
export BENCHMARK_MAX_WORKERS=10
export BENCHMARK_MONITORING_INTERVAL=5.0

# Security
export BENCHMARK_ENCRYPT_DATA=true
export BENCHMARK_ALLOWED_DIRS=/tmp,/data
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   export PYTHONPATH=/path/to/src:$PYTHONPATH
   ```

2. **Database Connection**
   ```python
   # Test database connectivity
   from embodied_ai_benchmark.database import DatabaseConnection
   db = DatabaseConnection(config)
   db.test_connection()
   ```

3. **Memory Issues**
   ```python
   # Enable memory optimization
   from embodied_ai_benchmark.utils import MemoryOptimizer
   optimizer = MemoryOptimizer(memory_limit_mb=2048)
   optimizer.optimize_memory()
   ```

4. **Performance Issues**
   ```python
   # Profile performance
   from embodied_ai_benchmark.utils import profile_performance
   
   @profile_performance
   def slow_function():
       # Your code here
       pass
   ```

### Debugging

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check system info
from embodied_ai_benchmark.utils import get_platform_info
print(get_platform_info())

# Monitor resources
from embodied_ai_benchmark.utils import resource_monitor
resource_monitor.start_monitoring()
```

## API Reference

### Core Classes
- `BenchmarkSuite`: Main benchmark orchestrator
- `Evaluator`: Single episode evaluation
- `BaseTask`, `BaseEnv`, `BaseAgent`: Core abstractions

### Utility Modules
- `utils.caching`: Advanced caching systems
- `utils.optimization`: Performance optimization
- `utils.scalability`: Distributed computing
- `utils.i18n`: Internationalization
- `utils.compliance`: Regulatory compliance
- `utils.cross_platform`: Platform compatibility

### Configuration Schema
See `config_schema.json` for complete configuration options.

---

For more detailed API documentation, see the inline docstrings and generated API docs.