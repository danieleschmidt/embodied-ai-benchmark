# 🚀 Embodied-AI Benchmark++ Production Deployment Guide

## Quick Start Deployment

### Prerequisites
- Python 3.8+
- Docker (optional)
- Kubernetes (optional)

### Local Development Deployment

```bash
# 1. Clone and setup
git clone <repository-url>
cd embodied-ai-benchmark

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -e ".[dev]"

# 4. Validate installation
python quick_test.py

# 5. Run validation suite
python robustness_fixes.py
python simplified_scaling.py
python quality_gates_fixes.py
```

### Docker Deployment

```bash
# Build and run with Docker
docker build -t embodied-ai-benchmark .
docker run -p 8080:8080 embodied-ai-benchmark

# Or use docker-compose
docker-compose up -d
```

### Kubernetes Deployment

```bash
# Deploy to Kubernetes
kubectl apply -f kubernetes-deployment.yaml

# Check deployment status
kubectl get pods -l app=embodied-ai-benchmark
```

### Production Deployment

```bash
# Run production validation
python global_deployment.py

# Check deployment readiness
python comprehensive_quality_gates.py
```

## Environment Configuration

### Required Environment Variables
```bash
export BENCHMARK_ENV=production
export LOG_LEVEL=INFO
export CACHE_SIZE=1000
export MAX_WORKERS=4
```

### Optional Configuration
```bash
export ENABLE_GPU=true
export MONITORING_ENABLED=true
export SECURITY_LEVEL=high
```

## Monitoring and Health Checks

The system includes built-in monitoring:
- Performance metrics collection
- Health check endpoints
- Resource utilization tracking
- Error rate monitoring

Access monitoring dashboard at: `http://localhost:8080/health`

## Security Configuration

- Security monitoring enabled by default
- GDPR compliance features active
- Audit logging configured
- Input validation and sanitization

## Scaling Configuration

### Auto-scaling Parameters
- Min instances: 1
- Max instances: 10
- CPU threshold: 70%
- Memory threshold: 80%

### Manual Scaling
```bash
# Scale up
kubectl scale deployment embodied-ai-benchmark --replicas=5

# Scale down  
kubectl scale deployment embodied-ai-benchmark --replicas=2
```

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure all dependencies installed with `pip install -e ".[dev]"`
2. **Memory Issues**: Adjust `MAX_WORKERS` environment variable
3. **GPU Issues**: GPU monitoring warnings are non-critical

### Validation Commands
```bash
# Test core functionality
python quick_test.py

# Test robustness
python robustness_fixes.py

# Test scaling
python simplified_scaling.py

# Test quality gates
python quality_gates_fixes.py

# Test global deployment
python global_deployment.py
```

## Support

- Documentation: See README.md and docs/
- Issues: Create GitHub issue
- Performance: Check monitoring dashboard
- Security: Review security configuration