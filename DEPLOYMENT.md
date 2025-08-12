# Embodied AI Benchmark++ Deployment Guide

This guide provides comprehensive instructions for deploying the Embodied AI Benchmark++ in various environments, from local development to production-scale Kubernetes clusters.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Local Development](#local-development)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Production Considerations](#production-considerations)
6. [Monitoring and Observability](#monitoring-and-observability)
7. [Security Guidelines](#security-guidelines)
8. [Troubleshooting](#troubleshooting)

## Quick Start

### Prerequisites

- Python 3.8+ 
- Docker and Docker Compose
- Kubernetes cluster (for production deployment)
- 4GB+ RAM recommended
- 10GB+ disk space

### Local Installation

```bash
# Clone the repository
git clone https://github.com/your-org/embodied-ai-benchmark.git
cd embodied-ai-benchmark

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Run basic validation
python simple_env_test.py
```

## Local Development

### Development Setup

1. **Environment Configuration**
   ```bash
   # Copy environment template
   cp .env.example .env
   
   # Edit configuration
   vim .env
   ```

2. **Development Dependencies**
   ```bash
   pip install -e ".[dev,test]"
   ```

3. **Run Development Server**
   ```bash
   # Start API server
   python -m src.embodied_ai_benchmark.api.app
   
   # Or use development mode
   BENCHMARK_ENV=development python -m src.embodied_ai_benchmark.api.app
   ```

### Development Tools

- **Code Formatting**: `black src/ tests/`
- **Linting**: `flake8 src/ tests/`
- **Type Checking**: `mypy src/`
- **Testing**: `pytest tests/`

## Docker Deployment

### Single Container Deployment

```bash
# Build the image
docker build -t embodied-ai-benchmark:latest .

# Run container
docker run -d \
  --name benchmark-app \
  -p 8080:8080 \
  -e BENCHMARK_ENV=production \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  embodied-ai-benchmark:latest
```

### Docker Compose Deployment

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f benchmark-app

# Scale application
docker-compose up -d --scale benchmark-app=3

# Stop services
docker-compose down
```

#### Services Included

- **benchmark-app**: Main application server
- **redis**: Caching and message queuing
- **postgres**: Data persistence
- **prometheus**: Metrics collection
- **grafana**: Monitoring dashboards
- **nginx**: Reverse proxy and load balancer

### Docker Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCHMARK_ENV` | `production` | Environment mode |
| `BENCHMARK_LOG_LEVEL` | `INFO` | Logging level |
| `BENCHMARK_API_HOST` | `0.0.0.0` | API bind address |
| `BENCHMARK_API_PORT` | `8080` | API port |
| `BENCHMARK_WORKERS` | `4` | Number of worker processes |
| `BENCHMARK_CACHE_SIZE` | `1000` | Cache size limit |
| `BENCHMARK_MAX_MEMORY_MB` | `2048` | Memory limit in MB |

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (1.20+)
- kubectl configured
- Helm 3.x (optional)
- Ingress controller (nginx recommended)

### Basic Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f kubernetes-deployment.yaml

# Check deployment status
kubectl get pods -n embodied-ai-benchmark

# View logs
kubectl logs -f deployment/benchmark-app -n embodied-ai-benchmark
```

### Configuration

1. **Update Secrets**
   ```bash
   # Encode your actual secrets
   echo -n "your_database_url" | base64
   echo -n "your_redis_url" | base64
   echo -n "your_secret_key" | base64
   
   # Update kubernetes-deployment.yaml with encoded values
   ```

2. **Configure Ingress**
   ```yaml
   # Update ingress host in kubernetes-deployment.yaml
   rules:
   - host: your-domain.com
     http:
       paths:
       - path: /
   ```

3. **SSL Certificate**
   ```bash
   # Create TLS secret
   kubectl create secret tls benchmark-tls \
     --cert=path/to/cert.crt \
     --key=path/to/cert.key \
     -n embodied-ai-benchmark
   ```

### Scaling

The deployment includes automatic horizontal pod autoscaling:

- **Min replicas**: 2
- **Max replicas**: 10  
- **CPU target**: 70%
- **Memory target**: 80%

Manual scaling:
```bash
kubectl scale deployment benchmark-app --replicas=5 -n embodied-ai-benchmark
```

### Storage

The deployment uses persistent volumes:

- **Application data**: 10GB SSD
- **Logs**: 5GB standard storage
- **Redis data**: 2GB SSD

Update storage classes and sizes in `kubernetes-deployment.yaml` as needed.

## Production Considerations

### Performance Optimization

1. **Resource Allocation**
   ```yaml
   resources:
     requests:
       memory: "1Gi"
       cpu: "500m"
     limits:
       memory: "4Gi"
       cpu: "2000m"
   ```

2. **Database Optimization**
   ```sql
   -- PostgreSQL optimization
   shared_buffers = '256MB'
   effective_cache_size = '1GB'
   checkpoint_completion_target = 0.9
   wal_buffers = '16MB'
   ```

### High Availability

1. **Multiple Replicas**
   ```yaml
   spec:
     replicas: 3
   ```

2. **Pod Disruption Budget**
   ```yaml
   apiVersion: policy/v1
   kind: PodDisruptionBudget
   metadata:
     name: benchmark-pdb
   spec:
     minAvailable: 2
     selector:
       matchLabels:
         app: benchmark-app
   ```

## Monitoring and Observability

### Metrics Collection

The application exposes metrics at `/metrics` endpoint compatible with Prometheus.

Key metrics:
- Request rate and latency
- Error rates
- Cache hit ratios
- Resource utilization
- Business metrics (evaluations, agents, etc.)

### Dashboards

Pre-configured Grafana dashboards are included:

1. **Application Overview**
   - Request rates and response times
   - Error rates and status codes
   - Resource utilization

2. **Business Metrics**
   - Benchmark executions
   - Agent performance
   - Task completion rates

3. **Infrastructure**
   - Pod resource usage
   - Database performance
   - Cache statistics

## Security Guidelines

### Container Security

1. **Non-root User**
   ```dockerfile
   RUN groupadd -r benchmark && useradd -r -g benchmark benchmark
   USER benchmark
   ```

2. **Minimal Base Image**
   ```dockerfile
   FROM python:3.11-slim
   ```

3. **Security Scanning**
   ```bash
   # Scan images for vulnerabilities
   docker scan embodied-ai-benchmark:latest
   ```

### Network Security

1. **Network Policies**
   ```yaml
   apiVersion: networking.k8s.io/v1
   kind: NetworkPolicy
   metadata:
     name: benchmark-network-policy
   spec:
     podSelector:
       matchLabels:
         app: benchmark-app
     policyTypes:
     - Ingress
     - Egress
   ```

2. **TLS Encryption**
   ```yaml
   spec:
     tls:
     - hosts:
       - benchmark.example.com
       secretName: benchmark-tls
   ```

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   ```bash
   # Check memory usage
   kubectl top pods -n embodied-ai-benchmark
   
   # Increase memory limits
   kubectl patch deployment benchmark-app -p '{"spec":{"template":{"spec":{"containers":[{"name":"benchmark-app","resources":{"limits":{"memory":"4Gi"}}}]}}}}'
   ```

2. **Database Connection Issues**
   ```bash
   # Check database connectivity
   kubectl exec -it deployment/benchmark-app -- python -c "import psycopg2; print('DB connection OK')"
   
   # Check database logs
   kubectl logs deployment/postgres -n embodied-ai-benchmark
   ```

### Health Checks

The application provides health check endpoints:

- `/health` - Liveness probe
- `/ready` - Readiness probe  
- `/metrics` - Prometheus metrics

For additional support, please refer to:
- [Project Documentation](https://github.com/your-org/embodied-ai-benchmark/docs)
- [Issue Tracker](https://github.com/your-org/embodied-ai-benchmark/issues)
- [Community Forums](https://github.com/your-org/embodied-ai-benchmark/discussions)