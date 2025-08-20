#!/usr/bin/env python3
"""Production deployment preparation for Embodied AI Benchmark++."""

import sys
import os
import json
import yaml
from datetime import datetime
from typing import Dict, Any, List
import subprocess
sys.path.insert(0, 'src')

class ProductionDeploymentPreparator:
    """Comprehensive production deployment preparation."""
    
    def __init__(self):
        self.deployment_config = {
            "timestamp": datetime.now().isoformat(),
            "framework_version": "1.0.0",
            "deployment_targets": {},
            "infrastructure_configs": {},
            "monitoring_setup": {},
            "security_configs": {},
            "backup_strategies": {},
            "deployment_ready": False
        }
        
    def create_docker_configuration(self) -> Dict[str, Any]:
        """Create Docker deployment configuration."""
        print("🐳 Creating Docker Configuration...")
        
        results = {"dockerfile_created": False, "docker_compose_created": False, "optimized_build": False}
        
        try:
            # Production Dockerfile
            dockerfile_content = '''# Production Dockerfile for Embodied AI Benchmark++
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

# Create app directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    libblas-dev \\
    liblapack-dev \\
    pkg-config \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for layer caching)
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# Copy source code
COPY src/ src/
COPY tests/ tests/
COPY docs/ docs/
COPY README.md LICENSE ./

# Create non-root user
RUN groupadd -r embodied_ai && useradd -r -g embodied_ai embodied_ai
RUN chown -R embodied_ai:embodied_ai /app
USER embodied_ai

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD python -c "import embodied_ai_benchmark; print('healthy')" || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "-m", "embodied_ai_benchmark.api.app"]
'''
            
            with open('Dockerfile.prod', 'w') as f:
                f.write(dockerfile_content)
            results["dockerfile_created"] = True
            print("  ✅ Production Dockerfile created")
            
            # Docker Compose for production
            docker_compose_content = {
                'version': '3.8',
                'services': {
                    'embodied-ai-benchmark': {
                        'build': {
                            'context': '.',
                            'dockerfile': 'Dockerfile.prod'
                        },
                        'ports': ['8000:8000'],
                        'environment': [
                            'ENVIRONMENT=production',
                            'LOG_LEVEL=info',
                            'WORKERS=4'
                        ],
                        'volumes': [
                            'benchmark_data:/app/data',
                            'benchmark_logs:/app/logs'
                        ],
                        'healthcheck': {
                            'test': ['CMD', 'python', '-c', 'import embodied_ai_benchmark'],
                            'interval': '30s',
                            'timeout': '10s',
                            'retries': 3
                        },
                        'restart': 'unless-stopped',
                        'deploy': {
                            'replicas': 3,
                            'resources': {
                                'limits': {'cpus': '2', 'memory': '4G'},
                                'reservations': {'cpus': '1', 'memory': '2G'}
                            }
                        }
                    },
                    'redis': {
                        'image': 'redis:7-alpine',
                        'ports': ['6379:6379'],
                        'volumes': ['redis_data:/data'],
                        'restart': 'unless-stopped'
                    },
                    'postgres': {
                        'image': 'postgres:15-alpine',
                        'environment': [
                            'POSTGRES_DB=embodied_ai_benchmark',
                            'POSTGRES_USER=benchmark_user',
                            'POSTGRES_PASSWORD_FILE=/run/secrets/db_password'
                        ],
                        'volumes': [
                            'postgres_data:/var/lib/postgresql/data',
                            'postgres_backup:/backup'
                        ],
                        'secrets': ['db_password'],
                        'restart': 'unless-stopped'
                    },
                    'nginx': {
                        'image': 'nginx:alpine',
                        'ports': ['80:80', '443:443'],
                        'volumes': [
                            './nginx.conf:/etc/nginx/nginx.conf:ro',
                            'ssl_certs:/etc/ssl/certs'
                        ],
                        'depends_on': ['embodied-ai-benchmark'],
                        'restart': 'unless-stopped'
                    }
                },
                'volumes': {
                    'benchmark_data': None,
                    'benchmark_logs': None,
                    'redis_data': None,
                    'postgres_data': None,
                    'postgres_backup': None,
                    'ssl_certs': None
                },
                'secrets': {
                    'db_password': {
                        'file': './secrets/db_password.txt'
                    }
                },
                'networks': {
                    'benchmark_network': {
                        'driver': 'bridge'
                    }
                }
            }
            
            with open('docker-compose.prod.yml', 'w') as f:
                yaml.dump(docker_compose_content, f, default_flow_style=False)
            results["docker_compose_created"] = True
            print("  ✅ Production Docker Compose created")
            
            # Multi-stage optimized Dockerfile
            optimized_dockerfile = '''# Multi-stage optimized Dockerfile
FROM python:3.11-slim as builder
WORKDIR /build
COPY pyproject.toml .
RUN pip install --user --no-cache-dir -e .

FROM python:3.11-slim as production
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
WORKDIR /app
COPY src/ src/
COPY --from=builder /build .
HEALTHCHECK --interval=30s CMD python -c "import embodied_ai_benchmark"
CMD ["python", "-m", "embodied_ai_benchmark.api.app"]
'''
            
            with open('Dockerfile.optimized', 'w') as f:
                f.write(optimized_dockerfile)
            results["optimized_build"] = True
            print("  ✅ Optimized multi-stage Dockerfile created")
            
        except Exception as e:
            print(f"  ❌ Docker configuration failed: {e}")
            
        return results
    
    def create_kubernetes_configuration(self) -> Dict[str, Any]:
        """Create Kubernetes deployment configuration."""
        print("☸️  Creating Kubernetes Configuration...")
        
        results = {"deployment_created": False, "service_created": False, "ingress_created": False, "monitoring_created": False}
        
        try:
            # Kubernetes deployment
            k8s_deployment = {
                'apiVersion': 'apps/v1',
                'kind': 'Deployment',
                'metadata': {
                    'name': 'embodied-ai-benchmark',
                    'labels': {'app': 'embodied-ai-benchmark'}
                },
                'spec': {
                    'replicas': 3,
                    'selector': {'matchLabels': {'app': 'embodied-ai-benchmark'}},
                    'template': {
                        'metadata': {'labels': {'app': 'embodied-ai-benchmark'}},
                        'spec': {
                            'containers': [{
                                'name': 'embodied-ai-benchmark',
                                'image': 'embodied-ai-benchmark:latest',
                                'ports': [{'containerPort': 8000}],
                                'env': [
                                    {'name': 'ENVIRONMENT', 'value': 'production'},
                                    {'name': 'LOG_LEVEL', 'value': 'info'}
                                ],
                                'resources': {
                                    'requests': {'cpu': '500m', 'memory': '1Gi'},
                                    'limits': {'cpu': '2', 'memory': '4Gi'}
                                },
                                'livenessProbe': {
                                    'httpGet': {'path': '/health', 'port': 8000},
                                    'initialDelaySeconds': 30,
                                    'periodSeconds': 10
                                },
                                'readinessProbe': {
                                    'httpGet': {'path': '/ready', 'port': 8000},
                                    'initialDelaySeconds': 5,
                                    'periodSeconds': 5
                                },
                                'volumeMounts': [
                                    {'name': 'benchmark-data', 'mountPath': '/app/data'},
                                    {'name': 'benchmark-logs', 'mountPath': '/app/logs'}
                                ]
                            }],
                            'volumes': [
                                {'name': 'benchmark-data', 'persistentVolumeClaim': {'claimName': 'benchmark-data-pvc'}},
                                {'name': 'benchmark-logs', 'persistentVolumeClaim': {'claimName': 'benchmark-logs-pvc'}}
                            ]
                        }
                    }
                }
            }
            
            with open('kubernetes-deployment.yaml', 'w') as f:
                yaml.dump(k8s_deployment, f)
            results["deployment_created"] = True
            print("  ✅ Kubernetes deployment configuration created")
            
            # Kubernetes service
            k8s_service = {
                'apiVersion': 'v1',
                'kind': 'Service',
                'metadata': {'name': 'embodied-ai-benchmark-service'},
                'spec': {
                    'selector': {'app': 'embodied-ai-benchmark'},
                    'ports': [{'port': 80, 'targetPort': 8000}],
                    'type': 'ClusterIP'
                }
            }
            
            # Kubernetes ingress
            k8s_ingress = {
                'apiVersion': 'networking.k8s.io/v1',
                'kind': 'Ingress',
                'metadata': {
                    'name': 'embodied-ai-benchmark-ingress',
                    'annotations': {
                        'kubernetes.io/ingress.class': 'nginx',
                        'cert-manager.io/cluster-issuer': 'letsencrypt-prod',
                        'nginx.ingress.kubernetes.io/rate-limit': '100'
                    }
                },
                'spec': {
                    'tls': [{'hosts': ['benchmark.example.com'], 'secretName': 'benchmark-tls'}],
                    'rules': [{
                        'host': 'benchmark.example.com',
                        'http': {
                            'paths': [{
                                'path': '/',
                                'pathType': 'Prefix',
                                'backend': {
                                    'service': {
                                        'name': 'embodied-ai-benchmark-service',
                                        'port': {'number': 80}
                                    }
                                }
                            }]
                        }
                    }]
                }
            }
            
            with open('kubernetes-service-ingress.yaml', 'w') as f:
                yaml.dump_all([k8s_service, k8s_ingress], f)
            results["service_created"] = True
            results["ingress_created"] = True
            print("  ✅ Kubernetes service and ingress created")
            
            # Monitoring configuration
            monitoring_config = {
                'apiVersion': 'v1',
                'kind': 'ConfigMap',
                'metadata': {'name': 'prometheus-config'},
                'data': {
                    'prometheus.yml': '''
global:
  scrape_interval: 15s
scrape_configs:
  - job_name: 'embodied-ai-benchmark'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        action: keep
        regex: embodied-ai-benchmark
'''
                }
            }
            
            with open('kubernetes-monitoring.yaml', 'w') as f:
                yaml.dump(monitoring_config, f)
            results["monitoring_created"] = True
            print("  ✅ Kubernetes monitoring configuration created")
            
        except Exception as e:
            print(f"  ❌ Kubernetes configuration failed: {e}")
            
        return results
    
    def create_cloud_configurations(self) -> Dict[str, Any]:
        """Create cloud provider configurations."""
        print("☁️  Creating Cloud Configurations...")
        
        results = {"aws_config": False, "gcp_config": False, "azure_config": False}
        
        try:
            # AWS ECS Task Definition
            aws_ecs_config = {
                'family': 'embodied-ai-benchmark',
                'networkMode': 'awsvpc',
                'requiresCompatibilities': ['FARGATE'],
                'cpu': '1024',
                'memory': '2048',
                'executionRoleArn': 'arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole',
                'taskRoleArn': 'arn:aws:iam::ACCOUNT:role/ecsTaskRole',
                'containerDefinitions': [{
                    'name': 'embodied-ai-benchmark',
                    'image': 'ACCOUNT.dkr.ecr.REGION.amazonaws.com/embodied-ai-benchmark:latest',
                    'portMappings': [{'containerPort': 8000, 'protocol': 'tcp'}],
                    'environment': [
                        {'name': 'ENVIRONMENT', 'value': 'production'},
                        {'name': 'AWS_REGION', 'value': 'us-west-2'}
                    ],
                    'logConfiguration': {
                        'logDriver': 'awslogs',
                        'options': {
                            'awslogs-group': '/ecs/embodied-ai-benchmark',
                            'awslogs-region': 'us-west-2',
                            'awslogs-stream-prefix': 'ecs'
                        }
                    },
                    'healthCheck': {
                        'command': ['CMD-SHELL', 'python -c "import embodied_ai_benchmark"'],
                        'interval': 30,
                        'timeout': 5,
                        'retries': 3
                    }
                }]
            }
            
            with open('aws-ecs-task-definition.json', 'w') as f:
                json.dump(aws_ecs_config, f, indent=2)
            results["aws_config"] = True
            print("  ✅ AWS ECS configuration created")
            
            # Google Cloud Run configuration
            gcp_config = {
                'apiVersion': 'serving.knative.dev/v1',
                'kind': 'Service',
                'metadata': {
                    'name': 'embodied-ai-benchmark',
                    'annotations': {
                        'run.googleapis.com/ingress': 'all',
                        'run.googleapis.com/execution-environment': 'gen2'
                    }
                },
                'spec': {
                    'template': {
                        'metadata': {
                            'annotations': {
                                'autoscaling.knative.dev/maxScale': '10',
                                'run.googleapis.com/cpu-throttling': 'false',
                                'run.googleapis.com/memory': '4Gi',
                                'run.googleapis.com/cpu': '2'
                            }
                        },
                        'spec': {
                            'containerConcurrency': 100,
                            'containers': [{
                                'image': 'gcr.io/PROJECT_ID/embodied-ai-benchmark:latest',
                                'ports': [{'containerPort': 8000}],
                                'env': [
                                    {'name': 'ENVIRONMENT', 'value': 'production'},
                                    {'name': 'GCP_PROJECT', 'value': 'PROJECT_ID'}
                                ],
                                'resources': {
                                    'limits': {'cpu': '2', 'memory': '4Gi'}
                                }
                            }]
                        }
                    },
                    'traffic': [{'percent': 100, 'latestRevision': True}]
                }
            }
            
            with open('gcp-cloud-run.yaml', 'w') as f:
                yaml.dump(gcp_config, f)
            results["gcp_config"] = True
            print("  ✅ GCP Cloud Run configuration created")
            
            # Azure Container Instances configuration
            azure_config = {
                '$schema': 'https://schema.management.azure.com/schemas/2019-12-01/deploymentTemplate.json#',
                'contentVersion': '1.0.0.0',
                'parameters': {
                    'containerName': {'type': 'string', 'defaultValue': 'embodied-ai-benchmark'},
                    'image': {'type': 'string', 'defaultValue': 'embodiedai.azurecr.io/benchmark:latest'}
                },
                'resources': [{
                    'type': 'Microsoft.ContainerInstance/containerGroups',
                    'apiVersion': '2021-03-01',
                    'name': '[parameters("containerName")]',
                    'location': '[resourceGroup().location]',
                    'properties': {
                        'containers': [{
                            'name': 'embodied-ai-benchmark',
                            'properties': {
                                'image': '[parameters("image")]',
                                'ports': [{'port': 8000}],
                                'environmentVariables': [
                                    {'name': 'ENVIRONMENT', 'value': 'production'}
                                ],
                                'resources': {
                                    'requests': {'cpu': 2, 'memoryInGB': 4}
                                }
                            }
                        }],
                        'osType': 'Linux',
                        'ipAddress': {
                            'type': 'Public',
                            'ports': [{'port': 8000, 'protocol': 'TCP'}]
                        }
                    }
                }]
            }
            
            with open('azure-container-instances.json', 'w') as f:
                json.dump(azure_config, f, indent=2)
            results["azure_config"] = True
            print("  ✅ Azure Container Instances configuration created")
            
        except Exception as e:
            print(f"  ❌ Cloud configuration failed: {e}")
            
        return results
    
    def create_monitoring_setup(self) -> Dict[str, Any]:
        """Create comprehensive monitoring setup."""
        print("📊 Creating Monitoring Setup...")
        
        results = {"prometheus_config": False, "grafana_config": False, "alerting_config": False, "logging_config": False}
        
        try:
            # Prometheus configuration
            prometheus_config = {
                'global': {
                    'scrape_interval': '15s',
                    'evaluation_interval': '15s'
                },
                'rule_files': ['alert_rules.yml'],
                'alerting': {
                    'alertmanagers': [{
                        'static_configs': [{'targets': ['alertmanager:9093']}]
                    }]
                },
                'scrape_configs': [
                    {
                        'job_name': 'embodied-ai-benchmark',
                        'static_configs': [{'targets': ['localhost:8000']}],
                        'metrics_path': '/metrics',
                        'scrape_interval': '30s'
                    },
                    {
                        'job_name': 'node-exporter',
                        'static_configs': [{'targets': ['node-exporter:9100']}]
                    }
                ]
            }
            
            with open('prometheus.yml', 'w') as f:
                yaml.dump(prometheus_config, f)
            results["prometheus_config"] = True
            print("  ✅ Prometheus configuration created")
            
            # Grafana dashboard configuration
            grafana_dashboard = {
                'dashboard': {
                    'id': None,
                    'title': 'Embodied AI Benchmark++ Monitoring',
                    'tags': ['embodied-ai', 'benchmark', 'research'],
                    'timezone': 'browser',
                    'panels': [
                        {
                            'id': 1,
                            'title': 'Request Rate',
                            'type': 'graph',
                            'targets': [{
                                'expr': 'rate(http_requests_total[5m])',
                                'legendFormat': 'Requests/sec'
                            }]
                        },
                        {
                            'id': 2,
                            'title': 'Response Time',
                            'type': 'graph',
                            'targets': [{
                                'expr': 'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))',
                                'legendFormat': '95th percentile'
                            }]
                        },
                        {
                            'id': 3,
                            'title': 'Error Rate',
                            'type': 'singlestat',
                            'targets': [{
                                'expr': 'rate(http_requests_total{status=~"5.."}[5m])',
                                'legendFormat': 'Error Rate'
                            }]
                        }
                    ],
                    'time': {'from': 'now-1h', 'to': 'now'},
                    'refresh': '30s'
                }
            }
            
            with open('grafana-dashboard.json', 'w') as f:
                json.dump(grafana_dashboard, f, indent=2)
            results["grafana_config"] = True
            print("  ✅ Grafana dashboard configuration created")
            
            # Alert rules
            alert_rules = {
                'groups': [{
                    'name': 'embodied-ai-benchmark.rules',
                    'rules': [
                        {
                            'alert': 'HighErrorRate',
                            'expr': 'rate(http_requests_total{status=~"5.."}[5m]) > 0.1',
                            'for': '5m',
                            'labels': {'severity': 'critical'},
                            'annotations': {
                                'summary': 'High error rate detected',
                                'description': 'Error rate is {{ $value }} errors per second'
                            }
                        },
                        {
                            'alert': 'HighResponseTime',
                            'expr': 'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1.0',
                            'for': '10m',
                            'labels': {'severity': 'warning'},
                            'annotations': {
                                'summary': 'High response time detected',
                                'description': '95th percentile response time is {{ $value }}s'
                            }
                        }
                    ]
                }]
            }
            
            with open('alert_rules.yml', 'w') as f:
                yaml.dump(alert_rules, f)
            results["alerting_config"] = True
            print("  ✅ Alerting configuration created")
            
            # Logging configuration
            logging_config = {
                'version': 1,
                'disable_existing_loggers': False,
                'formatters': {
                    'standard': {
                        'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
                    },
                    'json': {
                        'class': 'pythonjsonlogger.jsonlogger.JsonFormatter',
                        'format': '%(asctime)s %(name)s %(levelname)s %(message)s'
                    }
                },
                'handlers': {
                    'default': {
                        'level': 'INFO',
                        'formatter': 'json',
                        'class': 'logging.StreamHandler'
                    },
                    'file': {
                        'level': 'INFO',
                        'formatter': 'json',
                        'class': 'logging.handlers.RotatingFileHandler',
                        'filename': '/app/logs/benchmark.log',
                        'maxBytes': 10485760,
                        'backupCount': 5
                    }
                },
                'loggers': {
                    'embodied_ai_benchmark': {
                        'handlers': ['default', 'file'],
                        'level': 'INFO',
                        'propagate': False
                    }
                },
                'root': {
                    'level': 'INFO',
                    'handlers': ['default']
                }
            }
            
            with open('logging.yaml', 'w') as f:
                yaml.dump(logging_config, f)
            results["logging_config"] = True
            print("  ✅ Logging configuration created")
            
        except Exception as e:
            print(f"  ❌ Monitoring setup failed: {e}")
            
        return results
    
    def create_deployment_scripts(self) -> Dict[str, Any]:
        """Create deployment automation scripts."""
        print("🚀 Creating Deployment Scripts...")
        
        results = {"deploy_script": False, "rollback_script": False, "health_check": False, "backup_script": False}
        
        try:
            # Main deployment script
            deploy_script = '''#!/bin/bash
set -e

echo "🚀 Starting Embodied AI Benchmark++ Deployment"

# Configuration
ENVIRONMENT=${ENVIRONMENT:-production}
VERSION=${VERSION:-latest}
NAMESPACE=${NAMESPACE:-default}

echo "📋 Deployment Configuration:"
echo "  Environment: $ENVIRONMENT"
echo "  Version: $VERSION"
echo "  Namespace: $NAMESPACE"

# Pre-deployment checks
echo "🔍 Running pre-deployment checks..."
python3 final_quality_gates_validation.py
if [ $? -ne 0 ]; then
    echo "❌ Quality gates failed. Aborting deployment."
    exit 1
fi

# Build and push Docker image
echo "🐳 Building Docker image..."
docker build -f Dockerfile.prod -t embodied-ai-benchmark:$VERSION .
docker tag embodied-ai-benchmark:$VERSION embodied-ai-benchmark:latest

# Deploy to Kubernetes
echo "☸️  Deploying to Kubernetes..."
kubectl apply -f kubernetes-deployment.yaml -n $NAMESPACE
kubectl apply -f kubernetes-service-ingress.yaml -n $NAMESPACE

# Wait for rollout
echo "⏳ Waiting for deployment rollout..."
kubectl rollout status deployment/embodied-ai-benchmark -n $NAMESPACE

# Run health checks
echo "🏥 Running health checks..."
./health_check.sh

# Deploy monitoring
echo "📊 Deploying monitoring..."
kubectl apply -f kubernetes-monitoring.yaml -n $NAMESPACE

echo "✅ Deployment completed successfully!"
echo "🌐 Service available at: https://benchmark.example.com"
'''
            
            with open('deploy.sh', 'w') as f:
                f.write(deploy_script)
            os.chmod('deploy.sh', 0o755)
            results["deploy_script"] = True
            print("  ✅ Main deployment script created")
            
            # Rollback script
            rollback_script = '''#!/bin/bash
set -e

echo "🔄 Starting Embodied AI Benchmark++ Rollback"

NAMESPACE=${NAMESPACE:-default}
REVISION=${REVISION:-1}

echo "📋 Rolling back to revision: $REVISION"

# Rollback deployment
kubectl rollout undo deployment/embodied-ai-benchmark --to-revision=$REVISION -n $NAMESPACE

# Wait for rollback
kubectl rollout status deployment/embodied-ai-benchmark -n $NAMESPACE

# Verify health
./health_check.sh

echo "✅ Rollback completed successfully!"
'''
            
            with open('rollback.sh', 'w') as f:
                f.write(rollback_script)
            os.chmod('rollback.sh', 0o755)
            results["rollback_script"] = True
            print("  ✅ Rollback script created")
            
            # Health check script
            health_check_script = '''#!/bin/bash

echo "🏥 Running Health Checks..."

NAMESPACE=${NAMESPACE:-default}
SERVICE_URL=${SERVICE_URL:-http://localhost:8000}

# Check deployment status
READY_REPLICAS=$(kubectl get deployment embodied-ai-benchmark -n $NAMESPACE -o jsonpath='{.status.readyReplicas}')
DESIRED_REPLICAS=$(kubectl get deployment embodied-ai-benchmark -n $NAMESPACE -o jsonpath='{.spec.replicas}')

if [ "$READY_REPLICAS" != "$DESIRED_REPLICAS" ]; then
    echo "❌ Deployment not ready: $READY_REPLICAS/$DESIRED_REPLICAS replicas ready"
    exit 1
fi

# Check service endpoint
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" $SERVICE_URL/health)
if [ "$HTTP_CODE" != "200" ]; then
    echo "❌ Health check failed: HTTP $HTTP_CODE"
    exit 1
fi

# Check API functionality
API_RESPONSE=$(curl -s $SERVICE_URL/api/v1/tasks)
if ! echo "$API_RESPONSE" | grep -q "tasks"; then
    echo "❌ API functionality check failed"
    exit 1
fi

echo "✅ All health checks passed!"
echo "📊 Ready replicas: $READY_REPLICAS/$DESIRED_REPLICAS"
echo "🌐 Service endpoint: $SERVICE_URL"
'''
            
            with open('health_check.sh', 'w') as f:
                f.write(health_check_script)
            os.chmod('health_check.sh', 0o755)
            results["health_check"] = True
            print("  ✅ Health check script created")
            
            # Backup script
            backup_script = '''#!/bin/bash

echo "💾 Starting Backup Process..."

BACKUP_DIR="/backup/$(date +%Y-%m-%d-%H-%M-%S)"
NAMESPACE=${NAMESPACE:-default}

mkdir -p $BACKUP_DIR

# Backup Kubernetes configurations
echo "☸️  Backing up Kubernetes configurations..."
kubectl get all -n $NAMESPACE -o yaml > $BACKUP_DIR/kubernetes-resources.yaml
kubectl get configmaps -n $NAMESPACE -o yaml > $BACKUP_DIR/configmaps.yaml
kubectl get secrets -n $NAMESPACE -o yaml > $BACKUP_DIR/secrets.yaml

# Backup persistent volumes
echo "💿 Backing up persistent volumes..."
kubectl get pv -o yaml > $BACKUP_DIR/persistent-volumes.yaml

# Backup database (if applicable)
if kubectl get deployment postgres -n $NAMESPACE > /dev/null 2>&1; then
    echo "🗄️  Backing up database..."
    kubectl exec -n $NAMESPACE deployment/postgres -- pg_dump -U benchmark_user embodied_ai_benchmark > $BACKUP_DIR/database.sql
fi

# Create backup archive
cd /backup
tar -czf "backup-$(date +%Y-%m-%d-%H-%M-%S).tar.gz" $(basename $BACKUP_DIR)
rm -rf $BACKUP_DIR

echo "✅ Backup completed: backup-$(date +%Y-%m-%d-%H-%M-%S).tar.gz"
'''
            
            with open('backup.sh', 'w') as f:
                f.write(backup_script)
            os.chmod('backup.sh', 0o755)
            results["backup_script"] = True
            print("  ✅ Backup script created")
            
        except Exception as e:
            print(f"  ❌ Deployment scripts creation failed: {e}")
            
        return results
    
    def create_security_configurations(self) -> Dict[str, Any]:
        """Create security configurations."""
        print("🔒 Creating Security Configurations...")
        
        results = {"network_policies": False, "rbac_config": False, "secrets_config": False, "security_scanning": False}
        
        try:
            # Network policies
            network_policy = {
                'apiVersion': 'networking.k8s.io/v1',
                'kind': 'NetworkPolicy',
                'metadata': {'name': 'embodied-ai-benchmark-network-policy'},
                'spec': {
                    'podSelector': {'matchLabels': {'app': 'embodied-ai-benchmark'}},
                    'policyTypes': ['Ingress', 'Egress'],
                    'ingress': [{
                        'from': [{'podSelector': {'matchLabels': {'app': 'nginx'}}}],
                        'ports': [{'protocol': 'TCP', 'port': 8000}]
                    }],
                    'egress': [{
                        'to': [{'podSelector': {'matchLabels': {'app': 'postgres'}}}],
                        'ports': [{'protocol': 'TCP', 'port': 5432}]
                    }]
                }
            }
            
            with open('network-policy.yaml', 'w') as f:
                yaml.dump(network_policy, f)
            results["network_policies"] = True
            print("  ✅ Network policies created")
            
            # RBAC configuration
            rbac_config = [{
                'apiVersion': 'v1',
                'kind': 'ServiceAccount',
                'metadata': {'name': 'embodied-ai-benchmark-sa'}
            }, {
                'apiVersion': 'rbac.authorization.k8s.io/v1',
                'kind': 'Role',
                'metadata': {'name': 'embodied-ai-benchmark-role'},
                'rules': [{
                    'apiGroups': [''],
                    'resources': ['pods', 'services'],
                    'verbs': ['get', 'list', 'watch']
                }]
            }, {
                'apiVersion': 'rbac.authorization.k8s.io/v1',
                'kind': 'RoleBinding',
                'metadata': {'name': 'embodied-ai-benchmark-binding'},
                'subjects': [{
                    'kind': 'ServiceAccount',
                    'name': 'embodied-ai-benchmark-sa'
                }],
                'roleRef': {
                    'kind': 'Role',
                    'name': 'embodied-ai-benchmark-role',
                    'apiGroup': 'rbac.authorization.k8s.io'
                }
            }]
            
            with open('rbac.yaml', 'w') as f:
                yaml.dump_all(rbac_config, f)
            results["rbac_config"] = True
            print("  ✅ RBAC configuration created")
            
            # Secrets configuration
            secrets_template = {
                'apiVersion': 'v1',
                'kind': 'Secret',
                'metadata': {'name': 'embodied-ai-benchmark-secrets'},
                'type': 'Opaque',
                'data': {
                    'database-password': '<base64-encoded-password>',
                    'api-key': '<base64-encoded-api-key>',
                    'jwt-secret': '<base64-encoded-jwt-secret>'
                }
            }
            
            with open('secrets-template.yaml', 'w') as f:
                yaml.dump(secrets_template, f)
            results["secrets_config"] = True
            print("  ✅ Secrets configuration template created")
            
            # Security scanning configuration
            security_scan_config = {
                'image_scanning': {
                    'enabled': True,
                    'scanners': ['trivy', 'clair'],
                    'fail_on': ['HIGH', 'CRITICAL'],
                    'schedule': '0 2 * * *'
                },
                'runtime_security': {
                    'enabled': True,
                    'tools': ['falco'],
                    'rules': ['suspicious_activity', 'privilege_escalation']
                },
                'compliance_scanning': {
                    'enabled': True,
                    'frameworks': ['CIS', 'NIST'],
                    'schedule': '0 3 * * 0'
                }
            }
            
            with open('security-scan-config.yaml', 'w') as f:
                yaml.dump(security_scan_config, f)
            results["security_scanning"] = True
            print("  ✅ Security scanning configuration created")
            
        except Exception as e:
            print(f"  ❌ Security configuration failed: {e}")
            
        return results
    
    def run_deployment_preparation(self) -> Dict[str, Any]:
        """Run comprehensive deployment preparation."""
        print("🚀 EMBODIED AI BENCHMARK++ DEPLOYMENT PREPARATION")
        print("⚙️  Creating Production-Ready Deployment Configurations")
        print("=" * 80)
        
        # Run all preparation steps
        docker_results = self.create_docker_configuration()
        k8s_results = self.create_kubernetes_configuration()
        cloud_results = self.create_cloud_configurations()
        monitoring_results = self.create_monitoring_setup()
        scripts_results = self.create_deployment_scripts()
        security_results = self.create_security_configurations()
        
        # Compile all results
        all_results = {
            "docker_configuration": docker_results,
            "kubernetes_configuration": k8s_results,
            "cloud_configurations": cloud_results,
            "monitoring_setup": monitoring_results,
            "deployment_scripts": scripts_results,
            "security_configurations": security_results
        }
        
        # Calculate deployment readiness score
        total_configs = sum(len(results) for results in all_results.values())
        successful_configs = sum(
            sum(1 for success in results.values() if success) 
            for results in all_results.values()
        )
        
        deployment_score = (successful_configs / total_configs) * 100 if total_configs > 0 else 0
        deployment_ready = deployment_score >= 80.0
        
        # Update results
        self.deployment_config.update({
            "deployment_targets": all_results,
            "deployment_score": deployment_score,
            "deployment_ready": deployment_ready
        })
        
        # Print detailed summary
        print("\n" + "=" * 80)
        print("📊 DEPLOYMENT PREPARATION SUMMARY")
        print("=" * 80)
        
        for category, results in all_results.items():
            successful = sum(1 for r in results.values() if r)
            total = len(results)
            print(f"{category.replace('_', ' ').title()}: {successful}/{total} configurations ready")
        
        print(f"\n🏆 Deployment Readiness Score: {deployment_score:.1f}%")
        print(f"🚀 Production Ready: {'YES' if deployment_ready else 'NO'}")
        
        if deployment_ready:
            print("\n🎉 DEPLOYMENT PREPARATION COMPLETE!")
            print("🌐 Framework ready for production deployment across multiple platforms")
            print("\n📋 Available Deployment Options:")
            print("   • Docker Compose (docker-compose.prod.yml)")
            print("   • Kubernetes (kubernetes-deployment.yaml)")
            print("   • AWS ECS (aws-ecs-task-definition.json)")
            print("   • GCP Cloud Run (gcp-cloud-run.yaml)")
            print("   • Azure Container Instances (azure-container-instances.json)")
            print("\n🛠️  Deployment Commands:")
            print("   • Local: docker-compose -f docker-compose.prod.yml up")
            print("   • K8s: ./deploy.sh")
            print("   • Health Check: ./health_check.sh")
        else:
            print("\n⚠️  Deployment preparation needs attention")
            print("🔧 Complete missing configurations before deployment")
        
        return self.deployment_config

def main():
    """Run deployment preparation."""
    preparator = ProductionDeploymentPreparator()
    results = preparator.run_deployment_preparation()
    
    # Save results
    with open("production_deployment_config.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Configuration saved to: production_deployment_config.json")
    print("🎯 EMBODIED AI BENCHMARK++ AUTONOMOUS SDLC COMPLETE!")
    
    return results["deployment_ready"]

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)