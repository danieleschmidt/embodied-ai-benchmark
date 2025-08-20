#!/bin/bash
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
