#!/bin/bash
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
