#!/bin/bash

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
