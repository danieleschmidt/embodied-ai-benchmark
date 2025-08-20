#!/bin/bash

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
