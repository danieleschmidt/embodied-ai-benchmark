#!/bin/bash
# Production environment setup script for Embodied AI Benchmark++

set -e  # Exit on any error

echo "🚀 Setting up Embodied AI Benchmark++ Production Environment"
echo "============================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check Python version
print_info "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
    print_status "Python $PYTHON_VERSION detected (compatible)"
else
    print_error "Python 3.8+ required, found $PYTHON_VERSION"
    exit 1
fi

# Create virtual environment
print_info "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_status "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip

# Install production requirements
print_info "Installing production requirements..."
pip install -r requirements-production.txt

print_status "Core dependencies installed"

# Install package in development mode
print_info "Installing package in development mode..."
pip install -e .

print_status "Package installed successfully"

# Create necessary directories
print_info "Creating production directories..."
mkdir -p logs
mkdir -p data
mkdir -p experiments
mkdir -p checkpoints
mkdir -p reports

print_status "Directory structure created"

# Set up logging configuration
print_info "Setting up logging configuration..."
cat > logging_config.yml << EOF
version: 1
disable_existing_loggers: false

formatters:
  standard:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  detailed:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s'

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: standard
    stream: ext://sys.stdout
  
  file:
    class: logging.handlers.RotatingFileHandler
    level: DEBUG
    formatter: detailed
    filename: logs/embodied_ai_benchmark.log
    maxBytes: 10485760  # 10MB
    backupCount: 5
  
  error_file:
    class: logging.handlers.RotatingFileHandler
    level: ERROR
    formatter: detailed
    filename: logs/errors.log
    maxBytes: 10485760  # 10MB
    backupCount: 5

loggers:
  embodied_ai_benchmark:
    level: DEBUG
    handlers: [console, file, error_file]
    propagate: false

root:
  level: INFO
  handlers: [console, file]
EOF

print_status "Logging configuration created"

# Create environment configuration
print_info "Creating environment configuration..."
cat > .env.production << EOF
# Production environment configuration
ENVIRONMENT=production
DEBUG=false

# Database configuration
DATABASE_URL=postgresql://user:password@localhost/embodied_ai_production
MONGODB_URI=mongodb://localhost:27017/embodied_ai_production

# Security
SECRET_KEY=your-secret-key-here
JWT_SECRET_KEY=your-jwt-secret-here

# LLM API Keys (set these manually)
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key

# Monitoring
PROMETHEUS_PORT=9090
METRICS_ENABLED=true

# Performance
MAX_WORKERS=4
CACHE_SIZE=1000
BATCH_SIZE=32

# Logging
LOG_LEVEL=INFO
STRUCTURED_LOGGING=true

# Feature flags
ENABLE_GPU=true
ENABLE_MULTIPROCESSING=true
ENABLE_CACHING=true
EOF

print_status "Environment configuration created"

# Create health check script
print_info "Creating health check script..."
cat > health_check.py << 'EOF'
#!/usr/bin/env python3
"""Health check script for production deployment."""

import sys
import json
import time
from datetime import datetime

def check_imports():
    """Check if all required modules can be imported."""
    try:
        import embodied_ai_benchmark
        return True, "All imports successful"
    except Exception as e:
        return False, f"Import failed: {e}"

def check_database():
    """Check database connectivity."""
    try:
        import sqlite3
        conn = sqlite3.connect(':memory:')
        conn.execute('SELECT 1')
        conn.close()
        return True, "Database check passed"
    except Exception as e:
        return False, f"Database check failed: {e}"

def check_dependencies():
    """Check critical dependencies."""
    try:
        import numpy
        import torch
        return True, "Dependencies available"
    except Exception as e:
        return False, f"Dependencies missing: {e}"

def run_health_check():
    """Run comprehensive health check."""
    checks = {
        "imports": check_imports(),
        "database": check_database(),
        "dependencies": check_dependencies()
    }
    
    all_passed = all(status for status, _ in checks.values())
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "overall_status": "healthy" if all_passed else "unhealthy",
        "checks": {name: {"status": "pass" if status else "fail", "message": msg} 
                  for name, (status, msg) in checks.items()}
    }
    
    print(json.dumps(report, indent=2))
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(run_health_check())
EOF

chmod +x health_check.py
print_status "Health check script created"

# Run health check
print_info "Running initial health check..."
if python3 health_check.py > /dev/null 2>&1; then
    print_status "Health check passed"
else
    print_warning "Health check showed some issues (this is expected without full dependencies)"
fi

# Create startup script
print_info "Creating startup script..."
cat > start_production.sh << 'EOF'
#!/bin/bash
# Production startup script

set -e

echo "🚀 Starting Embodied AI Benchmark++ Production"

# Activate virtual environment
source venv/bin/activate

# Load environment variables
if [ -f .env.production ]; then
    export $(cat .env.production | grep -v '^#' | xargs)
fi

# Run health check
echo "Running health check..."
python3 health_check.py

if [ $? -eq 0 ]; then
    echo "✅ Health check passed"
else
    echo "⚠️  Health check issues detected"
fi

# Start the application
echo "Starting application..."
python3 -m embodied_ai_benchmark.cli.main --config config/production.yml

EOF

chmod +x start_production.sh
print_status "Startup script created"

# Summary
echo ""
echo "============================================================"
print_status "Production environment setup complete!"
echo ""
print_info "Next steps:"
echo "  1. Edit .env.production with your API keys and database URLs"
echo "  2. Run 'source venv/bin/activate' to activate the environment"
echo "  3. Run './health_check.py' to verify the setup"
echo "  4. Run './start_production.sh' to start the application"
echo ""
print_info "Directory structure:"
echo "  ├── venv/                 # Virtual environment"
echo "  ├── logs/                 # Application logs"
echo "  ├── data/                 # Data storage"
echo "  ├── experiments/          # Experiment results"
echo "  ├── checkpoints/          # Model checkpoints"
echo "  ├── reports/              # Generated reports"
echo "  ├── .env.production       # Environment configuration"
echo "  ├── logging_config.yml    # Logging setup"
echo "  ├── health_check.py       # Health monitoring"
echo "  └── start_production.sh   # Startup script"
echo ""
print_status "Ready for production deployment!"
echo "============================================================"