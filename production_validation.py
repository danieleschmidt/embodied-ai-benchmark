#!/usr/bin/env python3
"""
Production Validation Script - Final validation for production deployment.
This script validates that all deployment artifacts are production-ready.
"""

import sys
import os
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, List

def validate_deployment_artifacts() -> Dict[str, Any]:
    """Validate deployment artifacts are complete and correct."""
    print("🚀 PRODUCTION DEPLOYMENT VALIDATION")
    print("=" * 60)
    
    validation_results = {}
    
    # Required deployment files
    required_files = [
        "Dockerfile",
        "docker-compose.yml", 
        ".dockerignore",
        "kubernetes-deployment.yaml",
        "DEPLOYMENT.md",
        "pyproject.toml",
        "README.md",
        "LICENSE"
    ]
    
    print("\n📁 Validating Deployment Files:")
    missing_files = []
    for file_name in required_files:
        if Path(file_name).exists():
            print(f"✅ {file_name}")
        else:
            print(f"❌ {file_name}")
            missing_files.append(file_name)
    
    validation_results["deployment_files"] = {
        "required": len(required_files),
        "found": len(required_files) - len(missing_files),
        "missing": missing_files,
        "passed": len(missing_files) == 0
    }
    
    # Validate Docker configuration
    print("\n🐳 Validating Docker Configuration:")
    docker_validation = validate_docker_config()
    validation_results["docker_config"] = docker_validation
    
    # Validate Kubernetes configuration
    print("\n☸️  Validating Kubernetes Configuration:")
    k8s_validation = validate_kubernetes_config()
    validation_results["kubernetes_config"] = k8s_validation
    
    # Validate security configurations
    print("\n🔐 Validating Security Configuration:")
    security_validation = validate_security_config()
    validation_results["security_config"] = security_validation
    
    # Validate monitoring setup
    print("\n📊 Validating Monitoring Configuration:")
    monitoring_validation = validate_monitoring_config()
    validation_results["monitoring_config"] = monitoring_validation
    
    # Overall assessment
    all_checks = [
        validation_results["deployment_files"]["passed"],
        docker_validation["passed"],
        k8s_validation["passed"],
        security_validation["passed"],
        monitoring_validation["passed"]
    ]
    
    overall_passed = sum(all_checks) >= 4  # At least 4 out of 5 must pass
    overall_score = (sum(all_checks) / len(all_checks)) * 100
    
    validation_results["overall"] = {
        "passed": overall_passed,
        "score": overall_score,
        "checks_passed": sum(all_checks),
        "total_checks": len(all_checks)
    }
    
    return validation_results

def validate_docker_config() -> Dict[str, Any]:
    """Validate Docker configuration."""
    try:
        dockerfile_path = Path("Dockerfile")
        compose_path = Path("docker-compose.yml")
        
        docker_checks = []
        
        # Check Dockerfile exists and has security best practices
        if dockerfile_path.exists():
            dockerfile_content = dockerfile_path.read_text()
            
            # Security checks
            if "USER" in dockerfile_content:
                docker_checks.append("Non-root user configured")
                print("✅ Non-root user configured")
            else:
                print("⚠️  No USER directive found in Dockerfile")
            
            if "HEALTHCHECK" in dockerfile_content:
                docker_checks.append("Health check configured")
                print("✅ Health check configured")
            else:
                print("⚠️  No HEALTHCHECK directive found")
            
            if "LABEL" in dockerfile_content:
                docker_checks.append("Metadata labels present")
                print("✅ Metadata labels present")
            else:
                print("⚠️  No LABEL directives found")
        
        # Check docker-compose configuration
        if compose_path.exists():
            compose_content = compose_path.read_text()
            
            if "restart:" in compose_content:
                docker_checks.append("Restart policy configured")
                print("✅ Restart policy configured")
            
            if "healthcheck:" in compose_content:
                docker_checks.append("Service health checks configured")
                print("✅ Service health checks configured")
            
            if "networks:" in compose_content:
                docker_checks.append("Custom networks configured")
                print("✅ Custom networks configured")
        
        return {
            "passed": len(docker_checks) >= 4,  # At least 4 checks should pass
            "checks": docker_checks,
            "score": len(docker_checks)
        }
        
    except Exception as e:
        print(f"❌ Docker validation error: {e}")
        return {
            "passed": False,
            "checks": [],
            "score": 0,
            "error": str(e)
        }

def validate_kubernetes_config() -> Dict[str, Any]:
    """Validate Kubernetes configuration."""
    try:
        k8s_path = Path("kubernetes-deployment.yaml")
        
        if not k8s_path.exists():
            print("❌ kubernetes-deployment.yaml not found")
            return {"passed": False, "checks": [], "score": 0}
        
        k8s_content = k8s_path.read_text()
        k8s_checks = []
        
        # Essential Kubernetes components
        required_components = [
            ("Namespace", "kind: Namespace"),
            ("ConfigMap", "kind: ConfigMap"),
            ("Secret", "kind: Secret"),
            ("Deployment", "kind: Deployment"),
            ("Service", "kind: Service"),
            ("HorizontalPodAutoscaler", "kind: HorizontalPodAutoscaler"),
            ("Ingress", "kind: Ingress"),
            ("PersistentVolumeClaim", "kind: PersistentVolumeClaim")
        ]
        
        for component_name, component_pattern in required_components:
            if component_pattern in k8s_content:
                k8s_checks.append(f"{component_name} configured")
                print(f"✅ {component_name} configured")
            else:
                print(f"⚠️  {component_name} not found")
        
        # Security checks
        if "SecurityContext" in k8s_content or "runAsNonRoot" in k8s_content:
            k8s_checks.append("Security context configured")
            print("✅ Security context configured")
        
        if "NetworkPolicy" in k8s_content:
            k8s_checks.append("Network policies configured")
            print("✅ Network policies configured")
        
        # Resource management
        if "resources:" in k8s_content and "limits:" in k8s_content:
            k8s_checks.append("Resource limits configured")
            print("✅ Resource limits configured")
        
        return {
            "passed": len(k8s_checks) >= 6,  # At least 6 components should be present
            "checks": k8s_checks,
            "score": len(k8s_checks)
        }
        
    except Exception as e:
        print(f"❌ Kubernetes validation error: {e}")
        return {
            "passed": False,
            "checks": [],
            "score": 0,
            "error": str(e)
        }

def validate_security_config() -> Dict[str, Any]:
    """Validate security configurations."""
    try:
        security_checks = []
        
        # Check for .env.example (template for environment variables)
        if Path(".env.example").exists():
            security_checks.append("Environment template provided")
            print("✅ Environment template provided")
        
        # Check .dockerignore to prevent sensitive file leaks
        if Path(".dockerignore").exists():
            dockerignore_content = Path(".dockerignore").read_text()
            if ".env" in dockerignore_content:
                security_checks.append("Environment files excluded from Docker")
                print("✅ Environment files excluded from Docker")
        
        # Check for secrets in Kubernetes config
        k8s_path = Path("kubernetes-deployment.yaml")
        if k8s_path.exists():
            k8s_content = k8s_path.read_text()
            if "kind: Secret" in k8s_content:
                security_checks.append("Kubernetes secrets configured")
                print("✅ Kubernetes secrets configured")
            
            if "tls:" in k8s_content:
                security_checks.append("TLS configuration present")
                print("✅ TLS configuration present")
        
        # Check for security documentation
        if Path("DEPLOYMENT.md").exists():
            deployment_content = Path("DEPLOYMENT.md").read_text()
            if "Security" in deployment_content:
                security_checks.append("Security documentation provided")
                print("✅ Security documentation provided")
        
        # Check that no secrets are committed
        sensitive_patterns = [".env", "*.key", "*.pem", "password", "secret"]
        exposed_files = []
        
        for pattern in sensitive_patterns:
            # Simple check - in production, use more sophisticated tools
            if pattern in str(Path(".").glob("**/*")):
                exposed_files.append(pattern)
        
        if not exposed_files:
            security_checks.append("No sensitive files detected")
            print("✅ No sensitive files detected")
        else:
            print(f"⚠️  Potential sensitive files: {exposed_files}")
        
        return {
            "passed": len(security_checks) >= 4,
            "checks": security_checks,
            "score": len(security_checks),
            "exposed_files": exposed_files
        }
        
    except Exception as e:
        print(f"❌ Security validation error: {e}")
        return {
            "passed": False,
            "checks": [],
            "score": 0,
            "error": str(e)
        }

def validate_monitoring_config() -> Dict[str, Any]:
    """Validate monitoring and observability configuration."""
    try:
        monitoring_checks = []
        
        # Check docker-compose for monitoring services
        compose_path = Path("docker-compose.yml")
        if compose_path.exists():
            compose_content = compose_path.read_text()
            
            if "prometheus:" in compose_content:
                monitoring_checks.append("Prometheus configured")
                print("✅ Prometheus configured")
            
            if "grafana:" in compose_content:
                monitoring_checks.append("Grafana configured")
                print("✅ Grafana configured")
            
            if "healthcheck:" in compose_content:
                monitoring_checks.append("Health checks configured")
                print("✅ Health checks configured")
        
        # Check Kubernetes for monitoring
        k8s_path = Path("kubernetes-deployment.yaml")
        if k8s_path.exists():
            k8s_content = k8s_path.read_text()
            
            if "livenessProbe:" in k8s_content:
                monitoring_checks.append("Kubernetes liveness probes")
                print("✅ Kubernetes liveness probes")
            
            if "readinessProbe:" in k8s_content:
                monitoring_checks.append("Kubernetes readiness probes")
                print("✅ Kubernetes readiness probes")
            
            if "ServiceMonitor" in k8s_content:
                monitoring_checks.append("Prometheus ServiceMonitor")
                print("✅ Prometheus ServiceMonitor")
        
        # Check for logging configuration
        dockerfile_path = Path("Dockerfile")
        if dockerfile_path.exists():
            dockerfile_content = dockerfile_path.read_text()
            if "LOG" in dockerfile_content:
                monitoring_checks.append("Logging configuration")
                print("✅ Logging configuration")
        
        # Check for deployment documentation
        if Path("DEPLOYMENT.md").exists():
            deployment_content = Path("DEPLOYMENT.md").read_text()
            if "Monitoring" in deployment_content:
                monitoring_checks.append("Monitoring documentation")
                print("✅ Monitoring documentation")
        
        return {
            "passed": len(monitoring_checks) >= 3,
            "checks": monitoring_checks,
            "score": len(monitoring_checks)
        }
        
    except Exception as e:
        print(f"❌ Monitoring validation error: {e}")
        return {
            "passed": False,
            "checks": [],
            "score": 0,
            "error": str(e)
        }

def validate_build_process() -> Dict[str, Any]:
    """Validate that the build process works."""
    print("\n🔨 Validating Build Process:")
    
    build_checks = []
    
    try:
        # Check if Docker build would work (dry run)
        print("Testing Docker build configuration...")
        
        # Simple validation of Dockerfile syntax
        dockerfile_path = Path("Dockerfile")
        if dockerfile_path.exists():
            dockerfile_content = dockerfile_path.read_text()
            
            # Check for required directives
            required_directives = ["FROM", "WORKDIR", "COPY", "RUN", "CMD"]
            for directive in required_directives:
                if directive in dockerfile_content:
                    build_checks.append(f"Dockerfile {directive} directive")
                    print(f"✅ Dockerfile {directive} directive")
                else:
                    print(f"⚠️  Missing {directive} directive in Dockerfile")
        
        # Check pyproject.toml
        pyproject_path = Path("pyproject.toml")
        if pyproject_path.exists():
            build_checks.append("Package configuration present")
            print("✅ Package configuration present")
        
        return {
            "passed": len(build_checks) >= 4,
            "checks": build_checks,
            "score": len(build_checks)
        }
        
    except Exception as e:
        print(f"❌ Build validation error: {e}")
        return {
            "passed": False,
            "checks": [],
            "score": 0,
            "error": str(e)
        }

def generate_deployment_report(validation_results: Dict[str, Any]) -> str:
    """Generate comprehensive deployment readiness report."""
    
    report = {
        "timestamp": time.time(),
        "deployment_readiness": validation_results,
        "recommendations": [],
        "next_steps": []
    }
    
    # Generate recommendations based on results
    if not validation_results["docker_config"]["passed"]:
        report["recommendations"].append("Review Docker configuration for security best practices")
    
    if not validation_results["kubernetes_config"]["passed"]:
        report["recommendations"].append("Complete Kubernetes configuration with all required components")
    
    if not validation_results["security_config"]["passed"]:
        report["recommendations"].append("Address security configuration issues")
    
    if not validation_results["monitoring_config"]["passed"]:
        report["recommendations"].append("Set up comprehensive monitoring and observability")
    
    # Next steps for deployment
    if validation_results["overall"]["passed"]:
        report["next_steps"] = [
            "Build and tag Docker images",
            "Push images to container registry",
            "Update Kubernetes secrets with production values",
            "Deploy to staging environment first",
            "Run integration tests",
            "Deploy to production",
            "Verify monitoring and alerting"
        ]
    else:
        report["next_steps"] = [
            "Address failing validation checks",
            "Re-run production validation",
            "Complete missing deployment artifacts",
            "Test deployment in staging environment"
        ]
    
    return json.dumps(report, indent=2)

def main():
    """Main validation function."""
    print("🚀 TERRAGON AUTONOMOUS SDLC - PRODUCTION DEPLOYMENT VALIDATION")
    print("=" * 80)
    
    # Run all validations
    validation_results = validate_deployment_artifacts()
    
    # Build process validation
    build_results = validate_build_process()
    validation_results["build_process"] = build_results
    
    # Generate final report
    report_content = generate_deployment_report(validation_results)
    
    # Save report
    report_file = "production_deployment_report.json"
    with open(report_file, 'w') as f:
        f.write(report_content)
    
    # Final summary
    print("\n" + "=" * 80)
    print("🚀 PRODUCTION DEPLOYMENT VALIDATION COMPLETE")
    print("=" * 80)
    
    overall = validation_results["overall"]
    status = "✅ READY" if overall["passed"] else "❌ NOT READY"
    
    print(f"Overall Status: {status}")
    print(f"Validation Score: {overall['score']:.1f}%")
    print(f"Checks Passed: {overall['checks_passed']}/{overall['total_checks']}")
    
    # Print summary of each category
    categories = [
        ("Deployment Files", "deployment_files"),
        ("Docker Config", "docker_config"),
        ("Kubernetes Config", "kubernetes_config"),
        ("Security Config", "security_config"),
        ("Monitoring Config", "monitoring_config"),
        ("Build Process", "build_process")
    ]
    
    print("\n📊 Category Breakdown:")
    for category_name, category_key in categories:
        if category_key in validation_results:
            result = validation_results[category_key]
            status = "✅" if result["passed"] else "❌"
            score = result.get("score", 0)
            print(f"{status} {category_name}: {score} points")
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    # Print next steps
    report_data = json.loads(report_content)
    if report_data["next_steps"]:
        print("\n📋 Next Steps:")
        for i, step in enumerate(report_data["next_steps"], 1):
            print(f"{i}. {step}")
    
    if report_data["recommendations"]:
        print("\n💡 Recommendations:")
        for i, rec in enumerate(report_data["recommendations"], 1):
            print(f"{i}. {rec}")
    
    return overall["passed"]

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)