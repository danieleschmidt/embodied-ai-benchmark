#!/usr/bin/env python3
"""Test runner for the embodied AI benchmark with coverage reporting."""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Run comprehensive test suite with coverage reporting."""
    
    # Set up environment
    os.environ["PYTHONPATH"] = str(Path(__file__).parent / "src")
    
    # Test commands to run
    test_commands = [
        # Unit tests with coverage
        [
            "python3", "-m", "pytest", 
            "tests/unit/", 
            "-v", 
            "--cov=src/embodied_ai_benchmark",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            "--tb=short",
            "--maxfail=10"
        ],
        
        # Integration tests
        [
            "python3", "-m", "pytest",
            "tests/integration/",
            "-v",
            "--tb=short",
            "--maxfail=5",
            "-m", "not slow"  # Skip slow tests for quick CI
        ],
        
        # All tests summary
        [
            "python3", "-m", "pytest",
            "tests/",
            "--tb=short",
            "--maxfail=3",
            "-q",  # Quiet mode for summary
            "--cov=src/embodied_ai_benchmark",
            "--cov-fail-under=85"  # Require 85% coverage
        ]
    ]
    
    print("🧪 Running Embodied AI Benchmark Test Suite")
    print("=" * 60)
    
    for i, cmd in enumerate(test_commands, 1):
        print(f"\n📊 Running test phase {i}/{len(test_commands)}")
        print(f"Command: {' '.join(cmd)}")
        print("-" * 40)
        
        try:
            result = subprocess.run(cmd, check=False, cwd=Path(__file__).parent)
            
            if result.returncode != 0:
                print(f"❌ Test phase {i} failed with return code {result.returncode}")
                if i < len(test_commands):
                    print("⚠️  Continuing with next phase...")
                else:
                    print("🚫 Final test phase failed!")
                    return result.returncode
            else:
                print(f"✅ Test phase {i} passed!")
                
        except KeyboardInterrupt:
            print("\n🛑 Tests interrupted by user")
            return 130
        except Exception as e:
            print(f"❌ Error running test phase {i}: {e}")
            return 1
    
    print("\n🎉 All test phases completed!")
    print("\n📈 Coverage report generated in: htmlcov/index.html")
    print("💡 View with: firefox htmlcov/index.html")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())