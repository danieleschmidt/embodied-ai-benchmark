#!/usr/bin/env python3
"""
Mock ValidationError for testing without full dependency imports.
This ensures Generation 2 robustness tests can run in environments without numpy.
"""

class ValidationError(Exception):
    """Mock validation error for testing."""
    pass


def test_mock_validation():
    """Test that mock validation error works correctly."""
    try:
        raise ValidationError("Test validation error")
    except ValidationError as e:
        print(f"✅ Mock ValidationError works: {e}")
        return True
    return False


if __name__ == "__main__":
    test_mock_validation()