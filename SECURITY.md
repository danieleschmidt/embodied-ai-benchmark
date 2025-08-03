# Security Policy

## Supported Versions

We release patches for security vulnerabilities. Which versions are eligible
for receiving such patches depends on the CVSS v3.0 Rating:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

Please report (suspected) security vulnerabilities to
**[security@embodied-ai-benchmark.org]**. You will receive a response from
us within 48 hours. If the issue is confirmed, we will release a patch as soon
as possible depending on complexity but historically within a few days.

## Security Considerations

### Data Privacy
- No personally identifiable information is collected by default
- All experiment data remains local unless explicitly shared
- Optional telemetry can be disabled via configuration

### Code Execution
- Agent code execution is sandboxed within simulation environments
- No direct file system access beyond designated data directories
- Network access is limited to authorized endpoints only

### Input Validation
- All user inputs are validated and sanitized
- Action bounds are enforced to prevent simulator crashes
- Configuration files are validated against schemas

### Resource Limits
- Memory usage is monitored and limited
- CPU time limits prevent infinite loops
- GPU memory allocation is controlled

### Dependencies
- Regular security audits of dependencies
- Automated vulnerability scanning in CI/CD
- Pinned dependency versions for reproducibility

## Security Best Practices for Users

1. **Environment Isolation**: Run evaluations in isolated environments
2. **Code Review**: Review third-party agent implementations before use
3. **Resource Monitoring**: Monitor system resources during evaluation
4. **Network Security**: Use firewall rules for network-enabled simulations
5. **Data Backup**: Regularly backup experiment results and configurations

## Security Disclosure Process

1. **Report Reception**: Security reports are received and acknowledged
2. **Validation**: Issue is reproduced and impact assessed
3. **Fix Development**: Security patch is developed and tested
4. **Coordinated Disclosure**: Fix is released with security advisory
5. **Post-Incident Review**: Process improvements are implemented

## Known Security Considerations

- Simulation environments may consume significant system resources
- Multi-agent scenarios increase complexity and potential attack surface
- LLM integration requires careful prompt injection prevention
- External simulator dependencies may have their own security requirements

For security-related questions or concerns, please contact our security team.
