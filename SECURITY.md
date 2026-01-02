# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| latest  | Yes       |

## Reporting a Vulnerability

If you discover a security vulnerability in mojo-tokenizer:

1. **Do not** open a public issue
2. Email: security@atsentia.com
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

We will respond within 48 hours and work with you to address the issue.

## Security Considerations

### Input Validation

- Tokenizers process arbitrary text input
- Always validate input size before processing
- Be cautious with untrusted vocabulary files

### Vocabulary Files

- Only load vocabulary files from trusted sources
- Verify file integrity before loading
- Malicious vocabulary files could cause denial of service

### Memory Safety

- mojo-tokenizer uses Mojo's memory safety features
- Large inputs may consume significant memory
- Consider input size limits in production

## Best Practices

1. Keep mojo-tokenizer updated to the latest version
2. Validate and sanitize input text
3. Use trusted vocabulary files only
4. Monitor memory usage with large inputs
5. Run with appropriate resource limits in production
