# Contributing to mojo-tokenizer

Thank you for your interest in contributing!

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/mojo-tokenizer.git
   ```
3. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature
   ```

## Development Setup

### Prerequisites

- [Mojo](https://docs.modular.com/mojo/manual/get-started/) (24.6+)
- [pixi](https://pixi.sh/) (optional, for task running)

### Running Tests

```bash
mojo run tests/test_tokenizer.mojo
```

Or with pixi:

```bash
pixi run test
```

### Code Formatting

```bash
mojo format src/
```

Or with pixi:

```bash
pixi run format
```

## Code Style

- Run `mojo format` before committing
- Follow existing patterns in the codebase
- Add docstrings for public functions and structs
- Add tests for new functionality
- Keep functions focused and small

## Pull Request Process

1. Ensure all tests pass
2. Update README.md if adding features
3. Keep PRs focused and small
4. Write clear commit messages

### Commit Message Format

```
type: short description

Longer description if needed.
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Tests
- `refactor`: Code refactoring

## Testing Guidelines

- Add tests for all new functionality
- Test edge cases (empty input, special tokens, etc.)
- Use descriptive test function names: `test_feature_behavior()`
- Print pass/fail status for each test

## Documentation

- Update README.md for new features
- Add docstrings with examples
- Include type annotations

## Questions?

Open an issue or reach out to the maintainers.

## License

Contributions are licensed under Apache 2.0 with LLVM Exceptions.
