# Contributing to Atris

Thank you for your interest in contributing to Atris! We welcome contributions of all kinds, including bug reports, feature requests, code, and documentation.

## How to Contribute

1. **Fork the repository** and create your branch from `main`.
2. **Install dependencies** using `pip install -e .` and `pip install -r atris/requirements.txt`.
3. **Write clear, well-documented code** and add tests for new features or bug fixes.
4. **Run all tests** with `pytest tests` before submitting a pull request.
5. **Open a pull request** with a clear description of your changes and why they are needed.

## Code Style
- Follow [PEP8](https://www.python.org/dev/peps/pep-0008/) for Python code style.
- Use clear, descriptive variable and function names.
- Add docstrings to all public functions and classes.

## Adding New Models or Features
- Add new models to `atris/models.py` and expose them in the public API if appropriate.
- For new feature engineering transformers, add them to `atris/features.py`.
- For new ensemble logic, update `atris/ensemble.py` and add relevant tests.

## Reporting Issues
- Please use the GitHub Issues tab to report bugs or request features.
- Include as much detail as possible, including code snippets and error messages.

## License
By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for helping make Atris better! 