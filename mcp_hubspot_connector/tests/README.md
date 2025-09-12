# MCP HubSpot Connector Testing

This directory contains pytest-based tests for the MCP HubSpot Connector, following the same patterns established in the `data_v2_engine` project.

## Test Structure

```
mcp_hubspot_connector/
└── mcp_hubspot_connector/
    ├── tests/
    │   ├── test_mcp_hubspot_connector.py
    │   ├── pytest.ini
    │   ├── requirements-test.txt
    │   ├── README-testing.md
    │   └── __pycache__/
    ├── __init__.py
    └── mcp_hubspot_connector.py
```

## Setup

### Option 1: Using pyproject.toml (Recommended)
```bash
# Install with test dependencies
pip install -e ".[test]"

# Or install all dependencies including development tools
pip install -e ".[all]"
```

### Option 2: Direct requirements installation
```bash
# Install test dependencies only
pip install -r mcp_hubspot_connector/tests/requirements.txt

# Install main dependencies
pip install hubspot-api-client>=12.0.0 python-dotenv>=0.19.0
```

### Environment Configuration
```bash
# Create .env file in project root (optional for basic tests)
# Add HubSpot access token if testing with real API
hubspot_access_token=your_token_here
```

## Running Tests

### From Project Root (Recommended)
```bash
# Run all tests
python -m pytest

# Run with verbose output
python -m pytest -v

# Run specific test file
python -m pytest mcp_hubspot_connector/tests/test_mcp_hubspot_connector.py -v

# Run specific test function
python -m pytest mcp_hubspot_connector/tests/test_mcp_hubspot_connector.py::test_predict_lead_scores_py -v
```

### From Tests Directory (Alternative)
```bash
cd mcp_hubspot_connector/tests/
python -m pytest

# Run specific test
python -m pytest test_mcp_hubspot_connector.py::test_ping_py -v

# Run parametrized tests
python -m pytest test_mcp_hubspot_connector.py::test_get_operations_parametrized_py -v
```

### Advanced Options
```bash
# Run with coverage
python -m pytest --cov=mcp_hubspot_connector --cov-report=html

# Run in parallel (requires pytest-xdist)
python -m pytest -n auto

# Run tests by markers
python -m pytest -m "not slow"
python -m pytest -m integration
python -m pytest -m unit

# Run with detailed logging
python -m pytest -v --tb=long --log-cli-level=INFO

# Skip tests with specific marker
python -m pytest -m "not integration"
```

### Test Script (Direct Execution)
```bash
# Run test file directly with custom environment
python mcp_hubspot_connector/tests/test_mcp_hubspot_connector.py --env hubspot_access_token=your_token

# Run with pytest arguments
python mcp_hubspot_connector/tests/test_mcp_hubspot_connector.py -v --tb=short
```

## Test Categories

### 1. Unit Tests (Default)
- Use mocks when HubSpot SDK is not available
- Test connector structure and method signatures
- Validate error handling and edge cases
- Fast execution, no external dependencies

### 2. Integration Tests (Marked with `@pytest.mark.skip`)
- Require valid HubSpot access token
- Test against real HubSpot API
- Enable by removing `@pytest.mark.skip` decorator

### 3. Analytics Tests
- `test_get_contact_analytics_py`: Contact analytics and segmentation
- `test_analyze_campaign_performance_py`: Campaign performance analysis  
- `test_analyze_sales_pipeline_py`: Sales pipeline analytics
- `test_predict_lead_scores_py`: Lead scoring predictions (currently enabled)

## Test Features

### Mock Support
Tests automatically detect if the HubSpot SDK is available:
- **With SDK**: Tests run against real connector (but may fail with invalid token)
- **Without SDK**: Tests use comprehensive mock connector with predefined responses

The mock connector provides realistic responses for:
- Basic CRUD operations (contacts, deals, companies)
- Search and filtering operations
- Analytics functions with structured data
- Error simulation and edge cases

### Parametrized Testing
Following `data_v2_engine` patterns:
- `test_get_operations_parametrized_py`: Tests all GET operations (contacts, deals, companies, events)
- `test_create_contact_validation_py`: Tests contact creation with various property combinations
- `test_create_deal_validation_py`: Tests deal creation validation scenarios

### Async Support
- Uses `@pytest.mark.asyncio` for async method testing
- Includes helper function `_call_async_method()` for consistent async testing
- Supports analytics functions that return complex data structures

### Error Handling Tests
- Missing access token validation
- Invalid method arguments handling
- SDK availability detection and graceful fallback
- Structured error reporting with timing metrics

## Configuration

### Environment Variables
```bash
# Required for real API testing
hubspot_access_token=your_token_here

# Optional
region_name=us-east-1
app_env=test
base_dir=/path/to/project
```

### Pytest Configuration
Configuration is available in both `pytest.ini` and `pyproject.toml`:

**Test Discovery:**
- `python_files = test_*.py`
- `python_classes = Test*`
- `python_functions = test_*`
- `testpaths = mcp_hubspot_connector/tests`

**Markers:**
- `asyncio`: marks tests as async (requires pytest-asyncio)  
- `slow`: marks tests as slow (deselect with '-m "not slow"')
- `integration`: marks tests as integration tests requiring real API
- `unit`: marks tests as unit tests with mocks
- `parametrize`: marks parametrized tests

**Options:**
- `--strict-markers`: Ensures all markers are defined
- `--strict-config`: Strict configuration validation
- `--verbose`: Detailed test output
- `--tb=short`: Short traceback format
- `--durations=10`: Show 10 slowest tests

**Logging:**
- Console logging enabled with INFO level
- Structured format with timestamps
- Filters for deprecation warnings

## Test Pattern Examples

### Basic Method Test
```python
def test_ping_py(connector):
    if getattr(connector, "__is_real__", False):
        # Real connector testing
        result, error = _call_method(connector, "ping")
        assert result is not None or error is not None
    else:
        # Mock connector testing
        result, error = _call_method(connector, "ping")
        assert "Mock HubSpot" in result
```

### Parametrized Test
```python
@pytest.mark.parametrize("method,args,keys", [
    ("get_contacts", {"limit": 5}, ["total", "contacts"]),
    ("get_deals", {"limit": 3}, ["total", "deals"]),
])
def test_operations(connector, method, args, keys):
    result, error = _call_method(connector, method, args)
    if result:
        for key in keys:
            assert key in result
```

### Async Test
```python
@pytest.mark.asyncio
async def test_predict_lead_scores_py(connector):
    now = pendulum.now("UTC")
    three_months_ago = now.subtract(months=3)
    
    result, error = await _call_async_method(
        connector,
        "predict_lead_scores",
        {
            "params": {
                "model_type": "conversion_probability",
                "date_range": {
                    "start": three_months_ago.to_iso8601_string(),
                    "end": now.to_iso8601_string(),
                },
            }
        },
        "predict_lead_scores"
    )
    assert result is not None
    assert result["success"] is True
```

### Validation Test
```python
@pytest.mark.parametrize("properties,should_succeed", [
    ({"email": "valid@example.com", "firstname": "John"}, True),
    ({"firstname": "John", "lastname": "Doe"}, False),  # Missing email
])
def test_create_contact_validation_py(connector, properties, should_succeed):
    result, error = _call_method(connector, "create_contact", {"properties": properties})
    if should_succeed:
        assert result is not None
    else:
        assert error is not None
```

## Logging

Tests use structured logging similar to `data_v2_engine`:
- Request/response timing
- Correlation IDs for tracing
- Structured log format with timestamps

## Test Execution Status

### Currently Enabled Tests
- `test_predict_lead_scores_py`: Lead scoring analytics (async)

### Skipped Tests (Enable by removing `@pytest.mark.skip`)
All other tests are currently skipped with the decorator:
- Basic connectivity tests (`test_ping_py`)
- Contact management (`test_get_contacts_py`, `test_create_contact_py`, etc.)
- Deal management (`test_get_deals_py`, `test_create_deal_py`)  
- Company management (`test_get_companies_py`)
- Marketing events (`test_get_marketing_events_py`)
- Parametrized tests (`test_get_operations_parametrized_py`)
- Validation tests (`test_create_contact_validation_py`, `test_create_deal_validation_py`)
- Analytics tests (`test_get_contact_analytics_py`, `test_analyze_campaign_performance_py`, `test_analyze_sales_pipeline_py`)

### Development Dependencies

Additional tools available via `pyproject.toml`:
```bash
# Install development tools
pip install -e ".[dev]"

# Includes:
# - black>=22.0.0 (code formatting)  
# - flake8>=5.0.0 (linting)
# - mypy>=1.0.0 (type checking)
# - pre-commit>=2.20.0 (git hooks)
```

### Coverage Reporting
```bash
# Generate HTML coverage report
python -m pytest --cov=mcp_hubspot_connector --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS/Linux
start htmlcov/index.html  # Windows
```

## Continuous Integration

Tests are designed to work in CI environments:
- Graceful handling of missing dependencies
- Mock fallbacks for external services  
- Configurable via environment variables
- Fast execution with parallel support (`-n auto`)
- Comprehensive markers for test selection

## Extending Tests

To add new tests:
1. Follow the naming convention: `test_*_py`
2. Use the `_call_method`/`_call_async_method` helpers for consistency
3. Support both real and mock connectors with `__is_real__` detection
4. Add parametrized variants for comprehensive coverage
5. Include appropriate assertions for both success and failure cases
6. Add async support with `@pytest.mark.asyncio` when needed
7. Use structured logging for debugging and metrics
8. Consider adding markers (`@pytest.mark.slow`, `@pytest.mark.integration`)