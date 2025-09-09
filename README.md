# MCP HubSpot Connector

A Python connector for HubSpot CRM integration, providing clean and simple access to HubSpot's API for contact, deal, and company management.

## Features

- **Contact Management**: Create, read, update, and search contacts
- **Deal Management**: Create and retrieve deals  
- **Company Management**: Retrieve company information
- **Marketing Events**: Access marketing event data
- **Error Handling**: Graceful handling of API errors and SDK availability
- **Mock Support**: Built-in mock responses for development and testing

## Installation

### Basic Installation
```bash
pip install mcp_hubspot_connector
```

### Development Installation
```bash
git clone https://github.com/bibow/mcp_hubspot_connector.git
cd mcp_hubspot_connector
pip install -e .
```

### With Testing Dependencies
```bash
pip install mcp_hubspot_connector[test]
```

### With All Dependencies
```bash
pip install mcp_hubspot_connector[all]
```

## Quick Start

```python
import logging
from mcp_hubspot_connector import MCPHubspotConnector

# Setup logging
logger = logging.getLogger(__name__)

# Configure with HubSpot access token
settings = {
    "hubspot_access_token": "your_hubspot_access_token_here"
}

# Initialize connector
connector = MCPHubspotConnector(logger, **settings)

# Test connection
result = connector.ping()
print(result)

# Get contacts
contacts = connector.get_contacts(limit=10)
print(f"Found {contacts['total']} contacts")

# Create a new contact
new_contact = connector.create_contact(
    properties={
        "email": "example@company.com",
        "firstname": "John",
        "lastname": "Doe"
    }
)
print(f"Created contact with ID: {new_contact['id']}")
```

## API Methods

### Contact Management
- `get_contacts(limit=100, properties=[])` - Retrieve contacts
- `create_contact(properties={})` - Create new contact
- `update_contact(contact_id, properties={})` - Update existing contact
- `search_contacts(query="", limit=100)` - Search contacts
- `get_contact_by_email(email)` - Find contact by email

### Deal Management  
- `get_deals(limit=100, properties=[])` - Retrieve deals
- `create_deal(properties={})` - Create new deal

### Company Management
- `get_companies(limit=100, properties=[])` - Retrieve companies

### Marketing Events
- `get_marketing_events(limit=100)` - Retrieve marketing events

### Utility
- `ping()` - Test API connectivity

## Configuration

The connector requires a HubSpot access token. You can obtain one from your HubSpot developer account.

### Environment Variables
Create a `.env` file:
```
hubspot_access_token=your_token_here
```

### Direct Configuration
```python
settings = {
    "hubspot_access_token": "your_token_here"
}
connector = MCPHubspotConnector(logger, **settings)
```

## Testing

### Running Tests
```bash
# Install test dependencies
pip install -r mcp_hubspot_connector/tests/requirements-test.txt

# Run all tests (currently skipped by default)
cd mcp_hubspot_connector/tests
python -m pytest -v

# Run specific test
python -m pytest test_mcp_hubspot_connector.py::test_ping_py -v
```

### Mock vs Real Testing
The test suite automatically detects HubSpot SDK availability:
- **With valid token**: Tests against real HubSpot API
- **Without SDK/token**: Uses mock responses for development

### Enabling Tests
Tests are skipped by default. To enable specific tests, remove the `@pytest.mark.skip` decorator:
```python
# Remove this line to enable the test
# @pytest.mark.skip(reason="demonstrating skipping")
def test_ping_py(connector):
    # test code...
```

## Development

### Setup Development Environment
```bash
git clone https://github.com/bibow/mcp_hubspot_connector.git
cd mcp_hubspot_connector
pip install -e .[dev]
```

### Code Formatting
```bash
black mcp_hubspot_connector/
```

### Type Checking
```bash
mypy mcp_hubspot_connector/
```

### Linting
```bash
flake8 mcp_hubspot_connector/
```

## Requirements

- Python ≥ 3.8
- hubspot-api-client ≥ 12.0.0
- python-dotenv ≥ 0.19.0

## License

MIT License. See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Support

For issues and questions:
- GitHub Issues: https://github.com/bibow/mcp_hubspot_connector/issues
- Documentation: See tests/README-testing.md for detailed testing information