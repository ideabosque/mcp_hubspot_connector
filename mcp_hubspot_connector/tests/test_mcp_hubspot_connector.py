#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

__author__ = "bibow"
# Pytest-based test suite for mcp_hubspot_connector

import json
import logging
import os
import sys
import time
import uuid
from typing import Any, Literal
from unittest.mock import patch

import pendulum
import pytest
from dotenv import load_dotenv

load_dotenv()


# Make package importable in common local setups
base_dir = os.getenv("base_dir", os.getcwd())
sys.path.insert(0, base_dir)
sys.path.insert(0, os.path.join(base_dir, "mcp_hubspot_connector"))

from mcp_hubspot_connector import HubSpotSDKConfig, MCPHubspotConnector

# Test settings for mcp_hubspot_connector
SETTING = {
    # HubSpot API settings
    "hubspot_access_token": os.getenv("hubspot_access_token", "test_token"),
}


logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("test_mcp_hubspot_connector")


def _call_method(connector, method_name: str, arguments: dict = None, label: str = None):
    """Helper function to call connector methods with logging"""
    arguments = arguments or {}
    op = label or method_name
    cid = uuid.uuid4().hex[:8]
    logger.info("Method call: cid=%s op=%s arguments=%s", cid, op, json.dumps(arguments))
    t0 = time.perf_counter()

    try:
        method = getattr(connector, method_name)
        result = method(**arguments)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(
            "Method response: cid=%s op=%s elapsed_ms=%s success=True result=%s",
            cid,
            op,
            elapsed_ms,
            json.dumps(result) if not hasattr(result, "__dict__") else str(result),
        )
        return result, None
    except Exception as e:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(
            "Method response: cid=%s op=%s elapsed_ms=%s success=False error=%s",
            cid,
            op,
            elapsed_ms,
            str(e),
        )
        return None, str(e)


async def _call_async_method(
    connector, method_name: str, arguments: dict = None, label: str = None
):
    """Helper function to call async connector methods with logging"""
    arguments = arguments or {}
    op = label or method_name
    cid = uuid.uuid4().hex[:8]
    logger.info("Async method call: cid=%s op=%s arguments=%s", cid, op, json.dumps(arguments))
    t0 = time.perf_counter()

    try:
        method = getattr(connector, method_name)
        result = await method(**arguments)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        # Log result appropriately based on type
        if isinstance(result, dict):
            result_str = json.dumps(result, default=str)
        elif hasattr(result, "__dict__"):
            result_str = str(result)
        else:
            result_str = json.dumps(result, default=str)

        logger.info(
            "Async method response: cid=%s op=%s elapsed_ms=%s success=True result=%s",
            cid,
            op,
            elapsed_ms,
            result_str,
        )
        return result, None
    except Exception as e:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(
            "Async method response: cid=%s op=%s elapsed_ms=%s success=False error=%s",
            cid,
            op,
            elapsed_ms,
            str(e),
        )
        return None, str(e)


@pytest.fixture(scope="module")
def connector():
    """Provide a MCPHubspotConnector instance or mock."""

    class _Mock:
        __is_real__ = False

        def __init__(self):
            # Mock all methods
            self.ping = lambda **_: "Mock HubSpot connection successful"
            self.get_contacts = lambda **_: {
                "total": 2,
                "contacts": [
                    {
                        "id": "1",
                        "properties": {
                            "email": "test1@example.com",
                            "firstname": "Test",
                            "lastname": "User1",
                        },
                    },
                    {
                        "id": "2",
                        "properties": {
                            "email": "test2@example.com",
                            "firstname": "Test",
                            "lastname": "User2",
                        },
                    },
                ],
            }
            self.create_contact = lambda **_: {
                "id": "123",
                "properties": {"email": "new@example.com"},
            }
            self.update_contact = lambda **_: {
                "id": "123",
                "properties": {"email": "updated@example.com"},
            }
            self.get_deals = lambda **_: {
                "total": 1,
                "deals": [
                    {"id": "deal1", "properties": {"dealname": "Test Deal", "amount": "1000"}}
                ],
            }
            self.create_deal = lambda **_: {"id": "newdeal", "properties": {"dealname": "New Deal"}}
            self.get_companies = lambda **_: {
                "total": 1,
                "companies": [
                    {"id": "comp1", "properties": {"name": "Test Company", "domain": "test.com"}}
                ],
            }
            self.search_contacts = lambda **_: {
                "total": 1,
                "contacts": [{"id": "1", "properties": {"email": "search@example.com"}}],
                "query": "search",
            }
            self.get_contact_by_email = lambda **_: {
                "id": "1",
                "properties": {"email": "test@example.com"},
            }
            self.get_marketing_events = lambda **_: {
                "total": 1,
                "events": [{"id": "event1", "name": "Test Event", "event_type": "webinar"}],
            }

            # Mock analytics functions (return dictionary objects)
            async def mock_get_contact_analytics(**_):
                return {
                    "success": True,
                    "data": {
                        "totalContacts": 100,
                        "activeContacts": 75,
                        "avgEngagementScore": 8.5,
                        "segments": {
                            "high_engagement": 25,
                            "medium_engagement": 50,
                            "low_engagement": 25,
                        },
                    },
                    "insights": ["Good engagement levels overall"],
                    "recommendations": ["Focus on converting high-engagement contacts"],
                    "metadata": {"source": "mock"},
                    "error_message": None,
                }

            self.get_contact_analytics = mock_get_contact_analytics

            async def mock_analyze_campaign_performance(**_):
                return {
                    "success": True,
                    "data": {
                        "campaigns": [{"campaign_id": "c1", "open_rate": 0.25, "click_rate": 0.05}],
                        "summary_metrics": {"avg_open_rate": 0.25, "avg_click_rate": 0.05},
                        "benchmarks": {"industry_open_rate": 0.20, "industry_click_rate": 0.025},
                    },
                    "insights": ["Open rates above industry average"],
                    "recommendations": ["Improve click-through rates"],
                    "metadata": {"source": "mock"},
                    "error_message": None,
                }

            self.analyze_campaign_performance = mock_analyze_campaign_performance

            async def mock_analyze_sales_pipeline(**_):
                return {
                    "success": True,
                    "data": {
                        "totalValue": 50000,
                        "dealCount": 10,
                        "avgDealSize": 5000,
                        "conversionRate": 0.25,
                    },
                    "insights": ["Healthy pipeline value"],
                    "recommendations": ["Focus on deal conversion"],
                    "metadata": {"source": "mock"},
                    "error_message": None,
                }

            self.analyze_sales_pipeline = mock_analyze_sales_pipeline

            async def mock_predict_lead_scores(**_):
                return {
                    "success": True,
                    "data": {
                        "predictions": [{"contact_id": "c1", "score": 0.85, "risk_level": "high"}],
                        "model_performance": {"accuracy": 0.92},
                        "feature_importance": [{"feature": "engagement_score", "importance": 0.6}],
                    },
                    "insights": ["High model accuracy"],
                    "recommendations": ["Prioritize high-scoring contacts"],
                    "metadata": {"source": "mock"},
                    "error_message": None,
                }

            self.predict_lead_scores = mock_predict_lead_scores

            async def mock_create_contact_segments(**_):
                return {
                    "success": True,
                    "data": {
                        "segments": [
                            {
                                "segment_id": "high_value",
                                "name": "High Value Customers",
                                "contact_count": 250,
                                "criteria": {
                                    "lifetime_value": {"min": 10000},
                                    "engagement_score": {"min": 8},
                                },
                                "contacts": [
                                    {"contact_id": "c1", "score": 9.2},
                                    {"contact_id": "c2", "score": 8.7},
                                ],
                            },
                            {
                                "segment_id": "potential_churners",
                                "name": "Potential Churners",
                                "contact_count": 75,
                                "criteria": {
                                    "last_activity_days": {"min": 90},
                                    "engagement_score": {"max": 3},
                                },
                                "contacts": [
                                    {"contact_id": "c3", "score": 2.1},
                                    {"contact_id": "c4", "score": 2.8},
                                ],
                            },
                        ],
                        "segmentation_quality": {
                            "total_contacts": 325,
                            "num_segments": 2,
                            "silhouette_score": 0.8,
                        },
                        "contact_assignments": [
                            {
                                "contact_id": "c1",
                                "segment": "High Value Customers",
                                "engagement_score": 9.2,
                            },
                            {
                                "contact_id": "c2",
                                "segment": "High Value Customers",
                                "engagement_score": 8.7,
                            },
                            {
                                "contact_id": "c3",
                                "segment": "Potential Churners",
                                "engagement_score": 2.1,
                            },
                            {
                                "contact_id": "c4",
                                "segment": "Potential Churners",
                                "engagement_score": 2.8,
                            },
                        ],
                        "segment_characteristics": [
                            {
                                "label": "High Value Customers",
                                "size": 250,
                                "percentage": 76.9,
                                "characteristics": {"avg_engagement": 8.95},
                            },
                            {
                                "label": "Potential Churners",
                                "size": 75,
                                "percentage": 23.1,
                                "characteristics": {"avg_engagement": 2.45},
                            },
                        ],
                    },
                    "insights": [
                        "High value segment shows strong engagement",
                        "Churn risk identified in 75 contacts",
                    ],
                    "recommendations": [
                        "Focus retention efforts on potential churners",
                        "Expand high-value customer programs",
                    ],
                    "metadata": {
                        "source": "mock",
                        "algorithm": "k-means",
                        "segmentation_type": "behavioral",
                        "num_segments": 2,
                        "total_contacts": 325,
                        "contact_limit": 1000,
                    },
                    "error_message": None,
                }

            self.create_contact_segments = mock_create_contact_segments

            async def mock_forecast_revenue(**_):
                return {
                    "success": True,
                    "data": {
                        "forecast": {
                            "total_revenue": 1250000,
                            "period": "90_days",
                            "breakdown": [
                                {"period": "month_1", "revenue": 400000, "probability": 0.85},
                                {"period": "month_2", "revenue": 425000, "probability": 0.82},
                                {"period": "month_3", "revenue": 425000, "probability": 0.80},
                            ],
                        },
                        "confidence_interval": {"lower": 1100000, "upper": 1400000},
                        "scenarios": {
                            "optimistic": 1400000,
                            "realistic": 1250000,
                            "pessimistic": 1100000,
                        },
                        "model_performance": {"accuracy": 0.87, "rmse": 45000, "mae": 32000},
                    },
                    "insights": [
                        "Strong pipeline indicates healthy revenue forecast",
                        "Model shows high confidence in predictions",
                    ],
                    "recommendations": [
                        "Focus on high-probability deals",
                        "Monitor risk factors closely",
                    ],
                    "metadata": {
                        "forecast_period": "90_days",
                        "confidence_level": 0.95,
                        "historical_data_points": 150,
                        "pipeline_deals": 25,
                        "data_source": "hubspot_sdk_revenue_forecasting",
                    },
                    "error_message": None,
                }

            self.forecast_revenue = mock_forecast_revenue

            async def mock_generate_executive_report(**_):
                return {
                    "success": True,
                    "data": {
                        "report_id": "exec_report_2024_01",
                        "generated_at": "2024-01-15T10:00:00Z",
                        "period": {"start": "2024-01-01", "end": "2024-01-31"},
                        "executive_summary": {
                            "total_contacts": 2500,
                            "new_contacts": 150,
                            "total_deals": 85,
                            "won_deals": 12,
                            "total_revenue": 485000,
                            "pipeline_value": 1250000,
                        },
                        "key_metrics": {
                            "contact_growth_rate": 0.06,
                            "deal_win_rate": 0.14,
                            "avg_deal_size": 40416,
                            "sales_cycle_days": 45,
                            "customer_acquisition_cost": 1200,
                        },
                        "performance_highlights": [
                            "6% growth in contact base",
                            "Pipeline value increased by 15%",
                            "Sales cycle reduced by 8 days",
                        ],
                        "areas_of_concern": [
                            "Deal win rate below target (20%)",
                            "Customer acquisition cost trending up",
                        ],
                        "departmental_breakdown": {
                            "sales": {
                                "deals_closed": 12,
                                "revenue": 485000,
                                "target_achievement": 0.97,
                            },
                            "marketing": {
                                "leads_generated": 150,
                                "cost_per_lead": 80,
                                "lead_quality_score": 7.2,
                            },
                        },
                        "forecasts": {
                            "next_month_revenue": 425000,
                            "quarter_end_pipeline": 1400000,
                            "expected_new_customers": 18,
                        },
                    },
                    "insights": [
                        "Strong overall growth trajectory",
                        "Sales efficiency improvements needed",
                        "Marketing ROI remains strong",
                    ],
                    "recommendations": [
                        "Implement deal acceleration strategies",
                        "Review and optimize acquisition channels",
                        "Focus on high-value customer segments",
                    ],
                    "metadata": {"source": "mock", "report_type": "executive"},
                    "error_message": None,
                }

            self.generate_executive_report = mock_generate_executive_report

    if MCPHubspotConnector is None:
        return _Mock()

    try:
        # Try with new HubSpotSDKConfig if available
        if HubSpotSDKConfig:
            config = HubSpotSDKConfig(access_token=SETTING["hubspot_access_token"], debug_mode=True)
            c = MCPHubspotConnector(logger, config=config)
        else:
            # Fallback to old settings dict
            c = MCPHubspotConnector(logger, **SETTING)

        setattr(c, "__is_real__", True)
        return c

    except Exception as ex:
        logger.warning(f"MCPHubspotConnector failed: {ex}")
        return _Mock()


# Basic connectivity and initialization tests
@pytest.mark.skip(reason="demonstrating skipping")
def test_ping_py(connector: Any):
    """Test ping functionality for API connectivity"""
    if getattr(connector, "__is_real__", False):
        result, error = _call_method(connector, "ping", label="ping")
        if error:
            pytest.fail(f"ping failed with error: {error}")

        assert result is not None, "Result should not be None when no error occurs"
        assert "HubSpot" in result, "Result should contain 'HubSpot'"
    else:
        result, error = _call_method(connector, "ping", label="ping")
        assert result is not None
        assert "Mock HubSpot" in result


# Contact management tests
@pytest.mark.skip(reason="demonstrating skipping")
def test_get_contacts_py(connector: Any):
    """Test getting contacts from HubSpot"""
    arguments = {"limit": 10, "properties": ["email", "firstname", "lastname"]}

    if getattr(connector, "__is_real__", False):
        result, error = _call_method(connector, "get_contacts", arguments, label="get_contacts")
        # Real connector might fail due to SDK issues, but should handle gracefully
        if result:
            assert isinstance(result, dict)
            assert "total" in result
            assert "contacts" in result
            assert isinstance(result["contacts"], list)
        else:
            # If error, should be due to SDK not being available
            assert error is not None
    else:
        result, error = _call_method(connector, "get_contacts", arguments, label="get_contacts")
        assert result is not None
        assert result["total"] == 2
        assert len(result["contacts"]) == 2
        assert result["contacts"][0]["properties"]["email"] == "test1@example.com"


@pytest.mark.skip(reason="demonstrating skipping")
def test_create_contact_py(connector: Any):
    """Test creating a new contact in HubSpot"""
    arguments = {
        "properties": {"email": "newcontact@example.com", "firstname": "New", "lastname": "Contact"}
    }

    if getattr(connector, "__is_real__", False):
        result, error = _call_method(connector, "create_contact", arguments, label="create_contact")
        if result:
            assert isinstance(result, dict)
            assert "id" in result
            assert "properties" in result
        else:
            # Should fail gracefully
            assert error is not None
    else:
        result, error = _call_method(connector, "create_contact", arguments, label="create_contact")
        assert result is not None
        assert result["id"] == "123"
        assert result["properties"]["email"] == "new@example.com"


@pytest.mark.skip(reason="demonstrating skipping")
def test_update_contact_py(connector: Any):
    """Test updating an existing contact in HubSpot"""
    arguments = {"contact_id": "123", "properties": {"firstname": "Updated", "lastname": "Name"}}

    if getattr(connector, "__is_real__", False):
        result, error = _call_method(connector, "update_contact", arguments, label="update_contact")
        if result:
            assert isinstance(result, dict)
            assert "id" in result
            assert "properties" in result
        else:
            assert error is not None
    else:
        result, error = _call_method(connector, "update_contact", arguments, label="update_contact")
        assert result is not None
        assert result["id"] == "123"


@pytest.mark.skip(reason="demonstrating skipping")
def test_search_contacts_py(connector: Any):
    """Test searching contacts in HubSpot"""
    arguments = {
        "query": "test@example.com",
        "limit": 5,
        "properties": ["email", "firstname", "lastname"],
    }

    if getattr(connector, "__is_real__", False):
        result, error = _call_method(
            connector, "search_contacts", arguments, label="search_contacts"
        )
        if result:
            assert isinstance(result, dict)
            assert "total" in result
            assert "contacts" in result
            assert "query" in result
        else:
            assert error is not None
    else:
        result, error = _call_method(
            connector, "search_contacts", arguments, label="search_contacts"
        )
        assert result is not None
        assert result["query"] == "search"
        assert len(result["contacts"]) == 1


@pytest.mark.skip(reason="demonstrating skipping")
def test_get_contact_by_email_py(connector: Any):
    """Test getting a specific contact by email"""
    arguments = {"email": "test@example.com"}

    if getattr(connector, "__is_real__", False):
        result, error = _call_method(
            connector, "get_contact_by_email", arguments, label="get_contact_by_email"
        )
        if result:
            assert isinstance(result, dict)
            if result:  # Might return None if not found
                assert "id" in result
                assert "properties" in result
        else:
            assert error is not None
    else:
        result, error = _call_method(
            connector, "get_contact_by_email", arguments, label="get_contact_by_email"
        )
        assert result is not None
        assert result["properties"]["email"] == "test@example.com"


# Deal management tests
@pytest.mark.skip(reason="demonstrating skipping")
def test_get_deals_py(connector: Any):
    """Test getting deals from HubSpot"""
    arguments = {"limit": 10, "properties": ["dealname", "amount", "dealstage"]}

    if getattr(connector, "__is_real__", False):
        result, error = _call_method(connector, "get_deals", arguments, label="get_deals")
        if result:
            assert isinstance(result, dict)
            assert "total" in result
            assert "deals" in result
            assert isinstance(result["deals"], list)
        else:
            assert error is not None
    else:
        result, error = _call_method(connector, "get_deals", arguments, label="get_deals")
        assert result is not None
        assert result["total"] == 1
        assert result["deals"][0]["properties"]["dealname"] == "Test Deal"


@pytest.mark.skip(reason="demonstrating skipping")
def test_create_deal_py(connector: Any):
    """Test creating a new deal in HubSpot"""
    arguments = {
        "properties": {
            "dealname": "Test Deal Creation",
            "amount": "5000",
            "dealstage": "appointmentscheduled",
        }
    }

    if getattr(connector, "__is_real__", False):
        result, error = _call_method(connector, "create_deal", arguments, label="create_deal")
        if result:
            assert isinstance(result, dict)
            assert "id" in result
            assert "properties" in result
        else:
            assert error is not None
    else:
        result, error = _call_method(connector, "create_deal", arguments, label="create_deal")
        assert result is not None
        assert result["id"] == "newdeal"


# Company management tests
@pytest.mark.skip(reason="demonstrating skipping")
def test_get_companies_py(connector: Any):
    """Test getting companies from HubSpot"""
    arguments = {"limit": 10, "properties": ["name", "domain", "city"]}

    if getattr(connector, "__is_real__", False):
        result, error = _call_method(connector, "get_companies", arguments, label="get_companies")
        if result:
            assert isinstance(result, dict)
            assert "total" in result
            assert "companies" in result
            assert isinstance(result["companies"], list)
        else:
            assert error is not None
    else:
        result, error = _call_method(connector, "get_companies", arguments, label="get_companies")
        assert result is not None
        assert result["total"] == 1
        assert result["companies"][0]["properties"]["name"] == "Test Company"


# Marketing events tests
@pytest.mark.skip(reason="demonstrating skipping")
def test_get_marketing_events_py(connector: Any):
    """Test getting marketing events from HubSpot"""
    arguments = {"limit": 10}

    if getattr(connector, "__is_real__", False):
        result, error = _call_method(
            connector, "get_marketing_events", arguments, label="get_marketing_events"
        )
        if error:
            pytest.fail(f"get_marketing_events failed with error: {error}")

        assert result is not None, "Result should not be None when no error occurs"
        assert isinstance(result, dict), "Result should be a dictionary"
        assert "total" in result, "Result should contain 'total' key"
        assert "events" in result, "Result should contain 'events' key"
        assert isinstance(result["events"], list), "Events should be a list"
    else:
        result, error = _call_method(
            connector, "get_marketing_events", arguments, label="get_marketing_events"
        )
        assert result is not None
        assert result["total"] == 1
        assert result["events"][0]["name"] == "Test Event"


# Parametrized tests for different contact operations
@pytest.mark.parametrize(
    "method_name,arguments,expected_keys",
    [
        ("get_contacts", {"limit": 5}, ["total", "contacts"]),
        ("get_deals", {"limit": 3}, ["total", "deals"]),
        ("get_companies", {"limit": 2}, ["total", "companies"]),
        ("get_marketing_events", {"limit": 1}, ["total", "events"]),
    ],
    ids=["contacts", "deals", "companies", "events"],
)
@pytest.mark.skip(reason="demonstrating skipping")
def test_get_operations_parametrized_py(
    connector: Any,
    method_name: (
        Literal["get_contacts"]
        | Literal["get_deals"]
        | Literal["get_companies"]
        | Literal["get_marketing_events"]
    ),
    arguments: dict[str, int],
    expected_keys: list[str],
):
    """Parametrized test for various GET operations"""
    if getattr(connector, "__is_real__", False):
        result, error = _call_method(connector, method_name, arguments, label=method_name)
        if result:
            for key in expected_keys:
                assert key in result
        else:
            assert error is not None
    else:
        result, error = _call_method(connector, method_name, arguments, label=method_name)
        assert result is not None
        for key in expected_keys:
            assert key in result


@pytest.mark.parametrize(
    "properties,should_succeed",
    [
        ({"email": "valid@example.com", "firstname": "John", "lastname": "Doe"}, True),
        ({"email": "valid2@example.com"}, True),  # Email only is sufficient
        ({"firstname": "John", "lastname": "Doe"}, False),  # Missing email should fail
        ({}, False),  # Empty properties should fail
    ],
    ids=["full_properties", "email_only", "no_email", "empty_properties"],
)
@pytest.mark.skip(reason="demonstrating skipping")
def test_create_contact_validation_py(
    connector: Any, properties: dict[str, str], should_succeed: bool
):
    """Parametrized test for contact creation validation"""
    arguments = {"properties": properties}

    if getattr(connector, "__is_real__", False):
        result, error = _call_method(
            connector, "create_contact", arguments, label="create_contact_validation"
        )
        if should_succeed:
            # Valid data might still fail due to SDK issues, but should not be a validation error
            if error:
                # Should be SDK-related error, not validation error
                assert "required" not in error.lower() or "hubspot sdk" in error.lower()
        else:
            # Invalid data should fail with validation error
            assert error is not None
            if "Email is required" not in error:
                # If not validation error, should be SDK error
                assert "hubspot sdk" in error.lower() or "importerror" in error.lower()
    else:
        result, error = _call_method(
            connector, "create_contact", arguments, label="create_contact_validation"
        )
        if should_succeed:
            assert result is not None
        else:
            assert error is not None


@pytest.mark.parametrize(
    "properties,should_succeed",
    [
        ({"dealname": "Test Deal", "amount": "1000"}, True),
        ({"dealname": "Another Deal"}, True),  # Deal name only is sufficient
        ({"amount": "5000"}, False),  # Missing deal name should fail
        ({}, False),  # Empty properties should fail
    ],
    ids=["full_properties", "name_only", "no_name", "empty_properties"],
)
@pytest.mark.skip(reason="demonstrating skipping")
def test_create_deal_validation_py(
    connector: Any, properties: dict[str, str], should_succeed: bool
):
    """Parametrized test for deal creation validation"""
    arguments = {"properties": properties}

    if getattr(connector, "__is_real__", False):
        result, error = _call_method(
            connector, "create_deal", arguments, label="create_deal_validation"
        )
        if should_succeed:
            if error:
                assert "required" not in error.lower() or "hubspot sdk" in error.lower()
        else:
            assert error is not None
            if "dealname is required" not in error:
                assert "hubspot sdk" in error.lower() or "importerror" in error.lower()
    else:
        result, error = _call_method(
            connector, "create_deal", arguments, label="create_deal_validation"
        )
        if should_succeed:
            assert result is not None
        else:
            assert error is not None


# Error handling tests
@pytest.mark.skip(reason="demonstrating skipping")
def test_missing_access_token_py():
    """Test that missing access token is handled properly"""
    if MCPHubspotConnector is None:
        pytest.skip("MCPHubspotConnector not available")

    # Test with empty settings
    empty_settings = {}
    try:
        connector = MCPHubspotConnector(logger, **empty_settings)
        assert False, "Should have raised an error for missing access token"
    except ValueError as e:
        assert "hubspot_access_token is required" in str(e)
    except ImportError as e:
        # SDK not available is also acceptable
        assert "hubspot sdk" in str(e).lower()


@pytest.mark.skip(reason="demonstrating skipping")
def test_invalid_method_arguments_py(connector: Any):
    """Test handling of invalid method arguments"""
    # Test update_contact without contact_id
    result, error = _call_method(connector, "update_contact", {"properties": {"firstname": "Test"}})

    if getattr(connector, "__is_real__", False):
        assert error is not None
        if "contact_id is required" not in error:
            # Should be SDK-related error
            assert "hubspot sdk" in error.lower() or "importerror" in error.lower()
    else:
        # Mock doesn't validate, so it should succeed
        assert result is not None or error is not None


@pytest.mark.skip(reason="demonstrating skipping")
def test_sdk_availability_handling_py():
    """Test that the connector handles SDK availability properly"""
    if MCPHubspotConnector is None:
        pytest.skip("MCPHubspotConnector not available")

    # This test verifies the import error handling is working
    with patch("mcp_hubspot_connector.mcp_hubspot_connector.HUBSPOT_AVAILABLE", False):
        try:
            connector = MCPHubspotConnector(logger, **SETTING)
            assert False, "Should have raised ImportError when SDK not available"
        except ImportError as e:
            assert "hubspot sdk not available" in str(e).lower()


# Integration-style tests with mock HubSpot responses
@pytest.mark.skip(reason="Integration test - enable when HubSpot SDK is properly configured")
def test_full_contact_workflow_py(connector: Any):
    """Integration test for full contact workflow"""
    if not getattr(connector, "__is_real__", False):
        pytest.skip("Requires real HubSpot connector")

    # 1. Create a contact
    create_result, create_error = _call_method(
        connector,
        "create_contact",
        {
            "properties": {
                "email": "workflow@example.com",
                "firstname": "Workflow",
                "lastname": "Test",
            }
        },
        "workflow_create",
    )
    assert create_result is not None, f"Create failed: {create_error}"
    contact_id = create_result["id"]

    # 2. Update the contact
    update_result, update_error = _call_method(
        connector,
        "update_contact",
        {"contact_id": contact_id, "properties": {"lastname": "Updated"}},
        "workflow_update",
    )
    assert update_result is not None, f"Update failed: {update_error}"

    # 3. Search for the contact
    search_result, search_error = _call_method(
        connector, "search_contacts", {"query": "workflow@example.com"}, "workflow_search"
    )
    assert search_result is not None, f"Search failed: {search_error}"
    assert search_result["total"] > 0, "Contact not found in search"


# Analytics function tests
@pytest.mark.asyncio
@pytest.mark.skip(reason="demonstrating skipping")
async def test_get_contact_analytics_py(connector: Any):
    """Test contact analytics functionality"""
    # Calculate dynamic date range: 3 months ago to now using pendulum
    now = pendulum.now("UTC")
    three_months_ago = now.subtract(months=3)

    if getattr(connector, "__is_real__", False):
        result, error = await _call_async_method(
            connector,
            "get_contact_analytics",
            {
                "params": {
                    "segmentation": "engagement_level",
                    "limit": 200,
                    "date_range": {
                        "start": three_months_ago.to_iso8601_string(),
                        "end": now.to_iso8601_string(),
                    },
                    "include_engagement": True,
                }
            },
            "get_contact_analytics",
        )
        if error:
            pytest.fail(f"get_contact_analytics failed with error: {error}")

        assert result is not None, "Result should not be None when no error occurs"
        assert isinstance(result, dict), "Result should be a dictionary"
        assert "success" in result, "Result should have 'success' key"
        assert "data" in result, "Result should have 'data' key"

        if not result["success"]:
            pytest.fail(f"get_contact_analytics returned success=False")

        assert "totalContacts" in result["data"], "Result data should contain 'totalContacts'"
        assert "activeContacts" in result["data"], "Result data should contain 'activeContacts'"
    else:
        result, error = await _call_async_method(
            connector,
            "get_contact_analytics",
            {
                "params": {
                    "segmentation": "engagement_level",
                    "date_range": {
                        "start": three_months_ago.to_iso8601_string(),
                        "end": now.to_iso8601_string(),
                    },
                    "include_engagement": True,
                    "limit": 500,
                }
            },
            "get_contact_analytics",
        )
        if error:
            pytest.fail(f"get_contact_analytics failed with error: {error}")

        assert result is not None, "Result should not be None when no error occurs"
        assert result["success"] is True, "Mock result should always be successful"
        assert (
            result["data"]["totalContacts"] == 100
        ), "Mock result should have expected totalContacts"


@pytest.mark.asyncio
@pytest.mark.skip(reason="demonstrating skipping")
async def test_analyze_campaign_performance_py(connector: Any):
    """Test campaign performance analysis"""
    if getattr(connector, "__is_real__", False):
        # Calculate dynamic date range: 6 months ago to now for campaign analysis using pendulum
        now = pendulum.now("UTC")
        six_months_ago = now.subtract(months=6)

        result, error = await _call_async_method(
            connector,
            "analyze_campaign_performance",
            {
                "params": {
                    "benchmark_type": "industry",
                    "metrics": ["open_rate", "click_rate", "bounce_rate", "conversion_rate"],
                    "date_range": {
                        "start": six_months_ago.to_iso8601_string(),
                        "end": now.to_iso8601_string(),
                    },
                }
            },
            "analyze_campaign_performance",
        )
        if error:
            pytest.fail(f"analyze_campaign_performance failed with error: {error}")

        assert result is not None, "Result should not be None when no error occurs"
        assert isinstance(result, dict), "Result should be a dictionary"
        assert "success" in result, "Result should have 'success' key"
        assert "data" in result, "Result should have 'data' key"

        if not result["success"]:
            pytest.fail(f"analyze_campaign_performance returned success=False")

        assert "campaigns" in result["data"], "Result data should contain 'campaigns'"
        assert "summary_metrics" in result["data"], "Result data should contain 'summary_metrics'"
    else:
        result, error = await _call_async_method(
            connector,
            "analyze_campaign_performance",
            {"params": {}},
            "analyze_campaign_performance",
        )
        if error:
            pytest.fail(f"analyze_campaign_performance failed with error: {error}")

        assert result is not None, "Result should not be None when no error occurs"
        assert result["success"] is True, "Mock result should always be successful"
        assert (
            "summary_metrics" in result["data"]
        ), "Mock result should have expected summary_metrics"


@pytest.mark.asyncio
@pytest.mark.skip(reason="demonstrating skipping")
async def test_analyze_sales_pipeline_py(connector: Any):
    """Test sales pipeline analysis"""
    # Calculate dynamic date range: 3 weeks ago to now using pendulum
    now = pendulum.now("UTC")
    three_weeks_ago = now.subtract(weeks=3)

    if getattr(connector, "__is_real__", False):
        result, error = await _call_async_method(
            connector,
            "analyze_sales_pipeline",
            {
                "params": {
                    "analysis_type": "conversion_rates",
                    "timeframe": {
                        "start": three_weeks_ago.to_iso8601_string(),
                        "end": now.to_iso8601_string(),
                    },
                }
            },
            "analyze_sales_pipeline",
        )
        if error:
            pytest.fail(f"analyze_sales_pipeline failed with error: {error}")

        assert result is not None, "Result should not be None when no error occurs"
        assert isinstance(result, dict), "Result should be a dictionary"
        assert "success" in result, "Result should have 'success' key"
        assert "data" in result, "Result should have 'data' key"

        if not result["success"]:
            pytest.fail(f"analyze_sales_pipeline returned success=False")

        assert "totalValue" in result["data"], "Result data should contain 'totalValue'"
        assert "dealCount" in result["data"], "Result data should contain 'dealCount'"
    else:
        result, error = await _call_async_method(
            connector, "analyze_sales_pipeline", {"params": {}}, "analyze_sales_pipeline"
        )
        if error:
            pytest.fail(f"analyze_sales_pipeline failed with error: {error}")

        assert result is not None, "Result should not be None when no error occurs"
        assert result["success"] is True, "Mock result should always be successful"
        assert result["data"]["totalValue"] == 50000, "Mock result should have expected totalValue"


@pytest.mark.asyncio
@pytest.mark.skip(reason="demonstrating skipping")
async def test_predict_lead_scores_py(connector: Any):
    """Test lead scoring prediction"""
    # Calculate dynamic date range: 3 months ago to now using pendulum
    now = pendulum.now("UTC")
    three_months_ago = now.subtract(months=3)

    if getattr(connector, "__is_real__", False):
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
            "predict_lead_scores",
        )
        if error:
            pytest.fail(f"predict_lead_scores failed with error: {error}")

        assert result is not None, "Result should not be None when no error occurs"
        assert isinstance(result, dict), "Result should be a dictionary"
        assert "success" in result, "Result should have 'success' key"
        assert "data" in result, "Result should have 'data' key"

        if not result["success"]:
            pytest.fail(f"predict_lead_scores returned success=False")

        assert "predictions" in result["data"], "Result data should contain 'predictions'"
        assert (
            "model_performance" in result["data"]
        ), "Result data should contain 'model_performance'"
    else:
        result, error = await _call_async_method(
            connector, "predict_lead_scores", {"params": {}}, "predict_lead_scores"
        )
        if error:
            pytest.fail(f"predict_lead_scores failed with error: {error}")

        assert result is not None, "Result should not be None when no error occurs"
        assert result["success"] is True, "Mock result should always be successful"
        assert len(result["data"]["predictions"]) > 0, "Mock result should have predictions"


@pytest.mark.asyncio
# @pytest.mark.skip(reason="demonstrating skipping")
async def test_create_contact_segments_py(connector: Any):
    """Test contact segmentation functionality"""
    # Calculate dynamic date range: 6 months ago to now using pendulum (recommended for performance)
    now = pendulum.now("UTC")
    six_months_ago = now.subtract(months=6)

    if getattr(connector, "__is_real__", False):
        result, error = await _call_async_method(
            connector,
            "create_contact_segments",
            {
                "params": {
                    "segmentation_type": "behavioral",
                    "criteria": {
                        "engagement_score": {"min": 5},
                        "lifetime_value": {"min": 1000},
                        "date_range": {
                            "start": six_months_ago.to_iso8601_string(),
                            "end": now.to_iso8601_string(),
                        },
                    },
                    "number_of_segments": 5,
                    "limit": 500,  # Reasonable limit for testing
                }
            },
            "create_contact_segments",
        )
        if error:
            pytest.fail(f"create_contact_segments failed with error: {error}")

        assert result is not None, "Result should not be None when no error occurs"
        assert isinstance(result, dict), "Result should be a dictionary"
        assert "success" in result, "Result should have 'success' key"
        assert "data" in result, "Result should have 'data' key"

        if not result["success"]:
            pytest.fail(f"create_contact_segments returned success=False")

        assert "segments" in result["data"], "Result data should contain 'segments'"
        assert (
            "segmentation_quality" in result["data"]
        ), "Result data should contain 'segmentation_quality'"
        assert (
            "contact_assignments" in result["data"]
        ), "Result data should contain 'contact_assignments'"
        assert isinstance(result["data"]["segments"], list), "Segments should be a list"
        assert (
            "total_contacts" in result["data"]["segmentation_quality"]
        ), "Segmentation quality should contain 'total_contacts'"
    else:
        result, error = await _call_async_method(
            connector,
            "create_contact_segments",
            {
                "params": {
                    "segmentation_type": "behavioral",
                    "number_of_segments": 5,
                    "criteria": {
                        "engagement_score": {"min": 5},
                        "date_range": {
                            "start": six_months_ago.to_iso8601_string(),
                            "end": now.to_iso8601_string(),
                        },
                    },
                    "limit": 500,  # Reasonable limit for testing
                }
            },
            "create_contact_segments",
        )
        if error:
            pytest.fail(f"create_contact_segments failed with error: {error}")

        assert result is not None, "Result should not be None when no error occurs"
        assert result["success"] is True, "Mock result should always be successful"
        assert (
            result["data"]["segmentation_quality"]["num_segments"] == 2
        ), "Mock result should have expected segment count"
        assert len(result["data"]["segments"]) == 2, "Mock result should have 2 segments"
        assert (
            result["data"]["segments"][0]["segment_id"] == "high_value"
        ), "First segment should be high_value"


@pytest.mark.asyncio
@pytest.mark.skip(reason="demonstrating skipping")
async def test_forecast_revenue_py(connector: Any):
    """Test revenue forecasting functionality"""
    if getattr(connector, "__is_real__", False):
        result, error = await _call_async_method(
            connector,
            "forecast_revenue",
            {"params": {"forecast_period": "90_days", "confidence_level": 0.95}},
            "forecast_revenue",
        )
        if error:
            pytest.fail(f"forecast_revenue failed with error: {error}")

        assert result is not None, "Result should not be None when no error occurs"
        assert isinstance(result, dict), "Result should be a dictionary"
        assert "success" in result, "Result should have 'success' key"
        assert "data" in result, "Result should have 'data' key"

        if not result["success"]:
            pytest.fail(f"forecast_revenue returned success=False")

        assert "forecast" in result["data"], "Result data should contain 'forecast'"
        assert (
            "confidence_interval" in result["data"]
        ), "Result data should contain 'confidence_interval'"
        assert "scenarios" in result["data"], "Result data should contain 'scenarios'"
        assert (
            "model_performance" in result["data"]
        ), "Result data should contain 'model_performance'"

        # Check metadata structure
        assert "metadata" in result, "Result should contain 'metadata'"
        assert "forecast_period" in result["metadata"], "Metadata should contain 'forecast_period'"
        assert (
            "confidence_level" in result["metadata"]
        ), "Metadata should contain 'confidence_level'"
    else:
        result, error = await _call_async_method(
            connector,
            "forecast_revenue",
            {"params": {"forecast_period": "90_days", "confidence_level": 0.95}},
            "forecast_revenue",
        )
        if error:
            pytest.fail(f"forecast_revenue failed with error: {error}")

        assert result is not None, "Result should not be None when no error occurs"
        assert result["success"] is True, "Mock result should always be successful"
        assert (
            result["data"]["forecast"]["total_revenue"] == 1250000
        ), "Mock result should have expected forecast"
        assert (
            "confidence_interval" in result["data"]
        ), "Mock result should have confidence interval"
        assert (
            len(result["data"]["forecast"]["breakdown"]) == 3
        ), "Mock result should have 3 period breakdown"
        assert (
            result["data"]["scenarios"]["realistic"] == 1250000
        ), "Mock should have realistic scenario"
        assert (
            result["metadata"]["forecast_period"] == "90_days"
        ), "Mock should have correct forecast period"


@pytest.mark.asyncio
@pytest.mark.skip(reason="demonstrating skipping")
async def test_generate_executive_report_py(connector: Any):
    """Test executive report generation functionality"""
    # Calculate dynamic report period: last month using pendulum
    now = pendulum.now("UTC")
    last_month_start = now.subtract(months=1).start_of("month")
    last_month_end = now.subtract(months=1).end_of("month")

    if getattr(connector, "__is_real__", False):
        result, error = await _call_async_method(
            connector,
            "generate_executive_report",
            {
                "params": {
                    "report_type": "monthly",
                    "timeframe": {
                        "start": last_month_start.to_iso8601_string(),
                        "end": last_month_end.to_iso8601_string(),
                    },
                    "include_forecast": True,
                    "departments": ["sales", "marketing"],
                    "detail_level": "executive",
                }
            },
            "generate_executive_report",
        )
        if error:
            pytest.fail(f"generate_executive_report failed with error: {error}")

        assert result is not None, "Result should not be None when no error occurs"
        assert isinstance(result, dict), "Result should be a dictionary"
        assert "success" in result, "Result should have 'success' key"
        assert "data" in result, "Result should have 'data' key"

        if not result["success"]:
            pytest.fail(f"generate_executive_report returned success=False")

        assert "report_id" in result["data"], "Result data should contain 'report_id'"
        assert (
            "executive_summary" in result["data"]
        ), "Result data should contain 'executive_summary'"
        assert "key_metrics" in result["data"], "Result data should contain 'key_metrics'"
        assert (
            "departmental_breakdown" in result["data"]
        ), "Result data should contain 'departmental_breakdown'"
    else:
        result, error = await _call_async_method(
            connector,
            "generate_executive_report",
            {"params": {"report_type": "monthly", "include_forecast": True}},
            "generate_executive_report",
        )
        if error:
            pytest.fail(f"generate_executive_report failed with error: {error}")

        assert result is not None, "Result should not be None when no error occurs"
        assert result["success"] is True, "Mock result should always be successful"
        assert (
            result["data"]["report_id"] == "exec_report_2024_01"
        ), "Mock result should have expected report ID"
        assert "executive_summary" in result["data"], "Mock result should have executive summary"
        assert (
            result["data"]["executive_summary"]["total_contacts"] == 2500
        ), "Mock should have expected contact count"
        assert (
            "departmental_breakdown" in result["data"]
        ), "Mock result should have departmental breakdown"
        assert (
            len(result["data"]["performance_highlights"]) > 0
        ), "Mock should have performance highlights"


if __name__ == "__main__":
    # Allow running this file directly with Python and pass through pytest args
    import argparse as _argparse
    import sys as _sys

    import pytest as _pytest

    parser = _argparse.ArgumentParser(add_help=False)
    parser.add_argument("--env", action="append", help="Set env var: KEY=VALUE (can repeat)")
    known, unknown = parser.parse_known_args(_sys.argv[1:])

    if known.env:
        for pair in known.env:
            if "=" in pair:
                k, v = pair.split("=", 1)
                os.environ[k] = v

    _sys.exit(_pytest.main([__file__] + list(unknown)))
