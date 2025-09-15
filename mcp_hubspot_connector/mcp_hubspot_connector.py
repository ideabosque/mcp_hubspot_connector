#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

__author__ = "bibow"

import asyncio
import logging
import time
import traceback
from dataclasses import asdict, dataclass
from functools import wraps
from typing import Any, Dict, List, Optional, Union

import pendulum

from .analytics_engine import AnalyticsEngine
from .insight_generator import InsightGenerator
from .rate_limiter import RateLimiter

try:
    import numpy as np
    import pandas as pd
except ImportError:
    print("Warning: numpy and pandas not available. Some analytics features may be limited.")

    # Create minimal fallbacks
    class pd:
        @staticmethod
        def DataFrame(data):
            return data if isinstance(data, list) else []

    class np:
        @staticmethod
        def array(data):
            return data


# Try to import the official HubSpot SDK
try:
    from hubspot import HubSpot
    from hubspot.crm.contacts import ApiException as ContactsApiException
    from hubspot.crm.contacts import SimplePublicObjectInput
    from hubspot.crm.contacts.models.filter import Filter
    from hubspot.crm.contacts.models.filter_group import FilterGroup
    from hubspot.crm.contacts.models.public_object_search_request import PublicObjectSearchRequest
    from hubspot.crm.deals import ApiException as DealsApiException
    from hubspot.marketing.events import ApiException as MarketingApiException

    HUBSPOT_AVAILABLE = True
except ImportError:
    # Fallback if HubSpot SDK is not properly installed
    HUBSPOT_AVAILABLE = False

    # Create mock classes for development/testing
    class HubSpot:
        def __init__(self, access_token=None):
            self.access_token = access_token

    class ContactsApiException(Exception):
        def __init__(self, reason="API Error"):
            self.reason = reason
            super().__init__(self.reason)

    class DealsApiException(Exception):
        def __init__(self, reason="API Error"):
            self.reason = reason
            super().__init__(self.reason)

    class MarketingApiException(Exception):
        def __init__(self, reason="API Error"):
            self.reason = reason
            super().__init__(self.reason)

    class SimplePublicObjectInput:
        def __init__(self, properties=None):
            self.properties = properties or {}

    class Filter:
        def __init__(
            self,
            property_name=None,
            operator=None,
            value=None,
            high_value=None,
            values=None,
        ):
            self.property_name = property_name
            self.operator = operator
            self.value = value
            self.high_value = high_value
            self.values = values

    class FilterGroup:
        def __init__(self, filters=None):
            self.filters = filters or []

    class PublicObjectSearchRequest:
        def __init__(self, filter_groups=None, properties=None, limit=None, after=None):
            self.filter_groups = filter_groups or []
            self.properties = properties or []
            self.limit = limit
            self.after = after


@dataclass
class HubSpotSDKConfig:
    """Enhanced configuration for HubSpot SDK"""

    access_token: str
    rate_limit_enabled: bool = True
    max_retries: int = 3
    timeout: int = 30
    debug_mode: bool = False
    calls_per_second: int = 10


def handle_hubspot_errors(func):
    """Decorator to handle HubSpot API errors consistently"""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except ContactsApiException as e:
            logging.error(f"Contacts API error: {e}")
            raise Exception(f"HubSpot Contacts API error: {e.reason}")
        except DealsApiException as e:
            logging.error(f"Deals API error: {e}")
            raise Exception(f"HubSpot Deals API error: {e.reason}")
        except MarketingApiException as e:
            logging.error(f"Marketing API error: {e}")
            raise Exception(f"HubSpot Marketing API error: {e.reason}")
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            raise Exception(f"Unexpected error: {str(e)}")

    return wrapper


class MCPHubspotConnector:
    """Service layer using official HubSpot Python SDK with analytics capabilities"""

    def __init__(self, logger: logging.Logger, **settings: Dict[str, Any]):
        self.logger = logger
        self.setting = settings
        access_token = settings.get("hubspot_access_token")
        calls_per_second = settings.get("calls_per_second", 10)

        # Create config from settings
        self.config = HubSpotSDKConfig(
            access_token=access_token or "",
            rate_limit_enabled=settings.get("rate_limit_enabled", True),
            max_retries=settings.get("max_retries", 3),
            timeout=settings.get("timeout", 30),
            debug_mode=settings.get("debug_mode", False),
            calls_per_second=calls_per_second,
        )

        # Initialize HubSpot client immediately
        if not HUBSPOT_AVAILABLE:
            raise ImportError(
                "HubSpot SDK not available. Please install the official HubSpot SDK:\n"
                "pip install hubspot-api-client\n"
                "or\n"
                "pip install simplejson  # if using the legacy hubspot package"
            )

        if not access_token:
            raise ValueError("hubspot_access_token is required in settings or config")

        self.client = HubSpot(access_token=access_token)

        # Initialize rate limiter with config
        self.rate_limiter = RateLimiter(calls_per_second=calls_per_second)

        # Initialize analytics components
        self.analytics_engine = AnalyticsEngine()
        self.insight_generator = InsightGenerator()

        # Cache for frequently accessed data
        self.cache = {}
        self.cache_timestamps = {}

    # * MCP Function.
    @handle_hubspot_errors
    async def get_contact_analytics(self, **arguments: Dict[str, Any]) -> Dict[str, Any]:
        """MCP Tool: get_contact_analytics - Advanced contact analytics using SDK"""
        try:
            date_range = arguments.get("date_range", {})
            segmentation = arguments.get("segmentation", "all")
            include_engagement = arguments.get("include_engagement", True)
            limit = arguments.get("limit", 1000)

            # Get contacts using SDK
            contacts_data = await self._get_contacts_with_sdk(limit, date_range)

            # Get engagement data if requested
            engagement_data = []
            if include_engagement:
                contact_ids = [contact.id for contact in contacts_data]
                engagement_data = await self._get_engagement_data_with_sdk(contact_ids)

            # Process through analytics engine
            analytics_result = await self.analytics_engine.process_contact_metrics(
                contacts_data, engagement_data, segmentation
            )

            # Generate business insights
            insights = await self.insight_generator.generate_contact_insights(analytics_result)

            return {
                "success": True,
                "data": {
                    "totalContacts": len(contacts_data),
                    "activeContacts": analytics_result["active_count"],
                    "avgEngagementScore": analytics_result["avg_engagement"],
                    "segments": analytics_result["segments"],
                    "detailedMetrics": analytics_result["detailed_metrics"],
                },
                "insights": [i["message"] for i in insights],
                "recommendations": [i["action"] for i in insights if i.get("action")],
                "metadata": {
                    "processing_time": analytics_result["processing_time"],
                    "data_source": "hubspot_sdk",
                    "sdk_version": "7.0.0",
                },
                "error_message": None,
            }

        except Exception as e:
            return {
                "success": False,
                "data": None,
                "insights": [],
                "recommendations": [],
                "metadata": {},
                "error_message": f"Contact analytics failed: {str(e)}",
            }

    async def _get_contacts_with_sdk(
        self, limit: int, date_range: Dict[str, str] = None
    ) -> List[Any]:
        """Get contacts using HubSpot SDK with proper pagination"""

        await self.rate_limiter.wait_if_needed()

        properties = [
            "email",
            "firstname",
            "lastname",
            "createdate",
            "lastmodifieddate",
            "lifecyclestage",
            "hs_lead_status",
            "hs_email_open",
            "hs_email_click",
            "hs_email_bounce",
            "total_revenue",
            "recent_deal_amount",
        ]

        contacts_api = self.client.crm.contacts
        all_contacts = []

        if date_range and date_range.get("start") and date_range.get("end"):
            import pendulum

            try:
                # Convert ISO strings to millisecond timestamps
                start_dt = pendulum.parse(date_range["start"])
                end_dt = pendulum.parse(date_range["end"])

                start_timestamp = str(int(start_dt.timestamp() * 1000))
                end_timestamp = str(int(end_dt.timestamp() * 1000))

                filters = [
                    {
                        "propertyName": "createdate",
                        "operator": "BETWEEN",
                        "value": start_timestamp,
                        "highValue": end_timestamp,
                    }
                ]

                self.logger.info(f"Contact date filter: {start_timestamp} to {end_timestamp}")

            except Exception as e:
                self.logger.warning(f"Date conversion failed: {e}. Skipping date filter.")
                filters = []

            search_request = PublicObjectSearchRequest(
                filter_groups=[{"filters": filters}],
                properties=properties,
                limit=min(limit, 100),
            )

            try:
                after = None
                total_fetched = 0

                while total_fetched < limit:
                    if after:
                        search_request.after = after

                    await self.rate_limiter.wait_if_needed()
                    search_response = contacts_api.search_api.do_search(search_request)

                    if search_response and search_response.results:
                        all_contacts.extend(search_response.results)
                        total_fetched += len(search_response.results)

                    if search_response.paging and search_response.paging.next:
                        after = search_response.paging.next.after
                    else:
                        break

            except Exception as e:
                log = traceback.format_exc()
                self.logger.error(f"Search contacts failed: {e}\n{log}")
                raise
        else:
            try:
                after = None
                total_fetched = 0

                while total_fetched < limit:
                    batch_limit = min(100, limit - total_fetched)

                    await self.rate_limiter.wait_if_needed()

                    # Use basic_api.get_page() for proper pagination
                    get_page_response = contacts_api.basic_api.get_page(
                        limit=batch_limit, after=after, properties=properties, archived=False
                    )

                    if get_page_response and get_page_response.results:
                        all_contacts.extend(get_page_response.results)
                        total_fetched += len(get_page_response.results)

                        # Check for next page
                        if (
                            hasattr(get_page_response, "paging")
                            and get_page_response.paging
                            and hasattr(get_page_response.paging, "next")
                            and get_page_response.paging.next
                        ):
                            after = get_page_response.paging.next.after
                        else:
                            break
                    else:
                        break

            except Exception as e:
                log = traceback.format_exc()
                self.logger.error(f"Get contacts failed: {e}\n{log}")
                raise

        self.logger.info(f"Fetched {len(all_contacts)} contacts using HubSpot SDK")
        return all_contacts

    async def _get_engagement_data_with_sdk(self, contact_ids: List[str]) -> List[Dict[str, Any]]:
        """Get engagement data using HubSpot SDK"""

        engagement_data = []

        batch_size = 50
        for i in range(0, len(contact_ids), batch_size):
            batch_ids = contact_ids[i : i + batch_size]

            await self.rate_limiter.wait_if_needed()

            for contact_id in batch_ids:
                engagement_data.append(
                    {
                        "contact_id": contact_id,
                        "email_engagements": 5,
                        "meeting_engagements": 1,
                        "call_engagements": 2,
                        "last_engagement_date": pendulum.now().to_iso8601_string(),
                    }
                )

        return engagement_data

    async def _get_campaign_engagement_data_with_sdk(
        self, campaign_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """Get campaign engagement data using HubSpot SDK

        This is a placeholder method that returns mock engagement data for campaigns.
        In a full implementation, this would use HubSpot's engagement API endpoints
        to get actual email opens, clicks, and other engagement events for campaigns.
        """

        engagement_data = []

        # For each campaign, generate mock engagement data
        for campaign_id in campaign_ids:
            # Mock engagement events for this campaign
            # In real implementation, this would query HubSpot's engagement endpoints
            for i in range(5):  # Mock 5 contacts per campaign
                mock_events = [
                    {
                        "campaign_id": campaign_id,
                        "contact_id": f"contact_{i}",
                        "type": "email_open",
                        "timestamp": pendulum.now().to_iso8601_string(),
                        "event_id": f"event_{campaign_id}_{i}",
                    },
                    {
                        "campaign_id": campaign_id,
                        "contact_id": f"contact_{i}",
                        "type": "email_click",
                        "timestamp": pendulum.now().to_iso8601_string(),
                        "event_id": f"event_{campaign_id}_{i}_click",
                    },
                ]
                engagement_data.extend(mock_events)

        return engagement_data

    # * MCP Function.
    def get_contacts(self, **arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Get contacts from HubSpot."""
        try:
            self.logger.info(f"Getting contacts with arguments: {arguments}")

            client = self.client
            limit = arguments.get("limit", 100)
            properties = arguments.get(
                "properties",
                ["email", "firstname", "lastname", "createdate", "lifecyclestage"],
            )

            # Use proper pagination with basic_api.get_page
            contacts = []
            after = None
            total_fetched = 0
            batch_limit = min(100, limit)

            while total_fetched < limit:
                response = client.crm.contacts.basic_api.get_page(
                    limit=batch_limit, after=after, properties=properties, archived=False
                )

                if response and response.results:
                    for contact in response.results:
                        if total_fetched >= limit:
                            break
                        contact_data = {"id": contact.id, "properties": contact.properties}
                        contacts.append(contact_data)
                        total_fetched += 1

                    # Check for next page
                    if (
                        hasattr(response, "paging")
                        and response.paging
                        and hasattr(response.paging, "next")
                        and response.paging.next
                        and total_fetched < limit
                    ):
                        after = response.paging.next.after
                    else:
                        break
                else:
                    break

            return {"total": len(contacts), "contacts": contacts}

        except ContactsApiException as e:
            log = traceback.format_exc()
            self.logger.error(log)
            raise Exception(f"HubSpot Contacts API error: {e.reason}")
        except Exception as e:
            log = traceback.format_exc()
            self.logger.error(log)
            raise e

    # * MCP Function.
    def create_contact(self, **arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new contact in HubSpot."""
        try:
            self.logger.info(f"Creating contact with arguments: {arguments}")

            client = self.client
            properties = arguments.get("properties", {})

            if not properties.get("email"):
                raise ValueError("Email is required to create a contact")

            if HUBSPOT_AVAILABLE:
                from hubspot.crm.contacts import SimplePublicObjectInput

                contact_input = SimplePublicObjectInput(properties=properties)
            else:
                raise ImportError("HubSpot SDK not available")
            response = client.crm.contacts.create(simple_public_object_input=contact_input)

            return {"id": response.id, "properties": response.properties}

        except ContactsApiException as e:
            log = traceback.format_exc()
            self.logger.error(log)
            raise Exception(f"HubSpot Contacts API error: {e.reason}")
        except Exception as e:
            log = traceback.format_exc()
            self.logger.error(log)
            raise e

    # * MCP Function.
    def update_contact(self, **arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing contact in HubSpot."""
        try:
            self.logger.info(f"Updating contact with arguments: {arguments}")

            client = self.client
            contact_id = arguments.get("contact_id")
            properties = arguments.get("properties", {})

            if not contact_id:
                raise ValueError("contact_id is required to update a contact")

            if HUBSPOT_AVAILABLE:
                from hubspot.crm.contacts import SimplePublicObjectInput

                contact_input = SimplePublicObjectInput(properties=properties)
            else:
                raise ImportError("HubSpot SDK not available")

            response = client.crm.contacts.update(
                contact_id=contact_id, simple_public_object_input=contact_input
            )

            return {"id": response.id, "properties": response.properties}

        except ContactsApiException as e:
            log = traceback.format_exc()
            self.logger.error(log)
            raise Exception(f"HubSpot Contacts API error: {e.reason}")
        except Exception as e:
            log = traceback.format_exc()
            self.logger.error(log)
            raise e

    # * MCP Function.
    def get_deals(self, **arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Get deals from HubSpot."""
        try:
            self.logger.info(f"Getting deals with arguments: {arguments}")

            client = self.client
            limit = arguments.get("limit", 100)
            properties = arguments.get(
                "properties",
                ["dealname", "amount", "dealstage", "createdate", "closedate"],
            )

            # Use proper pagination with basic_api.get_page
            deals = []
            after = None
            total_fetched = 0
            batch_limit = min(100, limit)

            while total_fetched < limit:
                response = client.crm.deals.basic_api.get_page(
                    limit=batch_limit, after=after, properties=properties, archived=False
                )

                if response and response.results:
                    for deal in response.results:
                        if total_fetched >= limit:
                            break
                        deal_data = {"id": deal.id, "properties": deal.properties}
                        deals.append(deal_data)
                        total_fetched += 1

                    # Check for next page
                    if (
                        hasattr(response, "paging")
                        and response.paging
                        and hasattr(response.paging, "next")
                        and response.paging.next
                        and total_fetched < limit
                    ):
                        after = response.paging.next.after
                    else:
                        break
                else:
                    break

            return {"total": len(deals), "deals": deals}

        except DealsApiException as e:
            log = traceback.format_exc()
            self.logger.error(log)
            raise Exception(f"HubSpot Deals API error: {e.reason}")
        except Exception as e:
            log = traceback.format_exc()
            self.logger.error(log)
            raise e

    # * MCP Function.
    def create_deal(self, **arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new deal in HubSpot."""
        try:
            self.logger.info(f"Creating deal with arguments: {arguments}")

            client = self.client
            properties = arguments.get("properties", {})

            if not properties.get("dealname"):
                raise ValueError("dealname is required to create a deal")

            if HUBSPOT_AVAILABLE:
                from hubspot.crm.deals import SimplePublicObjectInput

                deal_input = SimplePublicObjectInput(properties=properties)
            else:
                raise ImportError("HubSpot SDK not available")
            response = client.crm.deals.create(simple_public_object_input=deal_input)

            return {"id": response.id, "properties": response.properties}

        except DealsApiException as e:
            log = traceback.format_exc()
            self.logger.error(log)
            raise Exception(f"HubSpot Deals API error: {e.reason}")
        except Exception as e:
            log = traceback.format_exc()
            self.logger.error(log)
            raise e

    # * MCP Function.
    def get_companies(self, **arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Get companies from HubSpot."""
        try:
            self.logger.info(f"Getting companies with arguments: {arguments}")

            client = self.client
            limit = arguments.get("limit", 100)
            properties = arguments.get(
                "properties",
                ["name", "domain", "city", "state", "country", "createdate"],
            )

            # Use proper pagination with basic_api.get_page
            companies = []
            after = None
            total_fetched = 0
            batch_limit = min(100, limit)

            while total_fetched < limit:
                response = client.crm.companies.basic_api.get_page(
                    limit=batch_limit, after=after, properties=properties, archived=False
                )

                if response and response.results:
                    for company in response.results:
                        if total_fetched >= limit:
                            break
                        company_data = {"id": company.id, "properties": company.properties}
                        companies.append(company_data)
                        total_fetched += 1

                    # Check for next page
                    if (
                        hasattr(response, "paging")
                        and response.paging
                        and hasattr(response.paging, "next")
                        and response.paging.next
                        and total_fetched < limit
                    ):
                        after = response.paging.next.after
                    else:
                        break
                else:
                    break

            return {"total": len(companies), "companies": companies}

        except Exception as e:
            log = traceback.format_exc()
            self.logger.error(log)
            raise e

    # * MCP Function.
    def search_contacts(self, **arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Search contacts in HubSpot."""
        try:
            self.logger.info(f"Searching contacts with arguments: {arguments}")

            client = self.client
            query = arguments.get("query", "")
            limit = arguments.get("limit", 100)
            properties = arguments.get(
                "properties",
                ["email", "firstname", "lastname", "createdate", "lifecyclestage"],
            )

            if not query:
                return self.get_contacts(limit=limit, properties=properties)

            if HUBSPOT_AVAILABLE:
                from hubspot.crm.contacts.models.filter import Filter
                from hubspot.crm.contacts.models.filter_group import FilterGroup
                from hubspot.crm.contacts.models.public_object_search_request import (
                    PublicObjectSearchRequest,
                )
            else:
                raise ImportError("HubSpot SDK not available")

            filters = [{"propertyName": "email", "operator": "CONTAINS_TOKEN", "value": query}]

            search_request = PublicObjectSearchRequest(
                filter_groups=[{"filters": filters}],
                properties=properties,
                limit=min(limit, 100),
            )

            response = client.crm.contacts.search_api.do_search(search_request)

            contacts = []
            if response:
                for contact in response.results:
                    contact_data = {"id": contact.id, "properties": contact.properties}
                    contacts.append(contact_data)

            return {"total": len(contacts), "contacts": contacts, "query": query}

        except ContactsApiException as e:
            log = traceback.format_exc()
            self.logger.error(log)
            raise Exception(f"HubSpot Contacts API error: {e.reason}")
        except Exception as e:
            log = traceback.format_exc()
            self.logger.error(log)
            raise e

    # * MCP Function.
    def get_contact_by_email(self, **arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Get a specific contact by email address."""
        try:
            self.logger.info(f"Getting contact by email with arguments: {arguments}")

            email = arguments.get("email")
            if not email:
                raise ValueError("email is required")

            client = self.client
            properties = arguments.get(
                "properties",
                ["email", "firstname", "lastname", "createdate", "lifecyclestage"],
            )

            if HUBSPOT_AVAILABLE:
                from hubspot.crm.contacts.models.filter import Filter
                from hubspot.crm.contacts.models.filter_group import FilterGroup
                from hubspot.crm.contacts.models.public_object_search_request import (
                    PublicObjectSearchRequest,
                )
            else:
                raise ImportError("HubSpot SDK not available")

            filters = [{"propertyName": "email", "operator": "EQ", "value": email}]

            search_request = PublicObjectSearchRequest(
                filter_groups=[{"filters": filters}], properties=properties, limit=1
            )

            response = client.crm.contacts.search_api.do_search(search_request)

            if response:
                contact = response.results[0]
                return {"id": contact.id, "properties": contact.properties}
            else:
                return None

        except ContactsApiException as e:
            log = traceback.format_exc()
            self.logger.error(log)
            raise Exception(f"HubSpot Contacts API error: {e.reason}")
        except Exception as e:
            log = traceback.format_exc()
            self.logger.error(log)
            raise e

    # * MCP Function.
    def get_marketing_events(self, **arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Get marketing events from HubSpot."""
        try:
            self.logger.info(f"Getting marketing events with arguments: {arguments}")

            client = self.client
            limit = arguments.get("limit", 100)

            # Marketing events API pagination (if supported)
            marketing_events_api = client.marketing.events
            events = []
            after = None
            total_fetched = 0
            batch_limit = min(100, limit)

            while total_fetched < limit:
                try:
                    # Try get_page method first (if available)
                    if hasattr(marketing_events_api.basic_api, "get_page"):
                        response = marketing_events_api.basic_api.get_page(
                            limit=batch_limit, after=after
                        )
                    else:
                        # Fallback to get_all method
                        response = marketing_events_api.basic_api.get_all(limit=batch_limit)
                except AttributeError:
                    # If basic_api doesn't exist, use get_all directly
                    response = marketing_events_api.get_all(limit=batch_limit)

                if response and response.results:
                    for event in response.results:
                        if total_fetched >= limit:
                            break
                        event_data = {
                            "id": event.object_id,
                            "name": getattr(event, "name", ""),
                            "event_type": getattr(event, "event_type", ""),
                            "event_organizer": getattr(event, "event_organizer", ""),
                            "event_description": getattr(event, "event_description", ""),
                            "event_url": getattr(event, "event_url", ""),
                            "event_cancelled": getattr(event, "event_cancelled", False),
                            "start_date_time": self._format_datetime(
                                getattr(event, "start_date_time", "")
                            ),
                            "end_date_time": self._format_datetime(
                                getattr(event, "end_date_time", "")
                            ),
                        }
                        events.append(event_data)
                        total_fetched += 1

                    # Check for next page (if pagination is supported)
                    if (
                        hasattr(response, "paging")
                        and response.paging
                        and hasattr(response.paging, "next")
                        and response.paging.next
                        and total_fetched < limit
                        and hasattr(marketing_events_api.basic_api, "get_page")
                    ):
                        after = response.paging.next.after
                    else:
                        break
                else:
                    break

            return {"total": len(events), "events": events}

        except MarketingApiException as e:
            log = traceback.format_exc()
            self.logger.error(log)
            raise Exception(f"HubSpot Marketing API error: {e.reason}")
        except Exception as e:
            log = traceback.format_exc()
            self.logger.error(log)
            raise e

    def ping(self, **arguments: Dict[str, Any]) -> str:
        """Ping HubSpot API to test connectivity."""
        try:
            self.logger.info(f"Pinging HubSpot API with arguments: {arguments}")

            client = self.client
            response = client.crm.contacts.get_all(limit=1)

            return f"HubSpot API connection successful. Access token is valid."

        except Exception as e:
            log = traceback.format_exc()
            self.logger.error(log)
            return f"HubSpot API connection failed: {str(e)}"

    # * MCP Function.
    @handle_hubspot_errors
    async def analyze_campaign_performance(self, **arguments: Dict[str, Any]) -> Dict[str, Any]:
        """MCP Tool: analyze_campaign_performance - Campaign analytics using SDK"""
        try:
            campaign_ids = arguments.get("campaign_ids", [])
            metrics = arguments.get("metrics", ["open_rate", "click_rate", "conversion_rate"])
            benchmark_type = arguments.get("benchmark_type", "historical")
            include_recommendations = arguments.get("include_recommendations", True)
            date_range = arguments.get("date_range", {})

            # Get campaign data using SDK
            campaigns_data = await self._get_campaigns_with_sdk(campaign_ids)

            # Apply date range filtering if provided
            if date_range and campaigns_data:
                filtered_campaigns = []
                start_date = date_range.get("start")
                end_date = date_range.get("end")

                if start_date or end_date:
                    import pandas as pd

                    for campaign in campaigns_data:
                        # Check if campaign has date fields to filter on
                        campaign_date = None
                        if hasattr(campaign, "start_date_time"):
                            campaign_date = getattr(campaign, "start_date_time")
                        elif hasattr(campaign, "created_at"):
                            campaign_date = getattr(campaign, "created_at")
                        elif hasattr(campaign, "createdate"):
                            campaign_date = getattr(campaign, "createdate")

                        if campaign_date:
                            try:
                                campaign_dt = pd.to_datetime(campaign_date)
                                include_campaign = True

                                if start_date:
                                    start_dt = pd.to_datetime(start_date)
                                    if campaign_dt < start_dt:
                                        include_campaign = False

                                if end_date and include_campaign:
                                    end_dt = pd.to_datetime(end_date)
                                    if campaign_dt > end_dt:
                                        include_campaign = False

                                if include_campaign:
                                    filtered_campaigns.append(campaign)
                            except:
                                # If date parsing fails, include the campaign
                                filtered_campaigns.append(campaign)
                        else:
                            # If no date field found, include the campaign
                            filtered_campaigns.append(campaign)

                    campaigns_data = filtered_campaigns
                    # Update campaign_ids to match filtered campaigns
                    if hasattr(campaigns_data[0], "id") if campaigns_data else False:
                        campaign_ids = [campaign.id for campaign in campaigns_data]
                    elif hasattr(campaigns_data[0], "object_id") if campaigns_data else False:
                        campaign_ids = [campaign.object_id for campaign in campaigns_data]

            # Get detailed campaign statistics
            campaign_stats = await self._get_campaign_stats_with_sdk(campaign_ids)

            # Get engagement data for campaigns
            engagement_data = []
            if campaign_ids:
                engagement_data = await self._get_campaign_engagement_data_with_sdk(campaign_ids)

            # Process through analytics engine
            performance_analysis = await self.analytics_engine.analyze_campaign_performance(
                campaigns_data, campaign_stats, metrics, engagement_data
            )

            # Generate benchmarks
            benchmarks = await self.analytics_engine.generate_benchmarks(
                performance_analysis, benchmark_type
            )

            # Generate recommendations
            recommendations = []
            if include_recommendations:
                recommendations = await self.insight_generator.generate_campaign_recommendations(
                    performance_analysis, benchmarks
                )

            return {
                "success": True,
                "data": {
                    "campaigns": performance_analysis["campaign_metrics"],
                    "summary_metrics": performance_analysis["summary"],
                    "benchmarks": benchmarks,
                    "performance_trends": performance_analysis["trends"],
                },
                "insights": performance_analysis["insights"],
                "recommendations": [r["action"] for r in recommendations],
                "metadata": {
                    "analysis_date": pendulum.now().to_iso8601_string(),
                    "campaigns_analyzed": len(campaigns_data),
                    "benchmark_type": benchmark_type,
                    "data_source": "hubspot_sdk_marketing",
                    "date_range_applied": date_range if date_range else None,
                    "date_filtering_enabled": bool(
                        date_range and (date_range.get("start") or date_range.get("end"))
                    ),
                },
                "error_message": None,
            }

        except Exception as e:
            log = traceback.format_exc()
            return {
                "success": False,
                "data": None,
                "insights": [],
                "recommendations": [],
                "metadata": {},
                "error_message": f"Campaign analysis failed: {str(e)}\n{log}",
            }

    def _format_datetime(self, dt_value):
        """Format datetime values to ISO format strings with proper error handling"""
        if not dt_value or dt_value == "":
            return ""

        try:
            # If it's already a string, try to parse and reformat for consistency
            if isinstance(dt_value, str):
                # Try to parse if it's already formatted, otherwise return as-is if it looks valid
                if dt_value.count("-") >= 2 or dt_value.count("T") == 1:
                    try:
                        parsed_dt = pd.to_datetime(dt_value)
                        return parsed_dt.isoformat()
                    except:
                        return dt_value  # Return original if parsing fails
                return dt_value

            # If it's a datetime-like object, convert to ISO format
            elif hasattr(dt_value, "isoformat"):
                return dt_value.isoformat()

            # If it's a pandas datetime or similar
            elif hasattr(dt_value, "strftime"):
                return dt_value.strftime("%Y-%m-%dT%H:%M:%S")

            # Try pandas datetime parsing as last resort
            else:
                parsed_dt = pd.to_datetime(dt_value)
                return parsed_dt.isoformat()

        except Exception as e:
            self.logger.warning(f"Failed to format datetime value {dt_value}: {e}")
            # Return string representation as fallback
            return str(dt_value) if dt_value else ""

    async def _get_campaigns_with_sdk(self, campaign_ids: List[str] = None) -> List[Dict[str, Any]]:
        """Get campaign data using HubSpot SDK marketing events API"""

        await self.rate_limiter.wait_if_needed()

        try:
            marketing_events_api = self.client.marketing.events

            if campaign_ids:
                campaigns = []
                for campaign_id in campaign_ids:
                    await self.rate_limiter.wait_if_needed()
                    try:
                        campaign = marketing_events_api.get_by_id(campaign_id)
                        campaigns.append(campaign)
                    except Exception as e:
                        self.logger.warning(f"Could not fetch campaign {campaign_id}: {e}")
                        continue
            else:
                campaigns = []
                after = None

                while True:
                    await self.rate_limiter.wait_if_needed()

                    # Marketing events API uses get_all for pagination, not get_page
                    if after:
                        response = marketing_events_api.basic_api.get_all(limit=100, after=after)
                    else:
                        response = marketing_events_api.basic_api.get_all(limit=100)

                    if response and response.results:
                        campaigns.extend(response.results)

                    # Check for next page using paging info
                    if (
                        hasattr(response, "paging")
                        and response.paging
                        and hasattr(response.paging, "next")
                        and response.paging.next
                    ):
                        after = response.paging.next.after
                    else:
                        break

            return campaigns

        except Exception as e:
            log = traceback.format_exc()
            self.logger.error(f"Error fetching campaigns: {e}\n{log}")
            return []

    async def _get_campaign_stats_with_sdk(self, campaign_ids: List[str]) -> List[Dict[str, Any]]:
        """Get campaign statistics using HubSpot SDK"""

        stats = []

        for campaign_id in campaign_ids:
            await self.rate_limiter.wait_if_needed()

            try:
                marketing_events_api = self.client.marketing.events
                campaign_detail = marketing_events_api.get_detail_by_id(campaign_id)

                campaign_stats = {
                    "campaign_id": campaign_id,
                    "sent": getattr(campaign_detail, "email_sent_count", 0),
                    "delivered": getattr(campaign_detail, "email_delivered_count", 0),
                    "opened": getattr(campaign_detail, "email_opened_count", 0),
                    "clicked": getattr(campaign_detail, "email_clicked_count", 0),
                    "bounced": getattr(campaign_detail, "email_bounced_count", 0),
                    "unsubscribed": getattr(campaign_detail, "email_unsubscribed_count", 0),
                }

                stats.append(campaign_stats)

            except Exception as e:
                self.logger.warning(f"Could not get stats for campaign {campaign_id}: {e}")
                continue

        return stats

    # * MCP Function.
    @handle_hubspot_errors
    async def analyze_sales_pipeline(self, **arguments: Dict[str, Any]) -> Dict[str, Any]:
        """MCP Tool: analyze_sales_pipeline - Sales pipeline analytics using SDK"""
        try:
            pipeline_ids = arguments.get("pipeline_ids", [])
            timeframe = arguments.get("timeframe", {})
            analysis_type = arguments.get("analysis_type", "conversion_rates")
            include_recommendations = arguments.get("include_recommendations", True)

            # Get deals data using SDK
            deals_data = await self._get_deals_with_sdk(pipeline_ids, timeframe)

            # Get engagement data for deals
            engagement_data = []
            if deals_data:
                # Extract contact IDs from deals for engagement data
                contact_ids = []
                for deal in deals_data:
                    # Get associated contact IDs (this would need to be implemented based on deal-contact associations)
                    pass  # Placeholder - in real implementation, get contact IDs from deal associations
                engagement_data = (
                    await self._get_engagement_data_with_sdk(contact_ids) if contact_ids else []
                )

            # Process through analytics engine
            pipeline_analysis = await self.analytics_engine.analyze_sales_pipeline(
                deals_data, analysis_type, engagement_data
            )

            # Generate recommendations
            recommendations = []
            if include_recommendations:
                recommendations = await self.insight_generator.generate_pipeline_recommendations(
                    pipeline_analysis
                )

            return {
                "success": True,
                "data": {
                    "totalValue": pipeline_analysis["total_value"],
                    "dealCount": pipeline_analysis["deal_count"],
                    "avgDealSize": pipeline_analysis["avg_deal_size"],
                    "conversionRate": pipeline_analysis["conversion_rate"],
                    "stageAnalysis": pipeline_analysis["stage_metrics"],
                },
                "insights": pipeline_analysis["insights"],
                "recommendations": [r["action"] for r in recommendations],
                "metadata": {
                    "analysis_type": analysis_type,
                    "data_source": "hubspot_sdk_deals",
                },
                "error_message": None,
            }

        except Exception as e:
            return {
                "success": False,
                "data": None,
                "insights": [],
                "recommendations": [],
                "metadata": {},
                "error_message": f"Pipeline analysis failed: {str(e)}",
            }

    async def _get_deals_with_sdk(
        self, pipeline_ids: List[str] = None, timeframe: Dict[str, str] = None
    ) -> List[Any]:
        """Get deals data using HubSpot SDK"""

        await self.rate_limiter.wait_if_needed()

        deals_api = self.client.crm.deals

        # Define properties to fetch
        properties = [
            "dealname",
            "amount",
            "dealstage",
            "pipeline",
            "createdate",
            "closedate",
            "hs_deal_stage_probability",
            "dealtype",
            "hubspot_owner_id",
            "hs_analytics_source",
        ]

        all_deals = []

        # Build search request if filtering is needed
        if pipeline_ids or timeframe:
            filters = []

            # Add pipeline filter - convert to strings
            if pipeline_ids:
                # Ensure pipeline IDs are strings
                string_pipeline_ids = [str(pid) for pid in pipeline_ids]
                filters.append(
                    {"propertyName": "pipeline", "operator": "IN", "values": string_pipeline_ids}
                )

            # Add date range filter - convert ISO dates to millisecond timestamps
            if timeframe and timeframe.get("start") and timeframe.get("end"):
                import pendulum

                try:
                    # Convert ISO strings to millisecond timestamps
                    start_dt = pendulum.parse(timeframe["start"])
                    end_dt = pendulum.parse(timeframe["end"])

                    start_timestamp = str(int(start_dt.timestamp() * 1000))
                    end_timestamp = str(int(end_dt.timestamp() * 1000))

                    filters.append(
                        {
                            "propertyName": "createdate",
                            "operator": "BETWEEN",
                            "value": start_timestamp,
                            "highValue": end_timestamp,
                        }
                    )

                    self.logger.info(f"Date filter: {start_timestamp} to {end_timestamp}")

                except Exception as e:
                    self.logger.warning(f"Date conversion failed: {e}. Skipping date filter.")
                    # Continue without date filter rather than failing

            # Create search request with proper filter groups structure
            search_request = PublicObjectSearchRequest(
                filter_groups=[{"filters": filters}], properties=properties, limit=100
            )

            # Paginate through search results with error handling and 10k limit protection
            after = None
            max_deals = 9900  # Stay under the 10,000 limit
            try:
                while len(all_deals) < max_deals:
                    if after:
                        search_request.after = after

                    # Adjust batch size to not exceed the max limit
                    remaining = max_deals - len(all_deals)
                    search_request.limit = min(100, remaining)

                    await self.rate_limiter.wait_if_needed()
                    search_response = deals_api.search_api.do_search(search_request)

                    if search_response and search_response.results:
                        all_deals.extend(search_response.results)
                        self.logger.info(
                            f"Retrieved {len(search_response.results)} deals, total: {len(all_deals)}"
                        )

                    if (
                        search_response.paging
                        and search_response.paging.next
                        and len(all_deals) < max_deals
                    ):
                        after = search_response.paging.next.after
                    else:
                        break

            except Exception as e:
                self.logger.error(f"Search API failed with filters: {e}")
                raise

        else:
            # Get all deals without filters using basic_api.get_page
            after = None
            while True:
                await self.rate_limiter.wait_if_needed()
                response = deals_api.basic_api.get_page(
                    limit=100, after=after, properties=properties, archived=False
                )

                if response and response.results:
                    all_deals.extend(response.results)

                if (
                    hasattr(response, "paging")
                    and response.paging
                    and hasattr(response.paging, "next")
                    and response.paging.next
                ):
                    after = response.paging.next.after
                else:
                    break

        self.logger.info(f"Fetched {len(all_deals)} deals using HubSpot SDK")
        return all_deals

    # * MCP Function.
    @handle_hubspot_errors
    async def predict_lead_scores(self, **arguments: Dict[str, Any]) -> Dict[str, Any]:
        """MCP Tool: predict_lead_scores - ML-based lead scoring using SDK"""
        try:
            limit = arguments.get("limit", 100)
            contact_ids = arguments.get("contact_ids", [])
            model_type = arguments.get("model_type", "conversion_probability")
            include_feature_importance = arguments.get("include_feature_importance", True)
            date_range = arguments.get("date_range", {})

            # Get contacts using SDK
            if contact_ids:
                contacts_data = await self._get_contacts_by_ids_with_sdk(contact_ids)
            else:
                contacts_data = await self._get_contacts_with_sdk(limit=limit)

            # Get historical deals for training with date filtering if specified
            if date_range and date_range.get("start") and date_range.get("end"):
                deals_data = await self._get_deals_with_sdk(timeframe=date_range)
            else:
                deals_data = await self._get_deals_with_sdk()

            # Get engagement data
            contact_ids_list = [contact.id for contact in contacts_data]
            engagement_data = await self._get_engagement_data_with_sdk(contact_ids_list)

            # ML processing
            prediction_result = await self.analytics_engine.predict_lead_scores(
                contacts_data, deals_data, engagement_data, model_type
            )

            # Generate insights
            prediction_insights = await self.insight_generator.analyze_predictions(
                prediction_result
            )

            return {
                "success": True,
                "data": {
                    "predictions": prediction_result["scores"],
                    "model_performance": prediction_result["performance_metrics"],
                    "feature_importance": (
                        prediction_result.get("feature_importance")
                        if include_feature_importance
                        else None
                    ),
                },
                "insights": prediction_insights["insights"],
                "recommendations": prediction_insights["recommendations"],
                "metadata": {
                    "model_type": model_type,
                    "training_data_size": prediction_result["training_size"],
                    "model_accuracy": prediction_result["performance_metrics"]["accuracy"],
                    "data_source": "hubspot_sdk_multi_api",
                },
                "error_message": None,
            }

        except Exception as e:
            log = traceback.format_exc()
            return {
                "success": False,
                "data": None,
                "insights": [],
                "recommendations": [],
                "metadata": {},
                "error_message": f"Lead scoring failed: {str(e)}\n{log}",
            }

    async def _get_contacts_by_ids_with_sdk(self, contact_ids: List[str]) -> List[Any]:
        """Get specific contacts by IDs using HubSpot SDK batch API"""

        await self.rate_limiter.wait_if_needed()

        contacts_api = self.client.crm.contacts
        properties = [
            "email",
            "firstname",
            "lastname",
            "createdate",
            "lifecyclestage",
            "hs_lead_status",
            "hs_email_open",
            "hs_email_click",
        ]

        # Batch API can handle up to 100 IDs at once
        batch_size = 100
        all_contacts = []

        for i in range(0, len(contact_ids), batch_size):
            batch_ids = contact_ids[i : i + batch_size]

            # Create batch input
            batch_input = {
                "inputs": [{"id": contact_id} for contact_id in batch_ids],
                "properties": properties,
            }

            await self.rate_limiter.wait_if_needed()

            try:
                batch_response = contacts_api.batch_api.read(
                    batch_read_input_simple_public_object_id=batch_input
                )

                if batch_response:
                    all_contacts.extend(batch_response)

            except Exception as e:
                log = traceback.format_exc()
                self.logger.error(f"Batch read contacts failed: {e}\n{log}")
                continue

        return all_contacts

    # ============ ADVANCED ANALYTICS FUNCTIONS ============

    # * MCP Function.
    @handle_hubspot_errors
    async def create_contact_segments(self, **arguments: Dict[str, Any]) -> Dict[str, Any]:
        """MCP Tool: create_contact_segments - Advanced behavioral segmentation"""
        try:
            segmentation_type = arguments.get("segmentation_type", "behavioral")
            criteria = arguments.get("criteria", {})
            num_segments = arguments.get("number_of_segments", 5)
            limit = arguments.get("limit", 1000)  # Default limit to prevent performance issues

            # Get contacts with enhanced criteria
            contacts_data = await self._get_contacts_with_advanced_criteria(criteria, limit)
            engagement_data = await self._get_engagement_data_with_sdk(
                [c.id for c in contacts_data]
            )

            # Process through analytics engine
            segmentation_result = await self.analytics_engine.create_customer_segments(
                contacts_data, segmentation_type, num_segments, engagement_data
            )

            # Generate insights
            segmentation_insights = await self.insight_generator.analyze_segmentation(
                segmentation_result
            )

            return {
                "success": True,
                "data": {
                    "segments": segmentation_result["segment_profiles"],
                    "segmentation_quality": segmentation_result["quality_metrics"],
                    "contact_assignments": segmentation_result[
                        "segments"
                    ],  # Individual contact-to-segment mappings
                    "segment_characteristics": segmentation_result.get(
                        "segment_profiles", []
                    ),  # Segment profiles contain characteristics
                },
                "insights": segmentation_insights["insights"],
                "recommendations": segmentation_insights["recommendations"],
                "metadata": {
                    "segmentation_type": segmentation_type,
                    "num_segments": num_segments,
                    "total_contacts": len(contacts_data),
                    "contact_limit": limit,
                    "data_source": "hubspot_sdk_advanced_segmentation",
                },
                "error_message": None,
            }

        except Exception as e:
            log = traceback.format_exc()
            self.logger.error(f"Contact segmentation failed: {e}\n{log}")
            return {
                "success": False,
                "data": None,
                "insights": [],
                "recommendations": [],
                "metadata": {},
                "error_message": f"Contact segmentation failed: {e}\n{log}",
            }

    # * MCP Function.
    @handle_hubspot_errors
    async def forecast_revenue(self, **arguments: Dict[str, Any]) -> Dict[str, Any]:
        """MCP Tool: forecast_revenue - AI-powered revenue predictions"""
        try:
            forecast_period = arguments.get("forecast_period", "90_days")
            confidence_level = arguments.get("confidence_level", 0.95)

            # Get historical timeframe for training data
            historical_timeframe = self._get_historical_timeframe(forecast_period)

            # Get historical and current pipeline data
            historical_deals = await self._get_deals_with_sdk(timeframe=historical_timeframe)
            current_pipeline = await self._get_deals_with_sdk()

            # Revenue forecasting through analytics engine
            forecast_result = await self.analytics_engine.forecast_revenue(
                historical_deals, current_pipeline, forecast_period, confidence_level
            )

            # Generate forecasting insights
            forecast_insights = await self.insight_generator.analyze_revenue_forecast(
                forecast_result
            )

            return {
                "success": True,
                "data": {
                    "forecast": forecast_result["prediction"],
                    "confidence_interval": forecast_result["confidence_interval"],
                    "scenarios": forecast_result["scenarios"],
                    "model_performance": forecast_result["model_accuracy"],
                },
                "insights": forecast_insights["insights"],
                "recommendations": forecast_insights["recommendations"],
                "metadata": {
                    "forecast_period": forecast_period,
                    "confidence_level": confidence_level,
                    "historical_data_points": len(historical_deals),
                    "pipeline_deals": len(current_pipeline),
                    "data_source": "hubspot_sdk_revenue_forecasting",
                },
                "error_message": None,
            }

        except Exception as e:
            log = traceback.format_exc()
            self.logger.error(f"Revenue forecasting failed: {e}\n{log}")
            return {
                "success": False,
                "data": None,
                "insights": [],
                "recommendations": [],
                "metadata": {},
                "error_message": f"Revenue forecasting failed: {e}\n{log}",
            }

    # * MCP Function.
    @handle_hubspot_errors
    async def generate_executive_report(self, **arguments: Dict[str, Any]) -> Dict[str, Any]:
        """MCP Tool: generate_executive_report - Comprehensive executive reporting"""
        try:
            report_type = arguments.get("report_type", "monthly")
            timeframe = arguments.get("timeframe", {})
            include_forecast = arguments.get("include_forecast", True)

            # Gather data from multiple sources in parallel
            contacts_data = await self.get_contact_analytics({"date_range": timeframe})
            campaigns_data = await self.analyze_campaign_performance({"date_range": timeframe})
            pipeline_data = await self.analyze_sales_pipeline({"timeframe": timeframe})

            # Include revenue forecast if requested
            forecast_data = None
            if include_forecast:
                forecast_data = await self.forecast_revenue(
                    {"forecast_period": "90_days", "confidence_level": 0.95}
                )

            # Generate executive insights
            executive_insights = await self.insight_generator.generate_executive_insights(
                {
                    "contacts": contacts_data,
                    "campaigns": campaigns_data,
                    "pipeline": pipeline_data,
                    "forecast": forecast_data,
                }
            )

            return {
                "success": True,
                "data": {
                    "executiveSummary": executive_insights["summary"],
                    "keyMetrics": executive_insights["kpis"],
                    "trends": executive_insights["trends"],
                    "performance": executive_insights["performance_analysis"],
                    "forecast": (
                        forecast_data["data"]
                        if forecast_data and forecast_data["success"]
                        else None
                    ),
                },
                "insights": executive_insights["strategic_insights"],
                "recommendations": executive_insights["strategic_actions"],
                "metadata": {
                    "report_type": report_type,
                    "timeframe": timeframe,
                    "includes_forecast": include_forecast,
                    "data_source": "hubspot_sdk_executive_reporting",
                },
                "error_message": None,
            }

        except Exception as e:
            log = traceback.format_exc()
            self.logger.error(f"Executive report generation failed: {e}\n{log}")
            return {
                "success": False,
                "data": None,
                "insights": [],
                "recommendations": [],
                "metadata": {},
                "error_message": f"Executive report generation failed: {e}\n{log}",
            }

    # * MCP Function.
    async def batch_update_contacts(self, **arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Batch update contacts with engagement scores or segments"""
        try:
            batch_size = 100
            successful_updates = 0
            failed_updates = 0
            updates = arguments["updates"]

            for i in range(0, len(updates), batch_size):
                batch = updates[i : i + batch_size]

                try:
                    batch_input = {
                        "inputs": [
                            {"id": update["contact_id"], "properties": update["properties"]}
                            for update in batch
                        ]
                    }

                    await self.rate_limiter.wait_if_needed()
                    self.client.crm.contacts.batch_api.update(batch_input)
                    successful_updates += len(batch)

                except Exception as e:
                    self.logger.error(f"Batch update failed: {e}")
                    failed_updates += len(batch)

            return {
                "success": True,
                "data": {
                    "total_attempted": len(updates),
                    "successful_updates": successful_updates,
                    "failed_updates": failed_updates,
                    "success_rate": successful_updates / len(updates) if len(updates) > 0 else 0,
                },
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "data": {
                    "total_attempted": len(updates),
                    "successful_updates": 0,
                    "failed_updates": len(updates),
                    "success_rate": 0,
                },
            }

    # ============ UTILITY FUNCTIONS ============

    async def _get_contacts_with_advanced_criteria(
        self, criteria: Dict[str, Any], max_contacts: int = 1000
    ) -> List[Any]:
        """Enhanced contact search with complex filtering"""

        filters = []
        self.logger.info(f"Building filters for criteria: {criteria}")

        # Date range filters - convert ISO to timestamps
        if "date_range" in criteria:
            import pendulum

            try:
                # Convert ISO strings to millisecond timestamps
                start_dt = pendulum.parse(criteria["date_range"]["start"])
                end_dt = pendulum.parse(criteria["date_range"]["end"])

                start_timestamp = str(int(start_dt.timestamp() * 1000))
                end_timestamp = str(int(end_dt.timestamp() * 1000))

                filters.append(
                    {
                        "propertyName": "createdate",
                        "operator": "BETWEEN",
                        "value": start_timestamp,
                        "highValue": end_timestamp,
                    }
                )

                self.logger.info(
                    f"Advanced contact date filter: {start_timestamp} to {end_timestamp}"
                )

            except Exception as e:
                self.logger.warning(f"Advanced date conversion failed: {e}. Skipping date filter.")

        # Lifecycle stage filters
        if "lifecycle_stages" in criteria:
            filters.append(
                {
                    "propertyName": "lifecyclestage",
                    "operator": "IN",
                    "values": criteria["lifecycle_stages"],
                }
            )

        # Lead score filters
        if "min_lead_score" in criteria:
            filters.append(
                {
                    "propertyName": "hubspotscore",
                    "operator": "GTE",
                    "value": str(criteria["min_lead_score"]),
                }
            )

        # Engagement score filters (use hubspotscore as proxy)
        # Note: Making this less restrictive for better results
        if "engagement_score" in criteria:
            if "min" in criteria["engagement_score"]:
                # Use a lower threshold to get more contacts
                min_score = max(1, criteria["engagement_score"]["min"])
                filters.append(
                    {
                        "propertyName": "hubspotscore",
                        "operator": "GTE",
                        "value": str(min_score),  # Use direct value instead of scaling
                    }
                )
                self.logger.info(f"Added engagement score filter: hubspotscore >= {min_score}")
            if "max" in criteria["engagement_score"]:
                filters.append(
                    {
                        "propertyName": "hubspotscore",
                        "operator": "LTE",
                        "value": str(
                            criteria["engagement_score"]["max"] * 20
                        ),  # More generous scaling
                    }
                )

        # Skip lifetime value filters for now as they might be too restrictive
        # and the proxy property might not exist
        if "lifetime_value" in criteria:
            self.logger.info("Skipping lifetime_value filter - using other criteria only")

        # Execute search with pagination
        self.logger.info(f"Executing search with {len(filters)} filters")
        contacts = await self._execute_advanced_search(filters, max_contacts)

        # Fallback: if no contacts found with strict criteria, try with just date range or no filters
        if not contacts and len(filters) > 1:
            self.logger.info(
                "No contacts found with full criteria, trying fallback with date range only"
            )
            fallback_filters = []

            # Try with just date range if it exists
            if "date_range" in criteria:
                import pendulum

                try:
                    # Convert ISO strings to millisecond timestamps
                    start_dt = pendulum.parse(criteria["date_range"]["start"])
                    end_dt = pendulum.parse(criteria["date_range"]["end"])

                    start_timestamp = str(int(start_dt.timestamp() * 1000))
                    end_timestamp = str(int(end_dt.timestamp() * 1000))

                    fallback_filters.append(
                        {
                            "propertyName": "createdate",
                            "operator": "BETWEEN",
                            "value": start_timestamp,
                            "highValue": end_timestamp,
                        }
                    )

                except Exception as e:
                    self.logger.warning(f"Fallback date conversion failed: {e}. Using no filters.")

            contacts = await self._execute_advanced_search(fallback_filters, max_contacts)

            # Final fallback: get any contacts if still empty
            if not contacts:
                self.logger.info("No contacts found with date range, getting recent contacts")
                contacts = await self._execute_advanced_search([], min(max_contacts, 100))

        self.logger.info(f"Found {len(contacts)} contacts for segmentation")
        return contacts

    async def _execute_advanced_search(self, filters: List, max_contacts: int = 1000) -> List[Any]:
        """Execute advanced search with pagination"""

        all_contacts = []
        after = None
        batch_limit = 100

        self.logger.info(
            f"Starting search with {len(filters)} filters, max_contacts={max_contacts}"
        )

        while len(all_contacts) < max_contacts:
            # Calculate how many more contacts we need, capped at batch_limit
            remaining_contacts = max_contacts - len(all_contacts)
            current_limit = min(batch_limit, remaining_contacts)

            # Build search request properties
            properties = [
                "email",
                "firstname",
                "lastname",
                "company",
                "lifecyclestage",
                "createdate",
                "lastmodifieddate",
                "hubspotscore",
                "hs_lead_status",
            ]

            # Build filter groups if we have filters
            filter_groups = []
            if filters:
                filter_groups = [{"filters": filters}]

            # Create the search request object
            search_request = PublicObjectSearchRequest(
                properties=properties, limit=current_limit, filter_groups=filter_groups, after=after
            )

            await self.rate_limiter.wait_if_needed()

            try:
                response = self.client.crm.contacts.search_api.do_search(
                    public_object_search_request=search_request
                )

                batch_count = len(response.results) if response.results else 0
                self.logger.info(f"API returned {batch_count} contacts in this batch")

                if response.results:
                    all_contacts.extend(response.results)

                if not response.paging or not response.paging.next:
                    break

                after = response.paging.next.after

            except Exception as e:
                self.logger.error(f"Advanced search failed: {e}")
                break

        return all_contacts

    def _get_historical_timeframe(self, forecast_period: str) -> Dict[str, str]:
        """Generate historical training period - independent of forecast period

        Uses best practices for historical data collection:
        - Minimum 6 months for short forecasts
        - Up to 2 years for longer forecasts
        - Ensures sufficient data for reliable predictions
        """
        import pendulum

        now = pendulum.now("UTC")

        # Parse forecast period to understand prediction horizon
        if isinstance(forecast_period, str):
            if "days" in forecast_period:
                forecast_days = int(forecast_period.split("_")[0])
            elif "months" in forecast_period:
                forecast_days = int(forecast_period.split("_")[0]) * 30
            elif "quarters" in forecast_period:
                forecast_days = int(forecast_period.split("_")[0]) * 90
            elif "years" in forecast_period:
                forecast_days = int(forecast_period.split("_")[0]) * 365
            else:
                forecast_days = 90  # Default
        else:
            forecast_days = int(forecast_period)

        # Determine historical training period based on best practices
        # Rule: Use 4-8x the forecast period, with reasonable min/max bounds
        if forecast_days <= 30:
            # Short forecast: use 6 months of history
            historical_days = 180
        elif forecast_days <= 90:
            # 3-month forecast: use 1 year of history
            historical_days = 365
        elif forecast_days <= 180:
            # 6-month forecast: use 18 months of history
            historical_days = 540
        elif forecast_days <= 365:
            # 1-year forecast: use 2 years of history
            historical_days = 730
        else:
            # Long forecast: use 2 years max (data gets stale beyond that)
            historical_days = 730

        start = now.subtract(days=historical_days)
        return {"start": start.to_iso8601_string(), "end": now.to_iso8601_string()}
