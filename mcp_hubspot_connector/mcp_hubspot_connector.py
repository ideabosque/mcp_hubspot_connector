#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

__author__ = "bibow"

import logging
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional

# Try to import the official HubSpot SDK
try:
    from hubspot import HubSpot
    from hubspot.crm.contacts import ApiException as ContactsApiException
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


class MCPHubspotConnector:
    def __init__(self, logger: logging.Logger, **setting: Dict[str, Any]):
        self.logger = logger
        self.setting = setting

        # Initialize HubSpot client immediately
        if not HUBSPOT_AVAILABLE:
            raise ImportError(
                "HubSpot SDK not available. Please install the official HubSpot SDK:\n"
                "pip install hubspot-api-client\n"
                "or\n"
                "pip install simplejson  # if using the legacy hubspot package"
            )

        access_token = self.setting.get("hubspot_access_token")
        if not access_token:
            raise ValueError("hubspot_access_token is required in settings")

        self.client = HubSpot(access_token=access_token)

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

            response = client.crm.contacts.get_all(
                properties=properties, limit=min(limit, 100)
            )

            contacts = []
            if response:
                for contact in response:
                    contact_data = {"id": contact.id, "properties": contact.properties}
                    contacts.append(contact_data)

            return {"total": len(contacts), "contacts": contacts}

        except ContactsApiException as e:
            log = traceback.format_exc()
            self.logger.error(log)
            raise Exception(f"HubSpot Contacts API error: {e.reason}")
        except Exception as e:
            log = traceback.format_exc()
            self.logger.error(log)
            raise e

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
            response = client.crm.contacts.create(
                simple_public_object_input=contact_input
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

            response = client.crm.deals.get_all(
                properties=properties, limit=min(limit, 100)
            )

            deals = []
            if response.results:
                for deal in response.results:
                    deal_data = {"id": deal.id, "properties": deal.properties}
                    deals.append(deal_data)

            return {"total": len(deals), "deals": deals}

        except DealsApiException as e:
            log = traceback.format_exc()
            self.logger.error(log)
            raise Exception(f"HubSpot Deals API error: {e.reason}")
        except Exception as e:
            log = traceback.format_exc()
            self.logger.error(log)
            raise e

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

            response = client.crm.companies.get_all(
                properties=properties, limit=min(limit, 100)
            )

            companies = []
            if response.results:
                for company in response.results:
                    company_data = {"id": company.id, "properties": company.properties}
                    companies.append(company_data)

            return {"total": len(companies), "companies": companies}

        except Exception as e:
            log = traceback.format_exc()
            self.logger.error(log)
            raise e

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

            filter_group = FilterGroup(
                filters=[
                    Filter(
                        property_name="email", operator="CONTAINS_TOKEN", value=query
                    )
                ]
            )

            search_request = PublicObjectSearchRequest(
                filter_groups=[filter_group],
                properties=properties,
                limit=min(limit, 100),
            )

            response = client.crm.contacts.search_api.do_search(search_request)

            contacts = []
            if response.results:
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

            filter_group = FilterGroup(
                filters=[Filter(property_name="email", operator="EQ", value=email)]
            )

            search_request = PublicObjectSearchRequest(
                filter_groups=[filter_group], properties=properties, limit=1
            )

            response = client.crm.contacts.search_api.do_search(search_request)

            if response.results:
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

    def get_marketing_events(self, **arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Get marketing events from HubSpot."""
        try:
            self.logger.info(f"Getting marketing events with arguments: {arguments}")

            client = self.client
            limit = arguments.get("limit", 100)

            response = client.marketing.events.get_all(limit=min(limit, 100))

            events = []
            if response.results:
                for event in response.results:
                    event_data = {
                        "id": event.id,
                        "name": getattr(event, "name", ""),
                        "event_type": getattr(event, "event_type", ""),
                        "start_date_time": getattr(event, "start_date_time", ""),
                        "end_date_time": getattr(event, "end_date_time", ""),
                    }
                    events.append(event_data)

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
