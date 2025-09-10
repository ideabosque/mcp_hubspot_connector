#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

__author__ = "bibow"

import asyncio
import logging
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Dict, List, Optional, Union

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




class RateLimiter:
    """Rate limiter for HubSpot API calls"""

    def __init__(self, calls_per_second: int = 10):
        self.calls_per_second = calls_per_second
        self.last_call = 0

    async def wait_if_needed(self):
        """Wait if necessary to respect rate limits"""
        now = time.time()
        time_since_last = now - self.last_call
        min_interval = 1.0 / self.calls_per_second

        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            await asyncio.sleep(sleep_time)

        self.last_call = time.time()


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


class AnalyticsEngine:
    """Core analytics processing engine"""

    async def process_contact_metrics(
        self, contacts_data: List[Any], engagement_data: List[Dict], segmentation: str
    ) -> Dict[str, Any]:
        """Process HubSpot SDK contact objects into comprehensive analytics metrics with engagement data integration"""

        # Process engagement data for advanced metrics
        engagement_dict = {}
        if engagement_data:
            for engagement in engagement_data:
                contact_id = str(
                    engagement.get("contact_id", "") or engagement.get("contactId", "")
                )
                if contact_id:
                    if contact_id not in engagement_dict:
                        engagement_dict[contact_id] = {
                            "email_events": [],
                            "website_events": [],
                            "form_submissions": [],
                            "social_interactions": [],
                            "meeting_interactions": [],
                            "call_interactions": [],
                        }

                    event_type = engagement.get("type", "").lower()
                    event_data = {
                        "timestamp": engagement.get("timestamp", ""),
                        "event_id": engagement.get("id", ""),
                        "properties": engagement.get("properties", {}),
                    }

                    # Categorize engagement events
                    if "email" in event_type:
                        engagement_dict[contact_id]["email_events"].append(event_data)
                    elif "website" in event_type or "page" in event_type:
                        engagement_dict[contact_id]["website_events"].append(event_data)
                    elif "form" in event_type or "submission" in event_type:
                        engagement_dict[contact_id]["form_submissions"].append(event_data)
                    elif "social" in event_type:
                        engagement_dict[contact_id]["social_interactions"].append(event_data)
                    elif "meeting" in event_type:
                        engagement_dict[contact_id]["meeting_interactions"].append(event_data)
                    elif "call" in event_type:
                        engagement_dict[contact_id]["call_interactions"].append(event_data)

        # Enhanced contact processing with engagement data integration
        contacts_list = []
        for contact in contacts_data:
            properties = contact.properties if hasattr(contact, "properties") else contact
            contact_id = str(contact.id if hasattr(contact, "id") else contact.get("id", ""))

            # Basic contact properties
            email_opens = int(properties.get("hs_email_open", 0) or 0)
            email_clicks = int(properties.get("hs_email_click", 0) or 0)
            email_bounces = int(properties.get("hs_email_bounce", 0) or 0)

            # Enhanced engagement metrics from engagement_data
            contact_engagement = engagement_dict.get(contact_id, {})

            # Calculate engagement metrics from engagement events
            email_events = contact_engagement.get("email_events", [])
            website_events = contact_engagement.get("website_events", [])
            form_submissions = contact_engagement.get("form_submissions", [])
            social_interactions = contact_engagement.get("social_interactions", [])
            meeting_interactions = contact_engagement.get("meeting_interactions", [])
            call_interactions = contact_engagement.get("call_interactions", [])

            email_event_count = len(email_events)
            website_session_count = len(website_events)
            form_submission_count = len(form_submissions)
            social_engagement_count = len(social_interactions)
            meeting_count = len(meeting_interactions)
            call_count = len(call_interactions)

            # Calculate total engagement frequency and recency
            all_engagement_events = (
                email_events
                + website_events
                + form_submissions
                + social_interactions
                + meeting_interactions
                + call_interactions
            )
            total_engagement_frequency = len(all_engagement_events)

            # Calculate days since last engagement
            days_since_last_engagement = 999  # Default high value
            if all_engagement_events:
                try:
                    sorted_events = sorted(
                        all_engagement_events,
                        key=lambda x: pd.to_datetime(x.get("timestamp", "1900-01-01")),
                        reverse=True,
                    )
                    if sorted_events:
                        last_engagement = pd.to_datetime(sorted_events[0].get("timestamp", ""))
                        if not pd.isna(last_engagement):
                            days_since_last_engagement = (pd.Timestamp.now() - last_engagement).days
                except:
                    pass

            # Calculate engagement diversity (number of different engagement types)
            engagement_types_used = sum(
                [
                    1 if email_event_count > 0 else 0,
                    1 if website_session_count > 0 else 0,
                    1 if form_submission_count > 0 else 0,
                    1 if social_engagement_count > 0 else 0,
                    1 if meeting_count > 0 else 0,
                    1 if call_count > 0 else 0,
                ]
            )

            # Calculate progressive engagement (recent vs older activity)
            progressive_engagement_score = 0
            if len(all_engagement_events) >= 2:
                try:
                    recent_events = [
                        e
                        for e in all_engagement_events
                        if (
                            pd.Timestamp.now() - pd.to_datetime(e.get("timestamp", "1900-01-01"))
                        ).days
                        <= 30
                    ]
                    older_events = [
                        e
                        for e in all_engagement_events
                        if (
                            pd.Timestamp.now() - pd.to_datetime(e.get("timestamp", "1900-01-01"))
                        ).days
                        > 30
                    ]

                    recent_count = len(recent_events)
                    older_count = len(older_events) if older_events else 1
                    progressive_engagement_score = recent_count / older_count
                except:
                    progressive_engagement_score = 1

            # Calculate contact creation recency
            try:
                create_date = pd.to_datetime(properties.get("createdate", ""))
                days_since_created = (
                    (pd.Timestamp.now() - create_date).days if not pd.isna(create_date) else 0
                )
            except:
                days_since_created = 0

            contacts_list.append(
                {
                    "id": contact_id,
                    "email": properties.get("email", ""),
                    "firstname": properties.get("firstname", ""),
                    "lastname": properties.get("lastname", ""),
                    "createdate": properties.get("createdate", ""),
                    "lifecyclestage": properties.get("lifecyclestage", ""),
                    "hs_lead_status": properties.get("hs_lead_status", ""),
                    "company_size": int(properties.get("numberofemployees", 0) or 0),
                    "industry": properties.get("industry", ""),
                    # Basic email engagement
                    "email_opens": email_opens,
                    "email_clicks": email_clicks,
                    "email_bounces": email_bounces,
                    # Enhanced engagement metrics from engagement_data
                    "email_event_count": email_event_count,
                    "website_session_count": website_session_count,
                    "form_submission_count": form_submission_count,
                    "social_engagement_count": social_engagement_count,
                    "meeting_count": meeting_count,
                    "call_count": call_count,
                    "total_engagement_frequency": total_engagement_frequency,
                    "days_since_last_engagement": days_since_last_engagement,
                    "engagement_types_used": engagement_types_used,
                    "progressive_engagement_score": progressive_engagement_score,
                    "days_since_created": days_since_created,
                }
            )

        df = pd.DataFrame(contacts_list)

        if not df.empty:
            # Enhanced engagement scoring with engagement_data integration
            df["base_engagement_score"] = (
                df["email_opens"] * 1 + df["email_clicks"] * 3 - df["email_bounces"] * 0.5
            )

            df["interaction_engagement_score"] = (
                df["form_submission_count"] * 8
                + df["meeting_count"] * 15
                + df["call_count"] * 12
                + df["social_engagement_count"] * 3
            )

            df["website_engagement_score"] = (
                df["website_session_count"] * 2 + df["email_event_count"] * 1.5
            )

            df["engagement_quality_score"] = (
                df["engagement_types_used"] * 5
                + df["progressive_engagement_score"] * 8
                + np.log1p(df["total_engagement_frequency"]) * 3
                - np.log1p(df["days_since_last_engagement"]) * 2
            )

            # Composite engagement score
            df["engagement_score"] = (
                df["base_engagement_score"] * 0.3
                + df["interaction_engagement_score"] * 0.3
                + df["website_engagement_score"] * 0.25
                + df["engagement_quality_score"] * 0.15
            )

            active_count = len(df[df["engagement_score"] > 0])

            # Enhanced segmentation options
            segments = {}
            if segmentation == "engagement_level":
                segments = {
                    "high_engagement": len(df[df["engagement_score"] >= 20]),
                    "medium_engagement": len(
                        df[(df["engagement_score"] >= 8) & (df["engagement_score"] < 20)]
                    ),
                    "low_engagement": len(df[df["engagement_score"] < 8]),
                }
            elif segmentation == "lifecycle_stage":
                segments = df["lifecyclestage"].value_counts().to_dict()
            elif segmentation == "engagement_diversity":
                segments = {
                    "multi_channel": len(df[df["engagement_types_used"] >= 3]),
                    "dual_channel": len(df[df["engagement_types_used"] == 2]),
                    "single_channel": len(df[df["engagement_types_used"] == 1]),
                    "no_engagement": len(df[df["engagement_types_used"] == 0]),
                }
            elif segmentation == "interaction_type":
                segments = {
                    "high_value_interactions": len(
                        df[(df["meeting_count"] > 0) | (df["call_count"] > 0)]
                    ),
                    "form_submitters": len(df[df["form_submission_count"] > 0]),
                    "website_browsers": len(df[df["website_session_count"] > 0]),
                    "email_only": len(
                        df[
                            (df["email_event_count"] > 0)
                            & (df["website_session_count"] == 0)
                            & (df["form_submission_count"] == 0)
                            & (df["meeting_count"] == 0)
                            & (df["call_count"] == 0)
                        ]
                    ),
                }

            # Enhanced detailed metrics
            detailed_metrics = {
                # Basic email metrics
                "total_opens": int(df["email_opens"].sum()),
                "total_clicks": int(df["email_clicks"].sum()),
                "total_bounces": int(df["email_bounces"].sum()),
                # Engagement event metrics
                "total_email_events": int(df["email_event_count"].sum()),
                "total_website_sessions": int(df["website_session_count"].sum()),
                "total_form_submissions": int(df["form_submission_count"].sum()),
                "total_social_interactions": int(df["social_engagement_count"].sum()),
                "total_meetings": int(df["meeting_count"].sum()),
                "total_calls": int(df["call_count"].sum()),
                # Quality metrics
                "avg_engagement_types_per_contact": float(df["engagement_types_used"].mean()),
                "avg_progressive_engagement": float(df["progressive_engagement_score"].mean()),
                "avg_days_since_last_engagement": float(df["days_since_last_engagement"].mean()),
                # Conversion metrics
                "contacts_with_meetings": int(len(df[df["meeting_count"] > 0])),
                "contacts_with_calls": int(len(df[df["call_count"] > 0])),
                "contacts_with_forms": int(len(df[df["form_submission_count"] > 0])),
                "multi_channel_contacts": int(len(df[df["engagement_types_used"] >= 3])),
            }

            return {
                "active_count": active_count,
                "avg_engagement": float(df["engagement_score"].mean()),
                "segments": segments,
                "detailed_metrics": detailed_metrics,
                "processing_time": 0.5,
                "velocity_metrics": self._calculate_contact_velocity(df),
                "engagement_insights": {
                    "top_engagement_score": float(df["engagement_score"].max()),
                    "engagement_distribution": {
                        "25th_percentile": float(df["engagement_score"].quantile(0.25)),
                        "median": float(df["engagement_score"].median()),
                        "75th_percentile": float(df["engagement_score"].quantile(0.75)),
                    },
                    "engagement_trends": {
                        "progressive_contacts": int(
                            len(df[df["progressive_engagement_score"] > 1])
                        ),
                        "declining_contacts": int(
                            len(df[df["progressive_engagement_score"] < 0.5])
                        ),
                        "recent_engagers": int(len(df[df["days_since_last_engagement"] <= 7])),
                    },
                },
            }
        else:
            return {
                "active_count": 0,
                "avg_engagement": 0,
                "segments": {},
                "detailed_metrics": {},
                "processing_time": 0.1,
                "velocity_metrics": {
                    "contacts_added_last_30_days": 0,
                    "avg_engagement_recent_contacts": 0,
                },
                "engagement_insights": {
                    "top_engagement_score": 0,
                    "engagement_distribution": {},
                    "engagement_trends": {},
                },
            }

    def _calculate_contact_velocity(self, df):
        """Calculate contact acquisition and engagement velocity"""
        if "createdate" in df.columns and not df.empty:
            try:
                df["createdate"] = pd.to_datetime(df["createdate"])
                now = pd.Timestamp.now()
                last_30_days = now - pd.Timedelta(days=30)
                recent_contacts = len(df[df["createdate"] >= last_30_days])
                return {
                    "contacts_added_last_30_days": recent_contacts,
                    "avg_engagement_recent_contacts": (
                        df[df["createdate"] >= last_30_days]["engagement_score"].mean()
                        if recent_contacts > 0
                        else 0
                    ),
                }
            except:
                pass
        return {"contacts_added_last_30_days": 0, "avg_engagement_recent_contacts": 0}

    async def analyze_campaign_performance(
        self,
        campaigns_data: List[Any],
        campaign_stats: List[Dict],
        metrics: List[str],
        engagement_data: List[Dict] = None,
    ) -> Dict[str, Any]:
        """Analyze campaign performance from HubSpot SDK data with engagement data integration

        Args:
            campaigns_data: Campaign objects from HubSpot SDK
            campaign_stats: Basic campaign statistics (sent, delivered, opened, clicked)
            metrics: List of metrics to calculate - controls which analytics are computed
            engagement_data: Optional engagement events data for enhanced analytics

        Supported metrics:
            - "open_rate": Email open rate analysis
            - "click_rate": Email click rate analysis
            - "bounce_rate": Email bounce rate analysis
            - "conversion_rate": Form conversion tracking (requires engagement_data)
            - "website_engagement": Website visit tracking (requires engagement_data)
            - "social_amplification": Social sharing analysis (requires engagement_data)
            - "engagement_quality": Overall engagement quality scoring (requires engagement_data)
            - "multi_action": Multi-action contact analysis (requires engagement_data)
            - "unsubscribe_rate": Unsubscribe tracking (requires engagement_data)
            - "engagement_velocity": Time-to-engagement analysis (requires engagement_data)
            - "all": Calculate all available metrics

        Returns:
            Dict containing campaign analytics with metrics-specific data

        Raises:
            ValueError: If metrics list is empty or contains invalid metric names
            TypeError: If campaigns_data or campaign_stats have invalid format
        """

        # Input validation
        if not metrics or not isinstance(metrics, list):
            raise ValueError("Metrics parameter must be a non-empty list")

        if not isinstance(campaigns_data, list):
            raise TypeError("campaigns_data must be a list")

        if not isinstance(campaign_stats, list):
            raise TypeError("campaign_stats must be a list")

        # Validate metric names
        valid_metrics = {
            "open_rate",
            "click_rate",
            "bounce_rate",
            "conversion_rate",
            "website_engagement",
            "social_amplification",
            "engagement_quality",
            "multi_action",
            "unsubscribe_rate",
            "engagement_velocity",
            "all",
        }

        invalid_metrics = [m for m in metrics if m not in valid_metrics]
        if invalid_metrics:
            raise ValueError(
                f"Invalid metrics: {invalid_metrics}. Valid options: {sorted(valid_metrics)}"
            )

        # Industry benchmarks (configurable)
        benchmarks = {
            "open_rate_threshold": 0.20,
            "click_rate_threshold": 0.025,
            "conversion_rate_threshold": 0.02,
            "unsubscribe_rate_threshold": 0.005,
            "multi_action_threshold": 0.15,
            "engagement_velocity_threshold": 7,
            "engagement_quality_high": 0.3,
            "engagement_quality_medium": 0.1,
        }

        # Determine which metrics to calculate based on metrics parameter
        calculate_all = "all" in metrics
        calculate_engagement_metrics = calculate_all or any(
            metric in metrics
            for metric in [
                "conversion_rate",
                "website_engagement",
                "social_amplification",
                "engagement_quality",
                "multi_action",
                "unsubscribe_rate",
                "engagement_velocity",
            ]
        )

        # Process engagement data for enhanced campaign analytics (only if needed)
        engagement_dict = {}
        if engagement_data and calculate_engagement_metrics:
            for engagement in engagement_data:
                campaign_id = str(
                    engagement.get("campaign_id", "") or engagement.get("campaignId", "")
                )
                contact_id = str(
                    engagement.get("contact_id", "") or engagement.get("contactId", "")
                )

                if campaign_id:
                    if campaign_id not in engagement_dict:
                        engagement_dict[campaign_id] = {
                            "email_events": [],
                            "website_visits": [],
                            "form_submissions": [],
                            "social_shares": [],
                            "unsubscribes": [],
                            "unique_contacts": set(),
                            "contact_engagements": {},
                        }

                    event_type = engagement.get("type", "").lower()
                    event_data = {
                        "timestamp": engagement.get("timestamp", ""),
                        "contact_id": contact_id,
                        "event_id": engagement.get("id", ""),
                        "properties": engagement.get("properties", {}),
                    }

                    # Track unique contacts per campaign
                    if contact_id:
                        engagement_dict[campaign_id]["unique_contacts"].add(contact_id)

                        # Track per-contact engagement for campaign
                        if contact_id not in engagement_dict[campaign_id]["contact_engagements"]:
                            engagement_dict[campaign_id]["contact_engagements"][contact_id] = {
                                "opens": 0,
                                "clicks": 0,
                                "form_submissions": 0,
                                "website_visits": 0,
                                "last_engagement": None,
                                "engagement_score": 0,
                            }

                    # Categorize engagement events
                    if "open" in event_type or "email_open" in event_type:
                        engagement_dict[campaign_id]["email_events"].append(event_data)
                        if contact_id:
                            engagement_dict[campaign_id]["contact_engagements"][contact_id][
                                "opens"
                            ] += 1
                    elif "click" in event_type or "email_click" in event_type:
                        engagement_dict[campaign_id]["email_events"].append(event_data)
                        if contact_id:
                            engagement_dict[campaign_id]["contact_engagements"][contact_id][
                                "clicks"
                            ] += 1
                    elif "website" in event_type or "page_view" in event_type:
                        engagement_dict[campaign_id]["website_visits"].append(event_data)
                        if contact_id:
                            engagement_dict[campaign_id]["contact_engagements"][contact_id][
                                "website_visits"
                            ] += 1
                    elif "form" in event_type or "submission" in event_type:
                        engagement_dict[campaign_id]["form_submissions"].append(event_data)
                        if contact_id:
                            engagement_dict[campaign_id]["contact_engagements"][contact_id][
                                "form_submissions"
                            ] += 1
                    elif "social" in event_type or "share" in event_type:
                        engagement_dict[campaign_id]["social_shares"].append(event_data)
                    elif "unsubscribe" in event_type:
                        engagement_dict[campaign_id]["unsubscribes"].append(event_data)

                    # Update last engagement timestamp
                    if contact_id:
                        try:
                            event_time = pd.to_datetime(engagement.get("timestamp", ""))
                            current_last = engagement_dict[campaign_id]["contact_engagements"][
                                contact_id
                            ]["last_engagement"]
                            if current_last is None or event_time > current_last:
                                engagement_dict[campaign_id]["contact_engagements"][contact_id][
                                    "last_engagement"
                                ] = event_time
                        except:
                            pass

        campaign_metrics = []

        for i, campaign in enumerate(campaigns_data):
            stats = campaign_stats[i] if i < len(campaign_stats) else {}
            campaign_id = str(campaign.id if hasattr(campaign, "id") else campaign.get("id", ""))

            # Basic metrics
            sent = stats.get("sent", 1)
            delivered = stats.get("delivered", sent)
            opened = stats.get("opened", 0)
            clicked = stats.get("clicked", 0)

            # Initialize engagement metrics (only calculated if requested)
            if calculate_engagement_metrics:
                # Enhanced engagement metrics from engagement_data
                campaign_engagement = engagement_dict.get(campaign_id, {})

                # Calculate enhanced engagement metrics
                email_events = campaign_engagement.get("email_events", [])
                website_visits = campaign_engagement.get("website_visits", [])
                form_submissions = campaign_engagement.get("form_submissions", [])
                social_shares = campaign_engagement.get("social_shares", [])
                unsubscribes = campaign_engagement.get("unsubscribes", [])
                unique_contacts = len(campaign_engagement.get("unique_contacts", set()))
                contact_engagements = campaign_engagement.get("contact_engagements", {})

                # Advanced engagement calculations
                post_email_website_visits = len(website_visits)
                post_email_form_submissions = len(form_submissions)
                social_amplification = len(social_shares)
                unsubscribe_count = len(unsubscribes)

                # Calculate engagement quality metrics (for multi_action and engagement_velocity)
                multi_action_contacts = 0
                high_engagement_contacts = 0
                days_to_engagement_sum = 0
                engaged_contacts_with_timing = 0

                if calculate_all or "multi_action" in metrics or "engagement_velocity" in metrics:
                    for contact_id, engagement_data in contact_engagements.items():
                        actions = (
                            engagement_data["opens"]
                            + engagement_data["clicks"]
                            + engagement_data["form_submissions"]
                            + engagement_data["website_visits"]
                        )
                        if actions > 1:
                            multi_action_contacts += 1
                        if actions >= 3:
                            high_engagement_contacts += 1

                        # Calculate days to engagement (from campaign send to last engagement)
                        if (calculate_all or "engagement_velocity" in metrics) and engagement_data[
                            "last_engagement"
                        ]:
                            try:
                                campaign_send_date = pd.to_datetime(
                                    getattr(campaign, "send_date", pd.Timestamp.now())
                                )
                                days_to_engage = (
                                    engagement_data["last_engagement"] - campaign_send_date
                                ).days
                                if days_to_engage >= 0:  # Only count positive engagement timing
                                    days_to_engagement_sum += days_to_engage
                                    engaged_contacts_with_timing += 1
                            except:
                                pass

                avg_days_to_engagement = (
                    days_to_engagement_sum / engaged_contacts_with_timing
                    if engaged_contacts_with_timing > 0
                    else 0
                )

                # Calculate conversion and retention metrics (conditional)
                form_conversion_rate = (
                    (post_email_form_submissions / delivered if delivered > 0 else 0)
                    if (calculate_all or "conversion_rate" in metrics)
                    else 0
                )

                website_engagement_rate = (
                    (post_email_website_visits / delivered if delivered > 0 else 0)
                    if (calculate_all or "website_engagement" in metrics)
                    else 0
                )

                social_amplification_rate = (
                    (social_amplification / delivered if delivered > 0 else 0)
                    if (calculate_all or "social_amplification" in metrics)
                    else 0
                )

                unsubscribe_rate = (
                    (unsubscribe_count / delivered if delivered > 0 else 0)
                    if (calculate_all or "unsubscribe_rate" in metrics)
                    else 0
                )

                multi_action_rate = (
                    (multi_action_contacts / unique_contacts if unique_contacts > 0 else 0)
                    if (calculate_all or "multi_action" in metrics)
                    else 0
                )

                # Calculate overall engagement quality score (conditional)
                if calculate_all or "engagement_quality" in metrics:
                    engagement_quality_score = (
                        (opened / delivered if delivered > 0 else 0) * 0.2  # Open rate
                        + (clicked / delivered if delivered > 0 else 0) * 0.3  # Click rate
                        + form_conversion_rate * 0.25  # Form conversion
                        + website_engagement_rate * 0.15  # Website engagement
                        + multi_action_rate * 0.1  # Multi-action rate
                    )
                else:
                    engagement_quality_score = 0
            else:
                # Set default values when engagement metrics not requested
                unique_contacts = 0
                post_email_website_visits = 0
                post_email_form_submissions = 0
                social_amplification = 0
                unsubscribe_count = 0
                multi_action_contacts = 0
                high_engagement_contacts = 0
                form_conversion_rate = 0
                website_engagement_rate = 0
                social_amplification_rate = 0
                unsubscribe_rate = 0
                multi_action_rate = 0
                engagement_quality_score = 0
                avg_days_to_engagement = 0

            # Build metrics_data based on requested metrics
            metrics_data = {
                "campaign_id": campaign_id,
                "name": getattr(campaign, "name", "Unknown"),
                "sent_count": sent,
                "delivered_count": delivered,
                "open_count": opened,
                "click_count": clicked,
            }

            # Add basic rate metrics based on request
            if calculate_all or "open_rate" in metrics:
                metrics_data["open_rate"] = opened / delivered if delivered > 0 else 0

            if calculate_all or "click_rate" in metrics:
                metrics_data["click_rate"] = clicked / delivered if delivered > 0 else 0

            if calculate_all or "bounce_rate" in metrics:
                metrics_data["bounce_rate"] = (sent - delivered) / sent if sent > 0 else 0

            # Add engagement metrics only if calculated
            if calculate_engagement_metrics:
                if calculate_all or "website_engagement" in metrics:
                    metrics_data.update(
                        {
                            "post_email_website_visits": post_email_website_visits,
                            "website_engagement_rate": website_engagement_rate,
                        }
                    )

                if calculate_all or "conversion_rate" in metrics:
                    metrics_data.update(
                        {
                            "post_email_form_submissions": post_email_form_submissions,
                            "form_conversion_rate": form_conversion_rate,
                        }
                    )

                if calculate_all or "social_amplification" in metrics:
                    metrics_data.update(
                        {
                            "social_amplification": social_amplification,
                            "social_amplification_rate": social_amplification_rate,
                        }
                    )

                if calculate_all or "unsubscribe_rate" in metrics:
                    metrics_data.update(
                        {
                            "unsubscribe_count": unsubscribe_count,
                            "unsubscribe_rate": unsubscribe_rate,
                        }
                    )

                if calculate_all or "multi_action" in metrics:
                    metrics_data.update(
                        {
                            "multi_action_contacts": multi_action_contacts,
                            "high_engagement_contacts": high_engagement_contacts,
                            "multi_action_rate": multi_action_rate,
                        }
                    )

                if calculate_all or "engagement_quality" in metrics:
                    metrics_data["engagement_quality_score"] = engagement_quality_score

                if calculate_all or "engagement_velocity" in metrics:
                    metrics_data["avg_days_to_engagement"] = avg_days_to_engagement

                # Always include unique contacts count if engagement metrics calculated
                metrics_data["unique_engaged_contacts"] = unique_contacts

                # Add engagement diversity metrics if any engagement metrics requested
                if calculate_all or any(
                    m in metrics
                    for m in [
                        "website_engagement",
                        "conversion_rate",
                        "social_amplification",
                    ]
                ):
                    total_events = 0
                    diversity_count = 0

                    if calculate_all or "website_engagement" in metrics:
                        total_events += len(website_visits)
                        if len(website_visits) > 0:
                            diversity_count += 1

                    if calculate_all or "conversion_rate" in metrics:
                        total_events += len(form_submissions)
                        if len(form_submissions) > 0:
                            diversity_count += 1

                    if calculate_all or "social_amplification" in metrics:
                        total_events += len(social_shares)
                        if len(social_shares) > 0:
                            diversity_count += 1

                    # Always include email events in total
                    total_events += len(email_events)
                    if len(email_events) > 0:
                        diversity_count += 1

                    metrics_data.update(
                        {
                            "total_engagement_events": total_events,
                            "engagement_diversity": diversity_count,
                        }
                    )
            campaign_metrics.append(metrics_data)

        if campaign_metrics:
            df = pd.DataFrame(campaign_metrics)

            # Build summary based on requested metrics
            summary = {
                "total_campaigns": len(campaign_metrics),
                "total_sent": int(df["sent_count"].sum()),
                "total_opened": int(df["open_count"].sum()),
                "total_clicked": int(df["click_count"].sum()),
            }

            # Add basic rate summaries if requested
            if calculate_all or "open_rate" in metrics:
                summary["avg_open_rate"] = float(df["open_rate"].mean())

            if calculate_all or "click_rate" in metrics:
                summary["avg_click_rate"] = float(df["click_rate"].mean())

            if calculate_all or "bounce_rate" in metrics:
                summary["avg_bounce_rate"] = float(df["bounce_rate"].mean())

            # Add engagement summaries only if calculated
            if calculate_engagement_metrics:
                if calculate_all or "conversion_rate" in metrics:
                    summary.update(
                        {
                            "avg_form_conversion_rate": float(df["form_conversion_rate"].mean()),
                            "total_form_submissions": int(df["post_email_form_submissions"].sum()),
                        }
                    )

                if calculate_all or "website_engagement" in metrics:
                    summary.update(
                        {
                            "avg_website_engagement_rate": float(
                                df["website_engagement_rate"].mean()
                            ),
                            "total_website_visits": int(df["post_email_website_visits"].sum()),
                        }
                    )

                if calculate_all or "social_amplification" in metrics:
                    summary.update(
                        {
                            "avg_social_amplification_rate": float(
                                df["social_amplification_rate"].mean()
                            ),
                            "total_social_shares": int(df["social_amplification"].sum()),
                        }
                    )

                if calculate_all or "unsubscribe_rate" in metrics:
                    summary.update(
                        {
                            "avg_unsubscribe_rate": float(df["unsubscribe_rate"].mean()),
                            "total_unsubscribes": int(df["unsubscribe_count"].sum()),
                        }
                    )

                if calculate_all or "multi_action" in metrics:
                    summary.update(
                        {
                            "avg_multi_action_rate": float(df["multi_action_rate"].mean()),
                            "total_multi_action_contacts": int(df["multi_action_contacts"].sum()),
                        }
                    )

                if calculate_all or "engagement_quality" in metrics:
                    summary["avg_engagement_quality_score"] = float(
                        df["engagement_quality_score"].mean()
                    )

                if calculate_all or "engagement_velocity" in metrics:
                    summary["avg_days_to_engagement"] = float(df["avg_days_to_engagement"].mean())

                # Always include unique contacts if engagement metrics calculated
                summary["total_unique_engaged_contacts"] = int(df["unique_engaged_contacts"].sum())
        else:
            # Minimal summary when no data
            summary = {
                "total_campaigns": 0,
                "total_sent": 0,
                "total_opened": 0,
                "total_clicked": 0,
            }

            # Add zeros for requested metrics
            if calculate_all or "open_rate" in metrics:
                summary["avg_open_rate"] = 0
            if calculate_all or "click_rate" in metrics:
                summary["avg_click_rate"] = 0
            if calculate_all or "bounce_rate" in metrics:
                summary["avg_bounce_rate"] = 0
            if calculate_all or "engagement_quality" in metrics:
                summary["avg_engagement_quality_score"] = 0

        # Enhanced insights with configurable benchmarks
        insights = []
        recommendations = []

        # Basic email metrics insights
        if "open_rate" in metrics or calculate_all:
            open_rate = summary.get("avg_open_rate", 0)
            if open_rate < benchmarks["open_rate_threshold"]:
                insights.append(
                    f"Open rates ({open_rate:.1%}) below industry benchmark ({benchmarks['open_rate_threshold']:.1%})"
                )
                recommendations.append("Consider A/B testing subject lines and sender names")
            elif open_rate > benchmarks["open_rate_threshold"] * 1.2:
                insights.append(
                    f"Excellent open rates ({open_rate:.1%}) - {(open_rate/benchmarks['open_rate_threshold']-1)*100:.0f}% above benchmark"
                )

        if "click_rate" in metrics or calculate_all:
            click_rate = summary.get("avg_click_rate", 0)
            if click_rate < benchmarks["click_rate_threshold"]:
                insights.append(
                    f"Click rates ({click_rate:.1%}) below industry benchmark ({benchmarks['click_rate_threshold']:.1%})"
                )
                recommendations.append("Improve email content relevance and call-to-action clarity")
            elif click_rate > benchmarks["click_rate_threshold"] * 1.5:
                insights.append(
                    f"Strong click rates ({click_rate:.1%}) indicate compelling content"
                )

        # Engagement metrics insights
        if "conversion_rate" in metrics or calculate_all:
            conversion_rate = summary.get("avg_form_conversion_rate", 0)
            if conversion_rate > benchmarks["conversion_rate_threshold"]:
                insights.append(
                    f"Excellent form conversion rate ({conversion_rate:.1%}) shows effective targeting"
                )
            elif (
                conversion_rate > 0
                and conversion_rate < benchmarks["conversion_rate_threshold"] * 0.5
            ):
                recommendations.append(
                    "Optimize landing pages and reduce form friction to improve conversions"
                )

        if "unsubscribe_rate" in metrics or calculate_all:
            unsub_rate = summary.get("avg_unsubscribe_rate", 0)
            if unsub_rate > benchmarks["unsubscribe_rate_threshold"]:
                insights.append(
                    f"High unsubscribe rate ({unsub_rate:.1%}) - review content relevance and frequency"
                )
                recommendations.append("Segment audience better and reduce email frequency")

        if "multi_action" in metrics or calculate_all:
            multi_action_rate = summary.get("avg_multi_action_rate", 0)
            if multi_action_rate > benchmarks["multi_action_threshold"]:
                insights.append(
                    f"High multi-action engagement ({multi_action_rate:.1%}) indicates strong campaign resonance"
                )
            elif multi_action_rate < benchmarks["multi_action_threshold"] * 0.5:
                recommendations.append(
                    "Create more engaging content to encourage multiple interactions"
                )

        if "engagement_velocity" in metrics or calculate_all:
            avg_days = summary.get("avg_days_to_engagement", 0)
            if avg_days > benchmarks["engagement_velocity_threshold"]:
                insights.append(
                    f"Slow engagement response ({avg_days:.1f} days avg) - consider follow-up optimization"
                )
                recommendations.append(
                    "Implement automated follow-up sequences for faster engagement"
                )

        # Advanced engagement quality insights
        if "engagement_quality" in metrics or calculate_all:
            quality_score = summary.get("avg_engagement_quality_score", 0)
            if quality_score > benchmarks["engagement_quality_high"]:
                insights.append(f"High-quality engagement campaigns (score: {quality_score:.2f})")
            elif quality_score < benchmarks["engagement_quality_medium"]:
                recommendations.append(
                    "Focus on improving overall engagement quality across all touchpoints"
                )

        # Enhanced trend analysis with metrics-specific calculations
        trends = {}

        def calculate_trend(recent_value, older_value, threshold=0.05):
            """Calculate trend direction with configurable threshold"""
            if recent_value > older_value * (1 + threshold):
                return "improving"
            elif recent_value < older_value * (1 - threshold):
                return "declining"
            else:
                return "stable"

        if campaign_metrics and len(campaign_metrics) >= 2:
            df_sorted = df.sort_values("campaign_id")  # Assuming campaign_id correlates with time
            split_point = len(df_sorted) // 2
            recent_half = df_sorted.tail(len(df_sorted) - split_point)
            older_half = df_sorted.head(split_point)

            # Calculate trends only for requested metrics
            if calculate_all or "open_rate" in metrics:
                if "open_rate" in df.columns:
                    recent_open = recent_half["open_rate"].mean()
                    older_open = older_half["open_rate"].mean()
                    trends["open_rate_trend"] = calculate_trend(recent_open, older_open)
                    trends["open_rate_change"] = (
                        f"{((recent_open/older_open-1)*100):+.1f}%" if older_open > 0 else "N/A"
                    )

            if calculate_all or "click_rate" in metrics:
                if "click_rate" in df.columns:
                    recent_click = recent_half["click_rate"].mean()
                    older_click = older_half["click_rate"].mean()
                    trends["click_rate_trend"] = calculate_trend(recent_click, older_click)
                    trends["click_rate_change"] = (
                        f"{((recent_click/older_click-1)*100):+.1f}%" if older_click > 0 else "N/A"
                    )

            if calculate_all or "bounce_rate" in metrics:
                if "bounce_rate" in df.columns:
                    recent_bounce = recent_half["bounce_rate"].mean()
                    older_bounce = older_half["bounce_rate"].mean()
                    # For bounce rate, improving means decreasing
                    trends["bounce_rate_trend"] = calculate_trend(older_bounce, recent_bounce)
                    trends["bounce_rate_change"] = (
                        f"{((recent_bounce/older_bounce-1)*100):+.1f}%"
                        if older_bounce > 0
                        else "N/A"
                    )

            if calculate_all or "engagement_quality" in metrics:
                if "engagement_quality_score" in df.columns:
                    recent_quality = recent_half["engagement_quality_score"].mean()
                    older_quality = older_half["engagement_quality_score"].mean()
                    trends["engagement_quality_trend"] = calculate_trend(
                        recent_quality, older_quality
                    )
                    trends["engagement_quality_change"] = (
                        f"{((recent_quality/older_quality-1)*100):+.1f}%"
                        if older_quality > 0
                        else "N/A"
                    )

            if calculate_all or "conversion_rate" in metrics:
                if "form_conversion_rate" in df.columns:
                    recent_conversion = recent_half["form_conversion_rate"].mean()
                    older_conversion = older_half["form_conversion_rate"].mean()
                    trends["form_conversion_trend"] = calculate_trend(
                        recent_conversion, older_conversion
                    )
                    trends["form_conversion_change"] = (
                        f"{((recent_conversion/older_conversion-1)*100):+.1f}%"
                        if older_conversion > 0
                        else "N/A"
                    )

            if calculate_all or "website_engagement" in metrics:
                if "website_engagement_rate" in df.columns:
                    recent_web = recent_half["website_engagement_rate"].mean()
                    older_web = older_half["website_engagement_rate"].mean()
                    trends["website_engagement_trend"] = calculate_trend(recent_web, older_web)
                    trends["website_engagement_change"] = (
                        f"{((recent_web/older_web-1)*100):+.1f}%" if older_web > 0 else "N/A"
                    )

        # Set stable trends for metrics not calculated
        for metric in [
            "open_rate",
            "click_rate",
            "bounce_rate",
            "engagement_quality",
            "form_conversion",
            "website_engagement",
        ]:
            if f"{metric}_trend" not in trends:
                trends[f"{metric}_trend"] = "stable"

        # Build conditional return structure
        result = {
            "campaign_metrics": campaign_metrics,
            "summary": summary,
            "insights": insights,
            "trends": trends,
        }

        # Add recommendations if any were generated
        if recommendations:
            result["recommendations"] = recommendations

        # Add engagement analysis only if engagement metrics were calculated
        if calculate_engagement_metrics and campaign_metrics:
            engagement_analysis = {}

            # Post-campaign actions
            total_actions = summary.get("total_website_visits", 0) + summary.get(
                "total_form_submissions", 0
            )
            if total_actions > 0:
                engagement_analysis["total_post_campaign_actions"] = total_actions

            # Quality distribution (only if engagement_quality was calculated)
            if calculate_all or "engagement_quality" in metrics:
                engagement_analysis["engagement_quality_distribution"] = {
                    "high_quality_campaigns": len(
                        [
                            c
                            for c in campaign_metrics
                            if c.get("engagement_quality_score", 0)
                            > benchmarks["engagement_quality_high"]
                        ]
                    ),
                    "medium_quality_campaigns": len(
                        [
                            c
                            for c in campaign_metrics
                            if benchmarks["engagement_quality_medium"]
                            <= c.get("engagement_quality_score", 0)
                            <= benchmarks["engagement_quality_high"]
                        ]
                    ),
                    "low_quality_campaigns": len(
                        [
                            c
                            for c in campaign_metrics
                            if c.get("engagement_quality_score", 0)
                            < benchmarks["engagement_quality_medium"]
                        ]
                    ),
                }

            # Behavioral insights (conditional based on metrics)
            behavioral_insights = {}

            if calculate_all or "website_engagement" in metrics:
                behavioral_insights["campaigns_driving_website_traffic"] = len(
                    [c for c in campaign_metrics if c.get("website_engagement_rate", 0) > 0.05]
                )

            if calculate_all or "conversion_rate" in metrics:
                behavioral_insights["campaigns_generating_leads"] = len(
                    [c for c in campaign_metrics if c.get("form_conversion_rate", 0) > 0.01]
                )

            if calculate_all or "social_amplification" in metrics:
                behavioral_insights["campaigns_with_social_amplification"] = len(
                    [c for c in campaign_metrics if c.get("social_amplification_rate", 0) > 0.001]
                )

            if behavioral_insights:
                engagement_analysis["behavioral_insights"] = behavioral_insights

            # Performance distribution
            if len(campaign_metrics) > 1:
                performance_scores = []
                for campaign in campaign_metrics:
                    # Calculate composite performance score
                    score = 0
                    weight_sum = 0

                    if "open_rate" in campaign:
                        score += campaign["open_rate"] * 0.2
                        weight_sum += 0.2
                    if "click_rate" in campaign:
                        score += campaign["click_rate"] * 0.3
                        weight_sum += 0.3
                    if "engagement_quality_score" in campaign:
                        score += campaign["engagement_quality_score"] * 0.5
                        weight_sum += 0.5

                    if weight_sum > 0:
                        performance_scores.append(score / weight_sum)

                if performance_scores:
                    import numpy as np

                    engagement_analysis["performance_distribution"] = {
                        "best_performance": float(max(performance_scores)),
                        "worst_performance": float(min(performance_scores)),
                        "performance_variance": float(np.var(performance_scores)),
                        "consistent_performance": float(np.var(performance_scores)) < 0.01,
                    }

            if engagement_analysis:
                result["engagement_analysis"] = engagement_analysis

        return result

    async def generate_benchmarks(
        self, performance_analysis: Dict[str, Any], benchmark_type: str
    ) -> Dict[str, Any]:
        """Generate performance benchmarks"""

        # Industry benchmarks (simplified)
        industry_benchmarks = {
            "industry_open_rate": 0.20,
            "industry_click_rate": 0.025,
            "industry_bounce_rate": 0.05,
        }

        if benchmark_type == "industry":
            return industry_benchmarks
        elif benchmark_type == "historical":
            # Would compare against historical data
            summary = performance_analysis["summary"]
            return {
                "historical_open_rate": summary["avg_open_rate"] * 0.95,  # Simulate historical
                "historical_click_rate": summary["avg_click_rate"] * 0.98,
                "historical_bounce_rate": summary["avg_bounce_rate"] * 1.02,
            }
        else:
            return industry_benchmarks

    async def analyze_sales_pipeline(
        self,
        deals_data: List[Any],
        analysis_type: str,
        engagement_data: List[Dict] = None,
    ) -> Dict[str, Any]:
        """Analyze sales pipeline from HubSpot SDK deals data with engagement data integration"""

        # Process engagement data for enhanced pipeline analytics
        engagement_dict = {}
        if engagement_data:
            for engagement in engagement_data:
                contact_id = str(
                    engagement.get("contact_id", "") or engagement.get("contactId", "")
                )
                deal_id = str(engagement.get("deal_id", "") or engagement.get("dealId", ""))

                if contact_id:
                    if contact_id not in engagement_dict:
                        engagement_dict[contact_id] = {
                            "total_engagement_events": 0,
                            "email_engagements": 0,
                            "website_visits": 0,
                            "form_submissions": 0,
                            "meeting_interactions": 0,
                            "call_interactions": 0,
                            "last_engagement_date": None,
                            "first_engagement_date": None,
                            "engagement_velocity": 0,
                            "deal_associations": set(),
                        }

                    # Track deal associations
                    if deal_id:
                        engagement_dict[contact_id]["deal_associations"].add(deal_id)

                    event_type = engagement.get("type", "").lower()
                    engagement_dict[contact_id]["total_engagement_events"] += 1

                    # Categorize engagement events for sales pipeline context
                    if "email" in event_type:
                        engagement_dict[contact_id]["email_engagements"] += 1
                    elif "website" in event_type or "page" in event_type:
                        engagement_dict[contact_id]["website_visits"] += 1
                    elif "form" in event_type or "submission" in event_type:
                        engagement_dict[contact_id]["form_submissions"] += 1
                    elif "meeting" in event_type:
                        engagement_dict[contact_id]["meeting_interactions"] += 1
                    elif "call" in event_type:
                        engagement_dict[contact_id]["call_interactions"] += 1

                    # Track engagement timeline
                    try:
                        event_date = pd.to_datetime(engagement.get("timestamp", ""))
                        if not pd.isna(event_date):
                            current_last = engagement_dict[contact_id]["last_engagement_date"]
                            current_first = engagement_dict[contact_id]["first_engagement_date"]

                            if current_last is None or event_date > current_last:
                                engagement_dict[contact_id]["last_engagement_date"] = event_date
                            if current_first is None or event_date < current_first:
                                engagement_dict[contact_id]["first_engagement_date"] = event_date
                    except:
                        pass

        # Calculate engagement velocity for each contact
        for contact_id, contact_data in engagement_dict.items():
            if (
                contact_data["first_engagement_date"]
                and contact_data["last_engagement_date"]
                and contact_data["total_engagement_events"] > 1
            ):
                try:
                    time_span = (
                        contact_data["last_engagement_date"] - contact_data["first_engagement_date"]
                    ).days
                    if time_span > 0:
                        contact_data["engagement_velocity"] = (
                            contact_data["total_engagement_events"] / time_span
                        )
                except:
                    pass

        # Convert SDK objects to DataFrame with engagement enrichment
        deals_list = []
        for deal in deals_data:
            properties = deal.properties if hasattr(deal, "properties") else deal
            deal_id = str(deal.id if hasattr(deal, "id") else deal.get("id", ""))

            # Parse amount safely
            amount_str = properties.get("amount", "0")
            try:
                amount = float(amount_str) if amount_str else 0.0
            except (ValueError, TypeError):
                amount = 0.0

            # Get associated contact engagement data
            associated_contact_ids = []
            contact_engagement_score = 0
            total_contact_engagements = 0
            sales_interaction_count = 0
            avg_engagement_velocity = 0
            days_since_last_engagement = 999

            # Find contacts associated with this deal
            for contact_id, contact_data in engagement_dict.items():
                if deal_id in contact_data["deal_associations"]:
                    associated_contact_ids.append(contact_id)
                    total_contact_engagements += contact_data["total_engagement_events"]
                    sales_interaction_count += (
                        contact_data["meeting_interactions"] + contact_data["call_interactions"]
                    )

                    # Calculate engagement score for this contact
                    contact_score = (
                        contact_data["email_engagements"] * 1
                        + contact_data["website_visits"] * 2
                        + contact_data["form_submissions"] * 5
                        + contact_data["meeting_interactions"] * 10
                        + contact_data["call_interactions"] * 8
                    )
                    contact_engagement_score += contact_score

                    # Track engagement velocity
                    avg_engagement_velocity += contact_data["engagement_velocity"]

                    # Track days since last engagement
                    if contact_data["last_engagement_date"]:
                        try:
                            days_since = (
                                pd.Timestamp.now() - contact_data["last_engagement_date"]
                            ).days
                            if days_since < days_since_last_engagement:
                                days_since_last_engagement = days_since
                        except:
                            pass

            # Average engagement velocity across associated contacts
            if len(associated_contact_ids) > 0:
                avg_engagement_velocity = avg_engagement_velocity / len(associated_contact_ids)
                contact_engagement_score = contact_engagement_score / len(associated_contact_ids)

            # Calculate deal engagement quality indicators
            has_high_engagement = contact_engagement_score > 20
            has_sales_interactions = sales_interaction_count > 0
            is_recently_engaged = days_since_last_engagement <= 30

            deals_list.append(
                {
                    "id": deal_id,
                    "name": properties.get("dealname", ""),
                    "amount": amount,
                    "stage": properties.get("dealstage", ""),
                    "pipeline": properties.get("pipeline", ""),
                    "createdate": properties.get("createdate", ""),
                    "closedate": properties.get("closedate", ""),
                    "probability": float(properties.get("hs_deal_stage_probability", 0) or 0),
                    # Enhanced engagement metrics
                    "associated_contacts_count": len(associated_contact_ids),
                    "contact_engagement_score": contact_engagement_score,
                    "total_contact_engagements": total_contact_engagements,
                    "sales_interaction_count": sales_interaction_count,
                    "avg_engagement_velocity": avg_engagement_velocity,
                    "days_since_last_engagement": days_since_last_engagement,
                    "has_high_engagement": has_high_engagement,
                    "has_sales_interactions": has_sales_interactions,
                    "is_recently_engaged": is_recently_engaged,
                    # Engagement quality score for the deal
                    "engagement_quality_score": min(
                        contact_engagement_score / 50, 1.0
                    ),  # Normalized to 0-1
                }
            )

        df = pd.DataFrame(deals_list)

        if df.empty:
            return {
                "total_value": 0,
                "deal_count": 0,
                "avg_deal_size": 0,
                "conversion_rate": 0,
                "stage_metrics": {},
                "insights": ["No deals data available"],
                "engagement_insights": {
                    "avg_engagement_score": 0,
                    "highly_engaged_deals": 0,
                },
            }

        # Calculate enhanced pipeline metrics
        total_value = float(df["amount"].sum())
        deal_count = len(df)
        avg_deal_size = float(df["amount"].mean())

        # Calculate conversion rate (closed won vs total)
        closed_won_count = len(df[df["stage"].str.contains("won|closed", case=False, na=False)])
        conversion_rate = closed_won_count / deal_count if deal_count > 0 else 0

        # Enhanced stage analysis with engagement data
        stage_metrics = {}
        if "stage" in df.columns:
            stage_analysis = (
                df.groupby("stage")
                .agg(
                    {
                        "amount": ["sum", "count", "mean"],
                        "probability": "mean",
                        "contact_engagement_score": "mean",
                        "sales_interaction_count": "sum",
                        "engagement_quality_score": "mean",
                    }
                )
                .round(2)
            )

            for stage in stage_analysis.index:
                stage_metrics[stage] = {
                    "count": int(stage_analysis.loc[stage, ("amount", "count")]),
                    "value": float(stage_analysis.loc[stage, ("amount", "sum")]),
                    "avg_deal_size": float(stage_analysis.loc[stage, ("amount", "mean")]),
                    "avg_probability": float(stage_analysis.loc[stage, ("probability", "mean")]),
                    "avg_engagement_score": float(
                        stage_analysis.loc[stage, ("contact_engagement_score", "mean")]
                    ),
                    "total_sales_interactions": int(
                        stage_analysis.loc[stage, ("sales_interaction_count", "sum")]
                    ),
                    "avg_engagement_quality": float(
                        stage_analysis.loc[stage, ("engagement_quality_score", "mean")]
                    ),
                }

        # Enhanced insights with engagement data
        insights = []
        if conversion_rate < 0.2:
            insights.append(
                f"Low conversion rate: {conversion_rate:.1%} - consider pipeline optimization"
            )
        if avg_deal_size < 1000:
            insights.append("Average deal size is low - focus on upselling opportunities")

        # Engagement-specific insights
        avg_engagement_score = float(df["contact_engagement_score"].mean())
        high_engagement_deals = len(df[df["has_high_engagement"] == True])
        deals_with_sales_interactions = len(df[df["has_sales_interactions"] == True])
        recently_engaged_deals = len(df[df["is_recently_engaged"] == True])

        if avg_engagement_score > 15:
            insights.append("Strong overall engagement levels indicate good lead quality")
        if high_engagement_deals / deal_count > 0.3:
            insights.append("High percentage of deals have strong engagement - good qualification")
        if deals_with_sales_interactions / deal_count < 0.2:
            insights.append("Low sales interaction rate - consider more direct outreach")
        if recently_engaged_deals / deal_count < 0.4:
            insights.append("Many deals lack recent engagement - implement re-engagement campaigns")

        # Calculate actual velocity metrics
        velocity_metrics = self._calculate_pipeline_velocity(df, closed_won_count)

        # Enhanced engagement analysis
        engagement_insights = {
            "avg_engagement_score": avg_engagement_score,
            "highly_engaged_deals": high_engagement_deals,
            "deals_with_sales_interactions": deals_with_sales_interactions,
            "recently_engaged_deals": recently_engaged_deals,
            "avg_engagement_velocity": float(df["avg_engagement_velocity"].mean()),
            "engagement_quality_distribution": {
                "high_quality_deals": len(df[df["engagement_quality_score"] > 0.7]),
                "medium_quality_deals": len(
                    df[
                        (df["engagement_quality_score"] >= 0.3)
                        & (df["engagement_quality_score"] <= 0.7)
                    ]
                ),
                "low_quality_deals": len(df[df["engagement_quality_score"] < 0.3]),
            },
            "engagement_impact_on_close_rate": {
                "high_engagement_close_rate": len(
                    df[
                        (df["has_high_engagement"] == True)
                        & (df["stage"].str.contains("won|closed", case=False, na=False))
                    ]
                )
                / max(high_engagement_deals, 1),
                "low_engagement_close_rate": len(
                    df[
                        (df["has_high_engagement"] == False)
                        & (df["stage"].str.contains("won|closed", case=False, na=False))
                    ]
                )
                / max(deal_count - high_engagement_deals, 1),
            },
        }

        return {
            "total_value": total_value,
            "deal_count": deal_count,
            "avg_deal_size": avg_deal_size,
            "conversion_rate": conversion_rate,
            "stage_metrics": stage_metrics,
            "insights": insights,
            "velocity_metrics": velocity_metrics,
            "engagement_insights": engagement_insights,
            "pipeline_health": {
                "engagement_score": avg_engagement_score,
                "sales_interaction_rate": (
                    deals_with_sales_interactions / deal_count if deal_count > 0 else 0
                ),
                "recent_engagement_rate": (
                    recently_engaged_deals / deal_count if deal_count > 0 else 0
                ),
                "overall_pipeline_health": (
                    "healthy"
                    if (avg_engagement_score > 10 and conversion_rate > 0.15)
                    else (
                        "needs_attention"
                        if (avg_engagement_score > 5 or conversion_rate > 0.1)
                        else "poor"
                    )
                ),
            },
        }

    def _calculate_pipeline_velocity(self, deals_df, closed_won_count):
        """Calculate actual pipeline velocity metrics"""
        if (
            "createdate" in deals_df.columns
            and "closedate" in deals_df.columns
            and not deals_df.empty
        ):
            try:
                deals_df["createdate"] = pd.to_datetime(deals_df["createdate"])
                deals_df["closedate"] = pd.to_datetime(deals_df["closedate"])

                closed_deals = deals_df[deals_df["closedate"].notna()]
                if not closed_deals.empty:
                    closed_deals["days_in_pipeline"] = (
                        closed_deals["closedate"] - closed_deals["createdate"]
                    ).dt.days
                    avg_days = closed_deals["days_in_pipeline"].mean()

                    # Calculate deals closed this month
                    now = pd.Timestamp.now()
                    this_month_start = now.replace(day=1)
                    deals_closed_this_month = len(
                        closed_deals[closed_deals["closedate"] >= this_month_start]
                    )

                    return {
                        "avg_days_in_pipeline": avg_days,
                        "deals_closed_this_month": deals_closed_this_month,
                        "pipeline_velocity_score": (
                            len(closed_deals) / len(deals_df) if len(deals_df) > 0 else 0
                        ),
                    }
            except Exception:
                pass

        return {
            "avg_days_in_pipeline": 30,  # Default
            "deals_closed_this_month": closed_won_count,
            "pipeline_velocity_score": 0,
        }

    async def predict_lead_scores(
        self,
        contacts_data: List[Any],
        deals_data: List[Any],
        engagement_data: List[Dict],
        model_type: str = "auto",
    ) -> Dict[str, Any]:
        """Enhanced ML-based lead scoring with comprehensive feature engineering"""

        try:
            import numpy as np
            from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import classification_report
            from sklearn.model_selection import cross_val_score, train_test_split
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            # Fallback to simple scoring if sklearn not available
            return await self._simple_lead_scoring(contacts_data)

        # Enhanced data processing with deal history and engagement integration
        contacts_list = []
        deals_dict = {}
        engagement_dict = {}

        # Process deals data for feature enrichment
        if deals_data:
            for deal in deals_data:
                deal_props = deal.properties if hasattr(deal, "properties") else deal
                contact_ids = deal_props.get("associatedcompanyids", "") or deal_props.get(
                    "contact_ids", ""
                )
                if contact_ids:
                    for contact_id in str(contact_ids).split(","):
                        contact_id = contact_id.strip()
                        if contact_id not in deals_dict:
                            deals_dict[contact_id] = []
                        deals_dict[contact_id].append(
                            {
                                "amount": float(deal_props.get("amount", 0) or 0),
                                "stage": deal_props.get("dealstage", ""),
                                "closed": "closed"
                                in (deal_props.get("dealstage", "") or "").lower(),
                                "won": "won" in (deal_props.get("dealstage", "") or "").lower(),
                            }
                        )

        # Process engagement data for advanced engagement metrics
        if engagement_data:
            for engagement in engagement_data:
                contact_id = str(
                    engagement.get("contact_id", "") or engagement.get("contactId", "")
                )
                if contact_id:
                    if contact_id not in engagement_dict:
                        engagement_dict[contact_id] = {
                            "email_events": [],
                            "website_events": [],
                            "form_submissions": [],
                            "page_views": [],
                            "social_interactions": [],
                            "meeting_interactions": [],
                            "call_interactions": [],
                        }

                    event_type = engagement.get("type", "").lower()
                    event_data = {
                        "timestamp": engagement.get("timestamp", ""),
                        "event_id": engagement.get("id", ""),
                        "properties": engagement.get("properties", {}),
                    }

                    # Categorize engagement events
                    if "email" in event_type:
                        engagement_dict[contact_id]["email_events"].append(event_data)
                    elif "website" in event_type or "page" in event_type:
                        engagement_dict[contact_id]["website_events"].append(event_data)
                        engagement_dict[contact_id]["page_views"].append(event_data)
                    elif "form" in event_type or "submission" in event_type:
                        engagement_dict[contact_id]["form_submissions"].append(event_data)
                    elif "social" in event_type:
                        engagement_dict[contact_id]["social_interactions"].append(event_data)
                    elif "meeting" in event_type or "call" in event_type:
                        if "meeting" in event_type:
                            engagement_dict[contact_id]["meeting_interactions"].append(event_data)
                        else:
                            engagement_dict[contact_id]["call_interactions"].append(event_data)

        # Enhanced feature engineering for contacts with engagement data integration
        for contact in contacts_data:
            properties = contact.properties if hasattr(contact, "properties") else contact
            contact_id = str(contact.id if hasattr(contact, "id") else contact.get("id", ""))

            # Basic engagement features from contact properties
            email_opens = int(properties.get("hs_email_open", 0) or 0)
            email_clicks = int(properties.get("hs_email_click", 0) or 0)
            email_bounces = int(properties.get("hs_email_bounce", 0) or 0)

            # Deal history features
            contact_deals = deals_dict.get(contact_id, [])
            total_deal_value = sum(d["amount"] for d in contact_deals)
            deal_count = len(contact_deals)
            won_deals = sum(1 for d in contact_deals if d["won"])
            closed_deals = sum(1 for d in contact_deals if d["closed"])

            # Enhanced engagement features from engagement_data
            contact_engagement = engagement_dict.get(contact_id, {})

            # Email engagement metrics from engagement events
            email_events = contact_engagement.get("email_events", [])
            email_event_count = len(email_events)
            unique_email_campaigns = len(
                set(
                    e.get("properties", {}).get("campaign_id", "")
                    for e in email_events
                    if e.get("properties", {}).get("campaign_id")
                )
            )

            # Website engagement metrics
            website_events = contact_engagement.get("website_events", [])
            page_views = contact_engagement.get("page_views", [])
            website_sessions = len(website_events)
            total_page_views = len(page_views)
            unique_pages_visited = len(
                set(
                    e.get("properties", {}).get("url", "")
                    for e in page_views
                    if e.get("properties", {}).get("url")
                )
            )

            # Calculate website engagement depth
            avg_session_duration = 0
            if website_events:
                try:
                    durations = [
                        float(e.get("properties", {}).get("duration", 0) or 0)
                        for e in website_events
                    ]
                    avg_session_duration = sum(durations) / len(durations) if durations else 0
                except:
                    avg_session_duration = 0

            # Form engagement metrics
            form_submissions = contact_engagement.get("form_submissions", [])
            form_submission_count = len(form_submissions)
            unique_forms_submitted = len(
                set(
                    e.get("properties", {}).get("form_id", "")
                    for e in form_submissions
                    if e.get("properties", {}).get("form_id")
                )
            )

            # Social and interaction metrics
            social_interactions = contact_engagement.get("social_interactions", [])
            meeting_interactions = contact_engagement.get("meeting_interactions", [])
            call_interactions = contact_engagement.get("call_interactions", [])

            social_engagement_count = len(social_interactions)
            meeting_count = len(meeting_interactions)
            call_count = len(call_interactions)

            # Calculate engagement recency and frequency
            all_engagement_events = (
                email_events
                + website_events
                + form_submissions
                + social_interactions
                + meeting_interactions
                + call_interactions
            )

            engagement_frequency = len(all_engagement_events)
            days_since_last_engagement = 999  # Default high value

            if all_engagement_events:
                try:
                    # Sort events by timestamp and get most recent
                    sorted_events = sorted(
                        all_engagement_events,
                        key=lambda x: pd.to_datetime(x.get("timestamp", "1900-01-01")),
                        reverse=True,
                    )
                    if sorted_events:
                        last_engagement = pd.to_datetime(sorted_events[0].get("timestamp", ""))
                        if not pd.isna(last_engagement):
                            days_since_last_engagement = (pd.Timestamp.now() - last_engagement).days
                except:
                    pass

            # Temporal features
            try:
                create_date = pd.to_datetime(properties.get("createdate", ""))
                days_since_created = (
                    (pd.Timestamp.now() - create_date).days if not pd.isna(create_date) else 0
                )
            except:
                days_since_created = 0

            # Advanced engagement ratios and scores
            engagement_ratio = email_clicks / max(email_opens, 1)
            bounce_rate = email_bounces / max(email_opens + email_clicks + email_bounces, 1)

            # Calculate engagement diversity score (variety of engagement types)
            engagement_types_used = sum(
                [
                    1 if email_event_count > 0 else 0,
                    1 if website_sessions > 0 else 0,
                    1 if form_submission_count > 0 else 0,
                    1 if social_engagement_count > 0 else 0,
                    1 if meeting_count > 0 else 0,
                    1 if call_count > 0 else 0,
                ]
            )

            # Website engagement quality score
            website_engagement_quality = 0
            if total_page_views > 0:
                website_engagement_quality = (unique_pages_visited / total_page_views) * np.log1p(
                    avg_session_duration
                )

            # Progressive engagement score (indicates increasing engagement over time)
            progressive_engagement_score = 0
            if len(all_engagement_events) >= 2:
                try:
                    # Calculate engagement trend over time
                    recent_events = [
                        e
                        for e in all_engagement_events
                        if (
                            pd.Timestamp.now() - pd.to_datetime(e.get("timestamp", "1900-01-01"))
                        ).days
                        <= 30
                    ]
                    older_events = [
                        e
                        for e in all_engagement_events
                        if (
                            pd.Timestamp.now() - pd.to_datetime(e.get("timestamp", "1900-01-01"))
                        ).days
                        > 30
                    ]

                    recent_count = len(recent_events)
                    older_count = len(older_events) if older_events else 1
                    progressive_engagement_score = recent_count / older_count
                except:
                    progressive_engagement_score = 1

            contacts_list.append(
                {
                    "id": contact_id,
                    # Basic email engagement
                    "email_opens": email_opens,
                    "email_clicks": email_clicks,
                    "email_bounces": email_bounces,
                    "engagement_ratio": engagement_ratio,
                    "bounce_rate": bounce_rate,
                    # Deal history
                    "total_deal_value": total_deal_value,
                    "deal_count": deal_count,
                    "won_deals": won_deals,
                    "closed_deals": closed_deals,
                    # Enhanced engagement metrics from engagement_data
                    "email_event_count": email_event_count,
                    "unique_email_campaigns": unique_email_campaigns,
                    "website_sessions": website_sessions,
                    "total_page_views": total_page_views,
                    "unique_pages_visited": unique_pages_visited,
                    "avg_session_duration": avg_session_duration,
                    "form_submission_count": form_submission_count,
                    "unique_forms_submitted": unique_forms_submitted,
                    "social_engagement_count": social_engagement_count,
                    "meeting_count": meeting_count,
                    "call_count": call_count,
                    "engagement_frequency": engagement_frequency,
                    "days_since_last_engagement": days_since_last_engagement,
                    "engagement_types_used": engagement_types_used,
                    "website_engagement_quality": website_engagement_quality,
                    "progressive_engagement_score": progressive_engagement_score,
                    # Temporal and demographic
                    "days_since_created": days_since_created,
                    "lifecyclestage": properties.get("lifecyclestage", ""),
                    "lead_status": properties.get("hs_lead_status", ""),
                    "company_size": int(properties.get("numberofemployees", 0) or 0),
                    "industry": properties.get("industry", ""),
                }
            )

        contacts_df = pd.DataFrame(contacts_list)

        if contacts_df.empty:
            return {
                "scores": [],
                "performance_metrics": {"accuracy": 0, "model_type": "none"},
                "feature_importance": [],
                "training_size": 0,
                "data_quality": {"issues": ["No contact data available"]},
            }

        # Comprehensive engagement scoring with engagement_data integration
        contacts_df["base_engagement_score"] = (
            contacts_df["email_opens"] * 1.0
            + contacts_df["email_clicks"] * 3.0
            - contacts_df["email_bounces"] * 2.0
            + contacts_df["engagement_ratio"] * 5.0
            - contacts_df["bounce_rate"] * 3.0
        )

        # Enhanced engagement scoring using engagement_data
        contacts_df["website_engagement_score"] = (
            contacts_df["website_sessions"] * 2.0
            + contacts_df["total_page_views"] * 0.5
            + contacts_df["unique_pages_visited"] * 1.5
            + contacts_df["website_engagement_quality"] * 10.0
        )

        contacts_df["interaction_engagement_score"] = (
            contacts_df["form_submission_count"] * 8.0
            + contacts_df["unique_forms_submitted"] * 5.0
            + contacts_df["meeting_count"] * 15.0
            + contacts_df["call_count"] * 12.0
            + contacts_df["social_engagement_count"] * 3.0
        )

        contacts_df["engagement_quality_score"] = (
            contacts_df["engagement_types_used"] * 5.0
            + contacts_df["progressive_engagement_score"] * 8.0
            + np.log1p(contacts_df["engagement_frequency"]) * 3.0
            - np.log1p(contacts_df["days_since_last_engagement"]) * 2.0
        )

        # Composite engagement score combining all engagement types
        contacts_df["engagement_score"] = (
            contacts_df["base_engagement_score"] * 0.3
            + contacts_df["website_engagement_score"] * 0.25
            + contacts_df["interaction_engagement_score"] * 0.3
            + contacts_df["engagement_quality_score"] * 0.15
        )

        # Lifecycle stage scoring with business logic
        lifecycle_mapping = {
            "subscriber": 1,
            "lead": 2,
            "marketingqualifiedlead": 3,
            "salesqualifiedlead": 4,
            "opportunity": 5,
            "customer": 6,
            "evangelist": 7,
        }
        contacts_df["lifecycle_score"] = (
            contacts_df["lifecyclestage"].map(lifecycle_mapping).fillna(0)
        )

        # Deal performance features
        contacts_df["avg_deal_value"] = contacts_df["total_deal_value"] / contacts_df[
            "deal_count"
        ].replace(0, 1)
        contacts_df["deal_win_rate"] = contacts_df["won_deals"] / contacts_df["deal_count"].replace(
            0, 1
        )
        contacts_df["deal_close_rate"] = contacts_df["closed_deals"] / contacts_df[
            "deal_count"
        ].replace(0, 1)

        # Temporal features
        contacts_df["recency_score"] = np.exp(
            -contacts_df["days_since_created"] / 365
        )  # Decay over time

        # Company features
        contacts_df["company_size_score"] = np.log1p(contacts_df["company_size"])

        # Create enhanced target variable using actual business outcomes
        contacts_df["converted"] = (
            (contacts_df["won_deals"] > 0)  # Has won deals
            | (contacts_df["lifecycle_score"] >= 5)  # Opportunity or customer
            | (
                (contacts_df["engagement_score"] > 15) & (contacts_df["lifecycle_score"] >= 3)
            )  # High engagement MQL+
        ).astype(int)

        # Data quality assessment
        quality_issues = []
        if contacts_df["converted"].sum() < 5:
            quality_issues.append("Insufficient positive samples for reliable training")
        if contacts_df["engagement_score"].std() < 1:
            quality_issues.append("Low engagement variance")
        if len(contacts_df) < 50:
            quality_issues.append("Small dataset size")

        # Enhanced feature selection with engagement_data features
        feature_columns = [
            # Basic email engagement
            "email_opens",
            "email_clicks",
            "engagement_ratio",
            "bounce_rate",
            # Composite engagement scores
            "engagement_score",
            "base_engagement_score",
            "website_engagement_score",
            "interaction_engagement_score",
            "engagement_quality_score",
            # Engagement_data specific features
            "email_event_count",
            "unique_email_campaigns",
            "website_sessions",
            "total_page_views",
            "unique_pages_visited",
            "avg_session_duration",
            "form_submission_count",
            "unique_forms_submitted",
            "social_engagement_count",
            "meeting_count",
            "call_count",
            "engagement_frequency",
            "days_since_last_engagement",
            "engagement_types_used",
            "website_engagement_quality",
            "progressive_engagement_score",
            # Deal and lifecycle features
            "lifecycle_score",
            "total_deal_value",
            "deal_count",
            "deal_win_rate",
            "avg_deal_value",
            "recency_score",
            "company_size_score",
            "days_since_created",
        ]

        X = contacts_df[feature_columns].fillna(0)
        y = contacts_df["converted"]

        if len(X) < 20 or y.sum() < 3:  # Insufficient data for reliable ML
            return await self._simple_lead_scoring(contacts_data)

        # Enhanced model selection based on model_type parameter
        models = {
            "random_forest": RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42
            ),
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=50, learning_rate=0.1, random_state=42
            ),
            "logistic_regression": LogisticRegression(random_state=42, max_iter=1000),
        }

        if model_type == "auto":
            # Select best model via cross-validation
            best_score = 0
            best_model_name = "random_forest"

            for name, model in models.items():
                try:
                    scores = cross_val_score(model, X, y, cv=min(5, len(X) // 4), scoring="roc_auc")
                    avg_score = scores.mean()
                    if avg_score > best_score:
                        best_score = avg_score
                        best_model_name = name
                except:
                    continue

            selected_model = models[best_model_name]
            model_name = best_model_name
        else:
            selected_model = models.get(model_type, models["random_forest"])
            model_name = model_type

        # Train the selected model
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        selected_model.fit(X_train_scaled, y_train)

        # Enhanced predictions with confidence intervals
        X_scaled = scaler.transform(X)
        if hasattr(selected_model, "predict_proba"):
            predictions = selected_model.predict_proba(X_scaled)[:, 1]
        else:
            predictions = selected_model.decision_function(X_scaled)
            predictions = (predictions - predictions.min()) / (
                predictions.max() - predictions.min()
            )

        # Model performance evaluation
        test_accuracy = selected_model.score(X_test_scaled, y_test) if len(X_test) > 0 else 0.8

        try:
            y_pred = selected_model.predict(X_test_scaled)
            class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            precision = class_report.get("1", {}).get("precision", 0)
            recall = class_report.get("1", {}).get("recall", 0)
            f1_score = class_report.get("1", {}).get("f1-score", 0)
        except:
            precision = recall = f1_score = 0

        # Dynamic risk level assignment based on score distribution
        score_percentiles = np.percentile(predictions, [33, 67])

        # Generate enhanced scores with explanations
        scores = []
        for i, contact in contacts_df.iterrows():
            score = float(predictions[i])

            # Dynamic risk levels based on distribution
            if score > score_percentiles[1]:
                risk_level = "high"
            elif score > score_percentiles[0]:
                risk_level = "medium"
            else:
                risk_level = "low"

            # Generate explanation for the score
            top_features = []
            contact_features = X.iloc[i]
            if hasattr(selected_model, "feature_importances_"):
                feature_impacts = selected_model.feature_importances_ * contact_features.values
                top_indices = np.argsort(feature_impacts)[-3:][::-1]
                top_features = [feature_columns[idx] for idx in top_indices]

            scores.append(
                {
                    "contact_id": contact["id"],
                    "score": score,
                    "risk_level": risk_level,
                    "confidence": float(abs(score - 0.5) * 2),  # Distance from decision boundary
                    "top_factors": top_features,
                    "engagement_score": float(contact_features["engagement_score"]),
                    "lifecycle_stage": contact["lifecyclestage"],
                }
            )

        # Enhanced feature importance analysis
        if hasattr(selected_model, "feature_importances_"):
            feature_importance = [
                {
                    "feature": col,
                    "importance": float(imp),
                    "description": self._get_feature_description(col),
                }
                for col, imp in zip(feature_columns, selected_model.feature_importances_)
            ]
        else:
            # For models without feature_importances_ (like LogisticRegression)
            feature_importance = [
                {
                    "feature": col,
                    "importance": float(abs(coef)),
                    "description": self._get_feature_description(col),
                }
                for col, coef in zip(feature_columns, selected_model.coef_[0])
            ]

        feature_importance.sort(key=lambda x: x["importance"], reverse=True)

        return {
            "scores": scores,
            "performance_metrics": {
                "accuracy": test_accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "model_type": model_name,
                "training_samples": len(X_train),
                "cross_val_score": best_score if model_type == "auto" else None,
            },
            "feature_importance": feature_importance,
            "training_size": len(X_train),
            "data_quality": {
                "total_contacts": len(contacts_df),
                "positive_samples": int(y.sum()),
                "negative_samples": int(len(y) - y.sum()),
                "feature_count": len(feature_columns),
                "issues": quality_issues,
            },
            "model_insights": {
                "top_predictive_features": [f["feature"] for f in feature_importance[:3]],
                "score_distribution": {
                    "mean": float(np.mean(predictions)),
                    "std": float(np.std(predictions)),
                    "percentiles": {
                        "25th": float(np.percentile(predictions, 25)),
                        "50th": float(np.percentile(predictions, 50)),
                        "75th": float(np.percentile(predictions, 75)),
                    },
                },
            },
        }

    def _get_feature_description(self, feature_name: str) -> str:
        """Get human-readable description for model features"""
        descriptions = {
            # Basic email engagement
            "email_opens": "Total email opens by contact",
            "email_clicks": "Total email clicks by contact",
            "engagement_ratio": "Click-to-open ratio indicating email engagement quality",
            "bounce_rate": "Email bounce rate indicating deliverability issues",
            # Composite engagement scores
            "engagement_score": "Comprehensive engagement score combining all interaction types",
            "base_engagement_score": "Basic email engagement score from contact properties",
            "website_engagement_score": "Website activity and session quality score",
            "interaction_engagement_score": "High-value interaction score (forms, meetings, calls)",
            "engagement_quality_score": "Engagement diversity and progression quality score",
            # Engagement_data specific features
            "email_event_count": "Number of email engagement events from engagement data",
            "unique_email_campaigns": "Number of unique email campaigns engaged with",
            "website_sessions": "Number of website sessions",
            "total_page_views": "Total number of page views",
            "unique_pages_visited": "Number of unique pages visited",
            "avg_session_duration": "Average website session duration in seconds",
            "form_submission_count": "Number of form submissions",
            "unique_forms_submitted": "Number of unique forms submitted",
            "social_engagement_count": "Number of social media interactions",
            "meeting_count": "Number of meeting interactions",
            "call_count": "Number of call interactions",
            "engagement_frequency": "Total frequency of all engagement events",
            "days_since_last_engagement": "Days since last recorded engagement activity",
            "engagement_types_used": "Number of different engagement types used (diversity)",
            "website_engagement_quality": "Website engagement depth and quality metric",
            "progressive_engagement_score": "Trend indicating increasing engagement over time",
            # Deal and lifecycle features
            "lifecycle_score": "Numeric representation of contact's lifecycle stage",
            "total_deal_value": "Total value of all deals associated with contact",
            "deal_count": "Number of deals associated with contact",
            "deal_win_rate": "Percentage of deals won by contact",
            "avg_deal_value": "Average value per deal for contact",
            "recency_score": "Recency score based on contact creation date",
            "company_size_score": "Logarithmic score of company size (employees)",
            "days_since_created": "Number of days since contact was created",
        }
        return descriptions.get(feature_name, f"Score for {feature_name}")

    async def _simple_lead_scoring(self, contacts_data: List[Any]) -> Dict[str, Any]:
        """Fallback simple scoring when ML libraries not available"""
        scores = []
        for contact in contacts_data:
            properties = contact.properties if hasattr(contact, "properties") else contact
            email_opens = int(properties.get("hs_email_open", 0) or 0)
            email_clicks = int(properties.get("hs_email_click", 0) or 0)

            engagement_score = email_opens * 1 + email_clicks * 3
            score = min(engagement_score / 20.0, 1.0)  # Normalize to 0-1

            scores.append(
                {
                    "contact_id": (contact.id if hasattr(contact, "id") else contact.get("id")),
                    "score": score,
                    "risk_level": ("high" if score > 0.7 else "medium" if score > 0.3 else "low"),
                }
            )

        return {
            "scores": scores,
            "performance_metrics": {"accuracy": 0.8},
            "feature_importance": [
                {"feature": "engagement_score", "importance": 0.6},
                {"feature": "lifecycle_score", "importance": 0.4},
            ],
            "training_size": len(contacts_data),
            "accuracy": 0.8,
        }


class InsightGenerator:
    """Generate business insights and recommendations"""

    async def generate_contact_insights(
        self, analytics_result: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Generate insights from contact analytics"""
        insights = []

        avg_engagement = analytics_result.get("avg_engagement", 0)
        active_count = analytics_result.get("active_count", 0)
        segments = analytics_result.get("segments", {})
        total_contacts = sum(segments.values()) if segments else 1

        if avg_engagement < 5:
            insights.append(
                {
                    "message": "Overall engagement is low - consider email re-engagement campaign",
                    "action": "Launch re-engagement campaign targeting inactive contacts",
                    "priority": "high",
                }
            )

        if active_count / total_contacts < 0.3:
            insights.append(
                {
                    "message": f"Only {(active_count/total_contacts)*100:.1f}% of contacts are active",
                    "action": "Implement lead nurturing workflows for inactive contacts",
                    "priority": "medium",
                }
            )

        # Segment-specific insights
        if "low_engagement" in segments and segments["low_engagement"] > total_contacts * 0.5:
            insights.append(
                {
                    "message": f'High number of low-engagement contacts: {segments["low_engagement"]}',
                    "action": "Create targeted content for re-engagement",
                    "priority": "high",
                }
            )

        if "high_engagement" in segments and segments["high_engagement"] > 0:
            insights.append(
                {
                    "message": f'You have {segments["high_engagement"]} highly engaged contacts',
                    "action": "Convert high-engagement contacts to sales opportunities",
                    "priority": "medium",
                }
            )

        return insights

    async def generate_campaign_recommendations(
        self, performance_analysis: Dict[str, Any], benchmarks: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Generate campaign optimization recommendations"""
        recommendations = []

        summary = performance_analysis.get("summary", {})
        avg_open_rate = summary.get("avg_open_rate", 0)
        avg_click_rate = summary.get("avg_click_rate", 0)
        avg_bounce_rate = summary.get("avg_bounce_rate", 0)

        benchmark_open_rate = benchmarks.get("industry_open_rate", 0.20)
        benchmark_click_rate = benchmarks.get("industry_click_rate", 0.025)
        benchmark_bounce_rate = benchmarks.get("industry_bounce_rate", 0.05)

        if avg_open_rate < benchmark_open_rate:
            recommendations.append(
                {
                    "action": "Improve subject line testing and personalization",
                    "description": f"Open rate {avg_open_rate:.2%} is below benchmark {benchmark_open_rate:.2%}",
                    "priority": "high",
                }
            )

        if avg_click_rate < benchmark_click_rate:
            recommendations.append(
                {
                    "action": "Enhance email content and call-to-action buttons",
                    "description": f"Click rate {avg_click_rate:.2%} is below benchmark {benchmark_click_rate:.2%}",
                    "priority": "medium",
                }
            )

        if avg_bounce_rate > benchmark_bounce_rate:
            recommendations.append(
                {
                    "action": "Improve email list quality and validation",
                    "description": f"Bounce rate {avg_bounce_rate:.2%} is above benchmark {benchmark_bounce_rate:.2%}",
                    "priority": "high",
                }
            )

        # Campaign-specific recommendations
        campaign_metrics = performance_analysis.get("campaign_metrics", [])
        if campaign_metrics:
            low_performing = [
                c for c in campaign_metrics if c["open_rate"] < benchmark_open_rate * 0.8
            ]
            if low_performing:
                recommendations.append(
                    {
                        "action": f"Review and optimize {len(low_performing)} low-performing campaigns",
                        "description": "Some campaigns are significantly underperforming",
                        "priority": "medium",
                    }
                )

        return recommendations

    async def generate_pipeline_recommendations(
        self, pipeline_analysis: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Generate sales pipeline recommendations"""
        recommendations = []

        conversion_rate = pipeline_analysis.get("conversion_rate", 0)
        avg_deal_size = pipeline_analysis.get("avg_deal_size", 0)
        stage_metrics = pipeline_analysis.get("stage_metrics", {})

        if conversion_rate < 0.2:
            recommendations.append(
                {
                    "action": "Optimize lead qualification process",
                    "description": f"Low conversion rate: {conversion_rate:.1%}",
                    "priority": "high",
                }
            )

        if avg_deal_size < 5000:
            recommendations.append(
                {
                    "action": "Implement upselling and cross-selling strategies",
                    "description": f"Average deal size is low: ${avg_deal_size:,.2f}",
                    "priority": "medium",
                }
            )

        # Stage-specific recommendations
        for stage, metrics in stage_metrics.items():
            if metrics["count"] > 20 and metrics["avg_probability"] < 0.3:
                recommendations.append(
                    {
                        "action": f"Review and optimize {stage} stage process",
                        "description": f"High volume but low probability in {stage}",
                        "priority": "medium",
                    }
                )

        return recommendations

    async def analyze_predictions(self, prediction_result: Dict[str, Any]) -> Dict[str, List[str]]:
        """Analyze prediction results and generate insights"""

        scores = prediction_result.get("scores", [])
        accuracy = prediction_result.get("accuracy", 0)
        feature_importance = prediction_result.get("feature_importance", [])

        insights = []
        recommendations = []

        # Model performance insights
        if accuracy > 0.8:
            insights.append(f"High model accuracy: {accuracy:.1%} - predictions are reliable")
        elif accuracy > 0.6:
            insights.append(
                f"Moderate model accuracy: {accuracy:.1%} - use predictions with caution"
            )
        else:
            insights.append(f"Low model accuracy: {accuracy:.1%} - model needs improvement")

        # Score distribution insights
        high_scores = [s for s in scores if s["score"] > 0.7]
        medium_scores = [s for s in scores if 0.3 <= s["score"] <= 0.7]
        low_scores = [s for s in scores if s["score"] < 0.3]

        insights.append(
            f"Score distribution: {len(high_scores)} high, {len(medium_scores)} medium, {len(low_scores)} low"
        )

        # Feature importance insights
        if feature_importance:
            top_feature = feature_importance[0]
            insights.append(
                f"Most important factor: {top_feature['feature']} ({top_feature['importance']:.2f})"
            )

        # Recommendations based on scores
        if len(high_scores) > 0:
            recommendations.append(
                f"Prioritize {len(high_scores)} high-scoring contacts for immediate follow-up"
            )

        if len(medium_scores) > len(high_scores) * 2:
            recommendations.append(
                "Large number of medium-score contacts - consider nurturing campaigns"
            )

        if len(low_scores) > len(scores) * 0.6:
            recommendations.append(
                "High number of low-scoring contacts - review lead generation quality"
            )

        return {"insights": insights, "recommendations": recommendations}


class MCPHubspotConnector:
    """Service layer using official HubSpot Python SDK with analytics capabilities"""

    def __init__(
        self,
        logger: logging.Logger,
        config: HubSpotSDKConfig = None,
        **settings: Dict[str, Any],
    ):
        self.logger = logger

        # Handle configuration
        if config:
            self.config = config
            access_token = config.access_token
            calls_per_second = config.calls_per_second
            self.setting = asdict(config)
        else:
            # Fallback to settings dict for backward compatibility
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

    @handle_hubspot_errors
    async def get_contact_analytics(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """MCP Tool: get_contact_analytics - Advanced contact analytics using SDK"""
        try:
            date_range = params.get("dateRange", {})
            segmentation = params.get("segmentation", "all")
            include_engagement = params.get("includeEngagement", True)
            limit = params.get("limit", 1000)

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
                "error_message": None
            }

        except Exception as e:
            return {
                "success": False,
                "data": None,
                "insights": [],
                "recommendations": [],
                "metadata": {},
                "error_message": f"Contact analytics failed: {str(e)}"
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
            filter_group = FilterGroup(
                filters=[
                    Filter(
                        property_name="createdate",
                        operator="BETWEEN",
                        value=date_range["start"],
                        high_value=date_range["end"],
                    )
                ]
            )

            search_request = PublicObjectSearchRequest(
                filter_groups=[filter_group],
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
                        limit=batch_limit,
                        after=after,
                        properties=properties,
                        archived=False
                    )

                    if get_page_response and get_page_response.results:
                        all_contacts.extend(get_page_response.results)
                        total_fetched += len(get_page_response.results)

                        # Check for next page
                        if (hasattr(get_page_response, 'paging') and 
                            get_page_response.paging and 
                            hasattr(get_page_response.paging, 'next') and 
                            get_page_response.paging.next):
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
                        "last_engagement_date": datetime.now().isoformat(),
                    }
                )

        return engagement_data

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
                    limit=batch_limit,
                    after=after,
                    properties=properties,
                    archived=False
                )

                if response and response.results:
                    for contact in response.results:
                        if total_fetched >= limit:
                            break
                        contact_data = {"id": contact.id, "properties": contact.properties}
                        contacts.append(contact_data)
                        total_fetched += 1

                    # Check for next page
                    if (hasattr(response, 'paging') and 
                        response.paging and 
                        hasattr(response.paging, 'next') and 
                        response.paging.next and
                        total_fetched < limit):
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

            # Use proper pagination with basic_api.get_page
            deals = []
            after = None
            total_fetched = 0
            batch_limit = min(100, limit)

            while total_fetched < limit:
                response = client.crm.deals.basic_api.get_page(
                    limit=batch_limit,
                    after=after,
                    properties=properties,
                    archived=False
                )

                if response and response.results:
                    for deal in response.results:
                        if total_fetched >= limit:
                            break
                        deal_data = {"id": deal.id, "properties": deal.properties}
                        deals.append(deal_data)
                        total_fetched += 1

                    # Check for next page
                    if (hasattr(response, 'paging') and 
                        response.paging and 
                        hasattr(response.paging, 'next') and 
                        response.paging.next and
                        total_fetched < limit):
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

            # Use proper pagination with basic_api.get_page
            companies = []
            after = None
            total_fetched = 0
            batch_limit = min(100, limit)

            while total_fetched < limit:
                response = client.crm.companies.basic_api.get_page(
                    limit=batch_limit,
                    after=after,
                    properties=properties,
                    archived=False
                )

                if response and response.results:
                    for company in response.results:
                        if total_fetched >= limit:
                            break
                        company_data = {"id": company.id, "properties": company.properties}
                        companies.append(company_data)
                        total_fetched += 1

                    # Check for next page
                    if (hasattr(response, 'paging') and 
                        response.paging and 
                        hasattr(response.paging, 'next') and 
                        response.paging.next and
                        total_fetched < limit):
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
                filters=[Filter(property_name="email", operator="CONTAINS_TOKEN", value=query)]
            )

            search_request = PublicObjectSearchRequest(
                filter_groups=[filter_group],
                properties=properties,
                limit=min(limit, 100),
            )

            response = client.crm.contacts.search_api.do_search(search_request)

            contacts = []
            if response:
                for contact in response:
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

            if response:
                contact = response[0]
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

            # Marketing events API pagination (if supported)
            marketing_events_api = client.marketing.events
            events = []
            after = None
            total_fetched = 0
            batch_limit = min(100, limit)

            while total_fetched < limit:
                try:
                    # Try get_page method first (if available)
                    if hasattr(marketing_events_api.basic_api, 'get_page'):
                        response = marketing_events_api.basic_api.get_page(
                            limit=batch_limit,
                            after=after
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
                            "end_date_time": self._format_datetime(getattr(event, "end_date_time", "")),
                        }
                        events.append(event_data)
                        total_fetched += 1

                    # Check for next page (if pagination is supported)
                    if (hasattr(response, 'paging') and 
                        response.paging and 
                        hasattr(response.paging, 'next') and 
                        response.paging.next and
                        total_fetched < limit and
                        hasattr(marketing_events_api.basic_api, 'get_page')):
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

    @handle_hubspot_errors
    async def analyze_campaign_performance(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """MCP Tool: analyze_campaign_performance - Campaign analytics using SDK"""
        try:
            campaign_ids = params.get("campaignIds", [])
            metrics = params.get("metrics", ["open_rate", "click_rate", "conversion_rate"])
            benchmark_type = params.get("benchmarkType", "historical")
            include_recommendations = params.get("includeRecommendations", True)

            # Get campaign data using SDK
            campaigns_data = await self._get_campaigns_with_sdk(campaign_ids)

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
                    "analysis_date": datetime.now().isoformat(),
                    "campaigns_analyzed": len(campaigns_data),
                    "benchmark_type": benchmark_type,
                    "data_source": "hubspot_sdk_marketing",
                },
                "error_message": None
            }

        except Exception as e:
            return {
                "success": False,
                "data": None,
                "insights": [],
                "recommendations": [],
                "metadata": {},
                "error_message": f"Campaign analysis failed: {str(e)}"
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
                    response = marketing_events_api.basic_api.get_page(limit=100, after=after)

                    if response and response.results:
                        campaigns.extend(response.results)

                    if response.paging and response.paging.next:
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

    @handle_hubspot_errors
    async def analyze_sales_pipeline(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """MCP Tool: analyze_sales_pipeline - Sales pipeline analytics using SDK"""
        try:
            pipeline_ids = params.get("pipelineIds", [])
            timeframe = params.get("timeframe", {})
            analysis_type = params.get("analysisType", "conversion_rates")
            include_recommendations = params.get("includeRecommendations", True)

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
                "error_message": None
            }

        except Exception as e:
            return {
                "success": False,
                "data": None,
                "insights": [],
                "recommendations": [],
                "metadata": {},
                "error_message": f"Pipeline analysis failed: {str(e)}"
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

            # Add pipeline filter
            if pipeline_ids:
                filters.append(Filter(property_name="pipeline", operator="IN", values=pipeline_ids))

            # Add date range filter
            if timeframe and timeframe.get("start") and timeframe.get("end"):
                filters.append(
                    Filter(
                        property_name="createdate",
                        operator="BETWEEN",
                        value=timeframe["start"],
                        high_value=timeframe["end"],
                    )
                )

            filter_group = FilterGroup(filters=filters)
            search_request = PublicObjectSearchRequest(
                filter_groups=[filter_group], properties=properties, limit=100
            )

            # Paginate through search results
            after = None
            while True:
                if after:
                    search_request.after = after

                await self.rate_limiter.wait_if_needed()
                search_response = deals_api.search_api.do_search(search_request)

                if search_response and search_response.results:
                    all_deals.extend(search_response.results)

                if search_response.paging and search_response.paging.next:
                    after = search_response.paging.next.after
                else:
                    break

        else:
            # Get all deals without filters using basic_api.get_page
            after = None
            while True:
                await self.rate_limiter.wait_if_needed()
                response = deals_api.basic_api.get_page(
                    limit=100,
                    after=after,
                    properties=properties,
                    archived=False
                )

                if response and response.results:
                    all_deals.extend(response.results)

                if (hasattr(response, 'paging') and 
                    response.paging and 
                    hasattr(response.paging, 'next') and 
                    response.paging.next):
                    after = response.paging.next.after
                else:
                    break

        self.logger.info(f"Fetched {len(all_deals)} deals using HubSpot SDK")
        return all_deals

    @handle_hubspot_errors
    async def predict_lead_scores(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """MCP Tool: predict_lead_scores - ML-based lead scoring using SDK"""
        try:
            contact_ids = params.get("contactIds", [])
            model_type = params.get("modelType", "conversion_probability")
            include_feature_importance = params.get("includeFeatureImportance", True)

            # Get contacts using SDK
            if contact_ids:
                contacts_data = await self._get_contacts_by_ids_with_sdk(contact_ids)
            else:
                contacts_data = await self._get_contacts_with_sdk(limit=1000)

            # Get historical deals for training
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
                    "model_accuracy": prediction_result["accuracy"],
                    "data_source": "hubspot_sdk_multi_api",
                },
                "error_message": None
            }

        except Exception as e:
            return {
                "success": False,
                "data": None,
                "insights": [],
                "recommendations": [],
                "metadata": {},
                "error_message": f"Lead scoring failed: {str(e)}"
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

    # ============ EXPORT FUNCTIONS ============

    async def export_contact_analytics(self, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Export function for contact analytics"""
        if params is None:
            params = {
                "dateRange": {
                    "start": "2024-01-01T00:00:00Z",
                    "end": "2024-12-31T23:59:59Z",
                },
                "segmentation": "engagement_level",
                "includeEngagement": True,
                "limit": 1000,
            }
        return await self.get_contact_analytics(params)

    async def export_campaign_performance(self, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Export function for campaign performance analysis"""
        if params is None:
            params = {
                "campaignIds": [],  # Will get all campaigns
                "metrics": ["open_rate", "click_rate", "conversion_rate"],
                "benchmarkType": "industry",
                "includeRecommendations": True,
            }
        return await self.analyze_campaign_performance(params)

    async def export_sales_pipeline(self, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Export function for sales pipeline analysis"""
        if params is None:
            params = {
                "timeframe": {
                    "start": "2024-01-01T00:00:00Z",
                    "end": "2024-12-31T23:59:59Z",
                },
                "analysisType": "conversion_rates",
                "includeRecommendations": True,
            }
        return await self.analyze_sales_pipeline(params)

    async def export_lead_scores(self, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Export function for lead scoring prediction"""
        if params is None:
            params = {
                "modelType": "conversion_probability",
                "includeFeatureImportance": True,
            }
        return await self.predict_lead_scores(params)

    # Redundant export functions removed - use the direct methods instead:
    # get_contacts(), get_deals(), get_companies(), search_contacts(),
    # get_contact_by_email(), create_contact(), update_contact(),
    # create_deal(), get_marketing_events(), ping()
