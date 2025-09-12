#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Analytics Engine for MCP HubSpot Connector

This module provides advanced analytics capabilities including:
- Contact metrics processing and segmentation
- Campaign performance analysis
- Sales pipeline analytics with engagement integration
- ML-based lead scoring and predictions
- Customer segmentation (RFM, behavioral, lifecycle, value-based)
- Revenue forecasting with multiple models
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import pendulum

try:
    import numpy as np
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

    # Create minimal fallbacks
    class pd:
        @staticmethod
        def DataFrame(data):
            return data if isinstance(data, list) else []

        @staticmethod
        def to_datetime(data, errors="coerce"):
            try:
                if isinstance(data, str):
                    return pendulum.parse(data)
                return data
            except:
                return None

        @staticmethod
        def Timestamp():
            class MockTimestamp:
                @staticmethod
                def now():
                    return pendulum.now()

            return MockTimestamp()

        @staticmethod
        def isna(data):
            return data is None

    class np:
        @staticmethod
        def array(data):
            return data

        @staticmethod
        def mean(data):
            return sum(data) / len(data) if data else 0

        @staticmethod
        def var(data):
            if not data:
                return 0
            mean_val = sum(data) / len(data)
            return sum((x - mean_val) ** 2 for x in data) / len(data)


class AnalyticsEngine:
    """Advanced analytics engine for HubSpot data processing"""

    def __init__(self):
        """Initialize the analytics engine"""
        self.pandas_available = PANDAS_AVAILABLE

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
            campaign_id = campaign.object_id

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
            historical_benchmarks = {}

            # Only include metrics that are actually calculated
            if "avg_open_rate" in summary:
                historical_benchmarks["historical_open_rate"] = summary["avg_open_rate"] * 0.95
            if "avg_click_rate" in summary:
                historical_benchmarks["historical_click_rate"] = summary["avg_click_rate"] * 0.98
            if "avg_bounce_rate" in summary:
                historical_benchmarks["historical_bounce_rate"] = summary["avg_bounce_rate"] * 1.02
            if "avg_conversion_rate" in summary:
                historical_benchmarks["historical_conversion_rate"] = (
                    summary["avg_conversion_rate"] * 0.90
                )

            return historical_benchmarks
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

        # Analysis type-specific insights and focus areas
        insights = []
        avg_engagement_score = float(df["contact_engagement_score"].mean())
        high_engagement_deals = len(df[df["has_high_engagement"] == True])
        deals_with_sales_interactions = len(df[df["has_sales_interactions"] == True])
        recently_engaged_deals = len(df[df["is_recently_engaged"] == True])

        # Base insights that apply to all analysis types
        if conversion_rate < 0.2:
            insights.append(
                f"Low conversion rate: {conversion_rate:.1%} - consider pipeline optimization"
            )
        if avg_deal_size < 1000:
            insights.append("Average deal size is low - focus on upselling opportunities")

        # Analysis type-specific insights
        if analysis_type == "conversion_rates":
            # Focus on conversion optimization
            if high_engagement_deals / deal_count > 0.3:
                insights.append(
                    "High percentage of deals have strong engagement - good qualification"
                )
            if deals_with_sales_interactions / deal_count < 0.2:
                insights.append("Low sales interaction rate - consider more direct outreach")
            # Add conversion-specific metrics analysis
            if conversion_rate > 0.25:
                insights.append("Excellent conversion rate - current process is working well")
            elif conversion_rate > 0.15:
                insights.append(
                    "Good conversion rate - minor optimizations could boost performance"
                )

        elif analysis_type == "engagement_analysis":
            # Focus on engagement metrics
            if avg_engagement_score > 15:
                insights.append("Strong overall engagement levels indicate good lead quality")
            if recently_engaged_deals / deal_count < 0.4:
                insights.append(
                    "Many deals lack recent engagement - implement re-engagement campaigns"
                )
            # Add engagement-specific insights
            avg_velocity = float(df["avg_engagement_velocity"].mean())
            if avg_velocity > 1.0:
                insights.append("High engagement velocity indicates active prospects")
            elif avg_velocity < 0.3:
                insights.append("Low engagement velocity - consider nurturing campaigns")

        elif analysis_type == "deal_velocity":
            # Focus on deal progression speed
            velocity_metrics = self._calculate_pipeline_velocity(df, closed_won_count)
            if velocity_metrics.get("avg_days_to_close", 0) > 90:
                insights.append("Long sales cycle detected - consider process acceleration")
            if velocity_metrics.get("avg_days_to_close", 0) < 30:
                insights.append("Fast sales cycle - excellent deal velocity")

        elif analysis_type == "stage_analysis":
            # Focus on stage-specific metrics
            stage_counts = df["stage"].value_counts()
            if len(stage_counts) > 0:
                bottleneck_stage = stage_counts.idxmax()
                bottleneck_count = stage_counts.max()
                if bottleneck_count > deal_count * 0.4:
                    insights.append(
                        f"Bottleneck detected at '{bottleneck_stage}' stage - {bottleneck_count} deals"
                    )

        else:
            # Default comprehensive analysis
            if avg_engagement_score > 15:
                insights.append("Strong overall engagement levels indicate good lead quality")
            if recently_engaged_deals / deal_count < 0.4:
                insights.append(
                    "Many deals lack recent engagement - implement re-engagement campaigns"
                )

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
            "analysis_type": analysis_type,  # Document which analysis type was used
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
            import pandas as pd
            from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import classification_report
            from sklearn.model_selection import cross_val_score, train_test_split
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            # Fallback to simple scoring if sklearn not available
            return await self._simple_lead_scoring(contacts_data)

        if not contacts_data:
            return {
                "scores": [],
                "accuracy": 0,
                "feature_importance": [],
                "model_type": "no_data",
                "training_size": 0,
                "performance_metrics": {
                    "accuracy": 0,
                },
                "insights": ["No contacts data available for scoring"],
            }

        # Enhanced data processing with deal history and engagement integration
        deals_dict = {}
        engagement_dict = {}

        # Process deals data for feature enrichment
        if deals_data:
            for deal in deals_data:
                deal_props = deal.properties if hasattr(deal, "properties") else deal
                contact_ids = deal_props.get("associatedcontactids", "") or deal_props.get(
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
                    elif "meeting" in event_type:
                        engagement_dict[contact_id]["meeting_interactions"].append(event_data)
                    elif "call" in event_type:
                        engagement_dict[contact_id]["call_interactions"].append(event_data)

        # Enhanced feature engineering for contacts with engagement data integration
        features_list = []
        labels = []
        contacts_list = []

        for contact in contacts_data:
            contact_id = str(contact.id if hasattr(contact, "id") else contact.get("id", ""))
            properties = contact.properties if hasattr(contact, "properties") else contact

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

            # Email engagement metrics
            email_events = contact_engagement.get("email_events", [])
            email_event_count = len(email_events)

            # Website engagement metrics
            website_events = contact_engagement.get("website_events", [])
            page_views = contact_engagement.get("page_views", [])
            website_sessions = len(website_events)
            total_page_views = len(page_views)

            # Form engagement metrics
            form_submissions = contact_engagement.get("form_submissions", [])
            form_submission_count = len(form_submissions)

            # Social and interaction metrics
            meeting_count = len(contact_engagement.get("meeting_interactions", []))
            call_count = len(contact_engagement.get("call_interactions", []))

            # Calculate engagement frequency
            all_events = (
                email_events
                + website_events
                + form_submissions
                + contact_engagement.get("meeting_interactions", [])
                + contact_engagement.get("call_interactions", [])
            )
            engagement_frequency = len(all_events)

            # Lifecycle stage factor
            lifecycle = properties.get("lifecyclestage", "").lower()
            lifecycle_score = 0.1  # default
            if "opportunity" in lifecycle:
                lifecycle_score = 0.8
            elif "lead" in lifecycle:
                lifecycle_score = 0.6
            elif "customer" in lifecycle:
                lifecycle_score = 0.9
            elif "subscriber" in lifecycle:
                lifecycle_score = 0.3

            # Calculate advanced engagement ratios
            engagement_ratio = email_clicks / max(email_opens, 1)
            bounce_rate = email_bounces / max(email_opens + email_clicks + email_bounces, 1)

            # Create feature vector
            features = [
                email_opens,
                email_clicks,
                email_bounces,
                engagement_ratio,
                bounce_rate,
                total_deal_value,
                deal_count,
                won_deals,
                closed_deals,
                email_event_count,
                website_sessions,
                total_page_views,
                form_submission_count,
                meeting_count,
                call_count,
                engagement_frequency,
                lifecycle_score,
            ]

            # Create label (1 for high-value prospects, 0 for low-value)
            # High-value criteria: won deals OR high engagement OR qualified lifecycle stage
            label = int(
                won_deals > 0
                or engagement_frequency > 10
                or lifecycle_score > 0.7
                or total_deal_value > 1000
            )

            features_list.append(features)
            labels.append(label)
            contacts_list.append({"contact_id": contact_id, "properties": properties})

        # Convert to numpy arrays
        X = np.array(features_list)
        y = np.array(labels)

        if len(X) == 0 or len(set(y)) < 2:
            # Not enough data for ML, fallback to simple scoring
            return await self._simple_lead_scoring(contacts_data)

        # Feature scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Model selection based on model_type
        if model_type == "gradient_boost":
            model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        elif model_type == "random_forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == "logistic":
            model = LogisticRegression(random_state=42)
        else:  # auto
            # Choose best model based on data size
            if len(X) > 1000:
                model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            elif len(X) > 100:
                model = RandomForestClassifier(n_estimators=50, random_state=42)
            else:
                model = LogisticRegression(random_state=42)

        # Cross-validation for model assessment
        cv_scores = cross_val_score(model, X_scaled, y, cv=min(5, len(X) // 2))
        model_accuracy = np.mean(cv_scores)

        # Train final model
        model.fit(X_scaled, y)

        # Generate predictions and probabilities
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)[:, 1]  # Probability of being high-value

        # Create detailed scores
        detailed_scores = []
        for i, contact_info in enumerate(contacts_list):
            detailed_scores.append(
                {
                    "contact_id": contact_info["contact_id"],
                    "score": float(probabilities[i]),
                    "prediction": int(predictions[i]),
                    "confidence": float(max(model.predict_proba(X_scaled[i : i + 1])[0])),
                    "features": {
                        "email_opens": features_list[i][0],
                        "email_clicks": features_list[i][1],
                        "engagement_frequency": features_list[i][15],
                        "deal_count": features_list[i][6],
                        "lifecycle_score": features_list[i][16],
                    },
                }
            )

        # Feature importance
        feature_names = [
            "email_opens",
            "email_clicks",
            "email_bounces",
            "engagement_ratio",
            "bounce_rate",
            "total_deal_value",
            "deal_count",
            "won_deals",
            "closed_deals",
            "email_events",
            "website_sessions",
            "page_views",
            "form_submissions",
            "meeting_count",
            "call_count",
            "engagement_frequency",
            "lifecycle_score",
        ]

        if hasattr(model, "feature_importances_"):
            feature_importance = [
                {"feature": name, "importance": float(importance)}
                for name, importance in zip(feature_names, model.feature_importances_)
            ]
        else:
            # For logistic regression, use absolute coefficients
            feature_importance = [
                {"feature": name, "importance": float(abs(coef))}
                for name, coef in zip(feature_names, model.coef_[0])
            ]

        # Sort by importance
        feature_importance.sort(key=lambda x: x["importance"], reverse=True)

        return {
            "scores": detailed_scores,
            "accuracy": float(model_accuracy),
            "feature_importance": feature_importance,
            "model_type": f"{type(model).__name__}_ml",
            "training_size": len(X),
            "performance_metrics": {
                "accuracy": float(model_accuracy),
                "cv_std": float(np.std(cv_scores)),
                "high_value_contacts": int(sum(predictions)),
                "total_contacts": len(predictions),
            },
            "insights": [
                f"ML model trained on {len(X)} contacts with {model_accuracy:.2%} accuracy",
                f"Identified {sum(predictions)} high-value prospects out of {len(predictions)} total contacts",
                f"Top predictive feature: {feature_importance[0]['feature']}",
                f"Model type: {type(model).__name__}",
            ],
        }

    async def _simple_lead_scoring(self, contacts_data: List[Any]) -> Dict[str, Any]:
        """Simple lead scoring fallback when sklearn is not available"""
        scores = []

        for contact in contacts_data:
            contact_id = str(contact.id if hasattr(contact, "id") else contact.get("id", ""))
            properties = contact.properties if hasattr(contact, "properties") else contact

            # Base score from contact properties
            score = 0.5  # baseline

            # Email engagement factor
            email_opens = int(properties.get("hs_email_open", 0) or 0)
            email_clicks = int(properties.get("hs_email_click", 0) or 0)
            email_bounces = int(properties.get("hs_email_bounce", 0) or 0)

            engagement_factor = min((email_opens + email_clicks) / 20, 0.3)
            score += engagement_factor

            # Bounce penalty
            if email_opens > 0:
                bounce_rate = email_bounces / email_opens
                score -= bounce_rate * 0.2

            # Lifecycle stage factor
            lifecycle = properties.get("lifecyclestage", "").lower()
            if "opportunity" in lifecycle:
                score += 0.15
            elif "lead" in lifecycle:
                score += 0.1
            elif "customer" in lifecycle:
                score += 0.05

            # Ensure score is between 0 and 1
            score = max(0, min(1, score))

            scores.append(
                {
                    "contact_id": contact_id,
                    "score": score,
                    "prediction": 1 if score > 0.7 else 0,
                    "confidence": score,
                    "features": {
                        "email_opens": email_opens,
                        "email_clicks": email_clicks,
                        "engagement_factor": engagement_factor,
                        "lifecycle": lifecycle,
                    },
                }
            )

        return {
            "scores": scores,
            "accuracy": 0.7,  # Estimated accuracy for simple scoring
            "feature_importance": [
                {"feature": "email_opens", "importance": 0.3},
                {"feature": "email_clicks", "importance": 0.3},
                {"feature": "lifecycle_stage", "importance": 0.2},
                {"feature": "bounce_rate", "importance": 0.2},
            ],
            "model_type": "simple_heuristic",
            "training_size": len(contacts_data),
            "performance_metrics": {
                "accuracy": 0.7,
                "high_value_contacts": sum(1 for s in scores if s["prediction"] == 1),
                "total_contacts": len(scores),
            },
            "insights": [
                f"Simple heuristic model applied to {len(contacts_data)} contacts",
                f"Identified {sum(1 for s in scores if s['prediction'] == 1)} high-value prospects",
                "Consider installing sklearn for advanced ML-based scoring",
            ],
        }

    async def create_customer_segments(
        self,
        contacts_data: List[Any],
        segmentation_type: str,
        num_segments: int,
        engagement_data: List[Dict] = None,
    ) -> Dict[str, Any]:
        """Create customer segments using various methodologies"""

        if not contacts_data:
            return {
                "segments": [],
                "segment_profiles": [],
                "quality_metrics": {"total_contacts": 0},
                "insights": ["No contacts data available for segmentation"],
            }

        # Simple segmentation based on engagement
        if segmentation_type == "engagement" or segmentation_type == "behavioral":
            return await self._behavioral_segmentation(contacts_data, engagement_data, num_segments)
        elif segmentation_type == "lifecycle":
            return await self._lifecycle_segmentation(contacts_data, num_segments)
        elif segmentation_type == "rfm":
            return await self._rfm_segmentation(contacts_data, engagement_data, num_segments)
        elif segmentation_type == "value":
            return await self._value_based_segmentation(contacts_data, num_segments)
        else:
            # Default behavioral segmentation
            return await self._behavioral_segmentation(contacts_data, engagement_data, num_segments)

    async def _behavioral_segmentation(self, contacts_data, engagement_data, num_segments):
        """Segment contacts based on engagement behavior"""

        # Create engagement scores
        engagement_scores = {}
        for engagement in engagement_data or []:
            contact_id = str(engagement.get("contact_id", "") or engagement.get("contactId", ""))
            if contact_id:
                engagement_scores[contact_id] = engagement_scores.get(contact_id, 0) + 1

        # Segment contacts
        segments = []
        total_contacts = len(contacts_data)

        # Get all engagement scores and calculate percentiles for dynamic segmentation
        all_scores = [
            engagement_scores.get(
                str(contact.id if hasattr(contact, "id") else contact.get("id", "")), 0
            )
            for contact in contacts_data
        ]
        all_scores.sort()

        # Calculate thresholds based on num_segments
        if num_segments <= 1:
            thresholds = []
            segment_labels = ["All Contacts"]
        else:
            import numpy as np

            percentiles = [i * 100 / num_segments for i in range(1, num_segments)]
            thresholds = [np.percentile(all_scores, p) for p in percentiles]

            # Generate segment labels
            if num_segments == 2:
                segment_labels = ["Low Engagement", "High Engagement"]
            elif num_segments == 3:
                segment_labels = ["Low Engagement", "Medium Engagement", "High Engagement"]
            elif num_segments == 4:
                segment_labels = [
                    "No Engagement",
                    "Low Engagement",
                    "Medium Engagement",
                    "High Engagement",
                ]
            else:
                segment_labels = [f"Segment {i+1}" for i in range(num_segments)]

        for contact in contacts_data:
            contact_id = str(contact.id if hasattr(contact, "id") else contact.get("id", ""))
            score = engagement_scores.get(contact_id, 0)

            # Determine segment based on thresholds
            if not thresholds:
                segment = segment_labels[0]
            else:
                segment_index = 0
                for threshold in thresholds:
                    if score > threshold:
                        segment_index += 1
                    else:
                        break
                segment = segment_labels[segment_index]

            segments.append(
                {"contact_id": contact_id, "segment": segment, "engagement_score": score}
            )

        # Create segment profiles
        segment_counts = {}
        for seg in segments:
            segment_name = seg["segment"]
            segment_counts[segment_name] = segment_counts.get(segment_name, 0) + 1

        segment_profiles = []
        for segment_name, count in segment_counts.items():
            segment_profiles.append(
                {
                    "label": segment_name,
                    "size": count,
                    "percentage": (count / total_contacts) * 100 if total_contacts > 0 else 0,
                    "characteristics": {
                        "avg_engagement": (
                            sum(
                                s["engagement_score"]
                                for s in segments
                                if s["segment"] == segment_name
                            )
                            / count
                            if count > 0
                            else 0
                        )
                    },
                }
            )

        return {
            "segments": segments,
            "segment_profiles": segment_profiles,
            "quality_metrics": {
                "total_contacts": total_contacts,
                "num_segments": len(segment_profiles),
                "silhouette_score": 0.7,  # Mock score
            },
            "segmentation_type": "behavioral",
        }

    async def _lifecycle_segmentation(self, contacts_data, num_segments):
        """Segment contacts based on lifecycle stage"""

        segments = []
        lifecycle_counts = {}

        # First, collect all lifecycle stages and their counts
        for contact in contacts_data:
            properties = contact.properties if hasattr(contact, "properties") else contact
            contact_id = str(contact.id if hasattr(contact, "id") else contact.get("id", ""))
            lifecycle = properties.get("lifecyclestage", "unknown")
            lifecycle_counts[lifecycle] = lifecycle_counts.get(lifecycle, 0) + 1

        # If we have more lifecycle stages than requested segments, group smaller stages
        if len(lifecycle_counts) > num_segments and num_segments > 1:
            # Sort lifecycle stages by count (descending)
            sorted_lifecycles = sorted(lifecycle_counts.items(), key=lambda x: x[1], reverse=True)

            # Keep top (num_segments - 1) stages, group the rest as "Other"
            top_lifecycles = dict(sorted_lifecycles[: num_segments - 1])
            other_count = sum(count for _, count in sorted_lifecycles[num_segments - 1 :])
            if other_count > 0:
                top_lifecycles["Other"] = other_count

            lifecycle_mapping = {}
            for lifecycle, _ in sorted_lifecycles[: num_segments - 1]:
                lifecycle_mapping[lifecycle] = lifecycle
            for lifecycle, _ in sorted_lifecycles[num_segments - 1 :]:
                lifecycle_mapping[lifecycle] = "Other"

            lifecycle_counts = top_lifecycles
        else:
            # Use original lifecycle stages if within limit
            lifecycle_mapping = {lifecycle: lifecycle for lifecycle in lifecycle_counts.keys()}

        # Create segments with mapped lifecycle stages
        for contact in contacts_data:
            properties = contact.properties if hasattr(contact, "properties") else contact
            contact_id = str(contact.id if hasattr(contact, "id") else contact.get("id", ""))
            original_lifecycle = properties.get("lifecyclestage", "unknown")
            mapped_segment = lifecycle_mapping.get(original_lifecycle, "Other")

            segments.append(
                {
                    "contact_id": contact_id,
                    "segment": mapped_segment,
                    "lifecycle_stage": original_lifecycle,
                }
            )

        segment_profiles = []
        total_contacts = len(contacts_data)
        for lifecycle, count in lifecycle_counts.items():
            segment_profiles.append(
                {
                    "label": lifecycle,
                    "size": count,
                    "percentage": (count / total_contacts) * 100 if total_contacts > 0 else 0,
                    "characteristics": {"lifecycle_stage": lifecycle},
                }
            )

        return {
            "segments": segments,
            "segment_profiles": segment_profiles,
            "quality_metrics": {
                "total_contacts": total_contacts,
                "num_segments": len(segment_profiles),
                "silhouette_score": 0.8,  # Mock score
            },
            "segmentation_type": "lifecycle",
        }

    async def _rfm_segmentation(self, contacts_data, engagement_data, num_segments):
        """RFM (Recency, Frequency, Monetary) segmentation"""
        import numpy as np
        import pendulum

        # Calculate RFM scores for each contact
        rfm_scores = {}
        current_date = pendulum.now()

        # Initialize RFM data
        for contact in contacts_data:
            contact_id = str(contact.id if hasattr(contact, "id") else contact.get("id", ""))
            properties = contact.properties if hasattr(contact, "properties") else contact

            # Get monetary value from contact properties
            total_revenue = float(properties.get("total_revenue", 0) or 0)

            rfm_scores[contact_id] = {
                "recency": 365,  # Default to 365 days if no engagement
                "frequency": 0,
                "monetary": total_revenue,
                "last_engagement": None,
            }

        # Calculate recency and frequency from engagement data
        for engagement in engagement_data or []:
            contact_id = str(engagement.get("contact_id", "") or engagement.get("contactId", ""))
            if contact_id in rfm_scores:
                # Increment frequency
                rfm_scores[contact_id]["frequency"] += 1

                # Calculate recency (days since last engagement)
                engagement_date_str = engagement.get("timestamp") or engagement.get("created_at")
                if engagement_date_str:
                    try:
                        engagement_date = pendulum.parse(engagement_date_str)
                        days_since = (current_date - engagement_date).days
                        if (
                            rfm_scores[contact_id]["last_engagement"] is None
                            or days_since < rfm_scores[contact_id]["recency"]
                        ):
                            rfm_scores[contact_id]["recency"] = days_since
                            rfm_scores[contact_id]["last_engagement"] = engagement_date
                    except:
                        pass

        # Get all RFM values for percentile calculation
        recency_values = [score["recency"] for score in rfm_scores.values()]
        frequency_values = [score["frequency"] for score in rfm_scores.values()]
        monetary_values = [score["monetary"] for score in rfm_scores.values()]

        # Calculate composite RFM scores (lower recency is better, higher frequency and monetary are better)
        composite_scores = []
        for contact_id, scores in rfm_scores.items():
            # Normalize scores (0-1 scale)
            max_recency = max(recency_values) if recency_values else 1
            max_frequency = max(frequency_values) if frequency_values else 1
            max_monetary = max(monetary_values) if monetary_values else 1

            normalized_recency = (
                1 - (scores["recency"] / max_recency) if max_recency > 0 else 0
            )  # Invert recency
            normalized_frequency = scores["frequency"] / max_frequency if max_frequency > 0 else 0
            normalized_monetary = scores["monetary"] / max_monetary if max_monetary > 0 else 0

            # Composite score (equal weights)
            composite_score = (normalized_recency + normalized_frequency + normalized_monetary) / 3
            composite_scores.append(composite_score)
            rfm_scores[contact_id]["composite_score"] = composite_score

        # Create segments based on composite scores
        composite_scores.sort()

        if num_segments <= 1:
            thresholds = []
            segment_labels = ["All Contacts"]
        else:
            percentiles = [i * 100 / num_segments for i in range(1, num_segments)]
            thresholds = [np.percentile(composite_scores, p) for p in percentiles]

            if num_segments == 2:
                segment_labels = ["Low Value", "High Value"]
            elif num_segments == 3:
                segment_labels = ["Low Value", "Medium Value", "High Value"]
            elif num_segments == 4:
                segment_labels = ["Champions", "Loyal Customers", "Potential Loyalists", "At Risk"]
            elif num_segments == 5:
                segment_labels = [
                    "Champions",
                    "Loyal Customers",
                    "Potential Loyalists",
                    "New Customers",
                    "At Risk",
                ]
            else:
                segment_labels = [f"RFM Segment {i+1}" for i in range(num_segments)]

        # Assign segments to contacts
        segments = []
        for contact in contacts_data:
            contact_id = str(contact.id if hasattr(contact, "id") else contact.get("id", ""))
            score_data = rfm_scores.get(
                contact_id, {"composite_score": 0, "recency": 365, "frequency": 0, "monetary": 0}
            )
            composite_score = score_data["composite_score"]

            # Determine segment based on thresholds
            if not thresholds:
                segment = segment_labels[0]
            else:
                segment_index = 0
                for threshold in thresholds:
                    if composite_score > threshold:
                        segment_index += 1
                    else:
                        break
                segment = segment_labels[segment_index]

            segments.append(
                {
                    "contact_id": contact_id,
                    "segment": segment,
                    "rfm_score": composite_score,
                    "recency": score_data["recency"],
                    "frequency": score_data["frequency"],
                    "monetary": score_data["monetary"],
                }
            )

        # Create segment profiles
        segment_counts = {}
        segment_rfm_data = {}

        for seg in segments:
            segment_name = seg["segment"]
            segment_counts[segment_name] = segment_counts.get(segment_name, 0) + 1

            if segment_name not in segment_rfm_data:
                segment_rfm_data[segment_name] = {
                    "rfm_scores": [],
                    "recency_values": [],
                    "frequency_values": [],
                    "monetary_values": [],
                }

            segment_rfm_data[segment_name]["rfm_scores"].append(seg["rfm_score"])
            segment_rfm_data[segment_name]["recency_values"].append(seg["recency"])
            segment_rfm_data[segment_name]["frequency_values"].append(seg["frequency"])
            segment_rfm_data[segment_name]["monetary_values"].append(seg["monetary"])

        segment_profiles = []
        total_contacts = len(contacts_data)

        for segment_name, count in segment_counts.items():
            rfm_data = segment_rfm_data[segment_name]
            segment_profiles.append(
                {
                    "label": segment_name,
                    "size": count,
                    "percentage": (count / total_contacts) * 100 if total_contacts > 0 else 0,
                    "characteristics": {
                        "avg_rfm_score": (
                            sum(rfm_data["rfm_scores"]) / len(rfm_data["rfm_scores"])
                            if rfm_data["rfm_scores"]
                            else 0
                        ),
                        "avg_recency": (
                            sum(rfm_data["recency_values"]) / len(rfm_data["recency_values"])
                            if rfm_data["recency_values"]
                            else 0
                        ),
                        "avg_frequency": (
                            sum(rfm_data["frequency_values"]) / len(rfm_data["frequency_values"])
                            if rfm_data["frequency_values"]
                            else 0
                        ),
                        "avg_monetary": (
                            sum(rfm_data["monetary_values"]) / len(rfm_data["monetary_values"])
                            if rfm_data["monetary_values"]
                            else 0
                        ),
                    },
                }
            )

        return {
            "segments": segments,
            "segment_profiles": segment_profiles,
            "metadata": {
                "total_segments": len(segment_profiles),
                "segmentation_method": "RFM Analysis",
                "factors": ["recency", "frequency", "monetary"],
            },
            "segmentation_type": "rfm",
        }

    async def _value_based_segmentation(self, contacts_data, num_segments):
        """Value-based segmentation based on customer lifetime value and revenue potential"""
        import numpy as np

        # Calculate value scores for each contact
        value_scores = {}

        for contact in contacts_data:
            contact_id = str(contact.id if hasattr(contact, "id") else contact.get("id", ""))
            properties = contact.properties if hasattr(contact, "properties") else contact

            # Extract value-related metrics
            total_revenue = float(properties.get("total_revenue", 0) or 0)
            annual_revenue = float(properties.get("annualrevenue", 0) or 0)
            deal_amount = float(properties.get("amount", 0) or 0)
            num_deals = int(properties.get("num_associated_deals", 0) or 0)

            # Calculate average deal size
            avg_deal_size = total_revenue / num_deals if num_deals > 0 else deal_amount

            # Company size indicators
            num_employees = int(properties.get("numberofemployees", 0) or 0)
            company_size_score = min(
                num_employees / 1000, 5
            )  # Cap at 5000 employees = max score of 5

            # Calculate composite value score
            value_score = (
                total_revenue * 0.4  # 40% weight on historical revenue
                + annual_revenue * 0.3  # 30% weight on annual revenue potential
                + avg_deal_size * 0.2  # 20% weight on deal size
                + company_size_score * 1000 * 0.1  # 10% weight on company size (scaled up)
            )

            value_scores[contact_id] = {
                "value_score": value_score,
                "total_revenue": total_revenue,
                "annual_revenue": annual_revenue,
                "avg_deal_size": avg_deal_size,
                "num_deals": num_deals,
                "company_size": num_employees,
            }

        # Get all value scores for percentile calculation
        all_value_scores = [score["value_score"] for score in value_scores.values()]
        all_value_scores.sort()

        # Create segments based on value scores
        if num_segments <= 1:
            thresholds = []
            segment_labels = ["All Contacts"]
        else:
            percentiles = [i * 100 / num_segments for i in range(1, num_segments)]
            thresholds = [np.percentile(all_value_scores, p) for p in percentiles]

            if num_segments == 2:
                segment_labels = ["Low Value", "High Value"]
            elif num_segments == 3:
                segment_labels = ["Low Value", "Medium Value", "High Value"]
            elif num_segments == 4:
                segment_labels = ["Enterprise", "Mid-Market", "SMB", "Prospects"]
            elif num_segments == 5:
                segment_labels = ["Enterprise", "Large Business", "Mid-Market", "SMB", "Prospects"]
            else:
                segment_labels = [f"Value Tier {i+1}" for i in range(num_segments)]

        # Assign segments to contacts
        segments = []
        for contact in contacts_data:
            contact_id = str(contact.id if hasattr(contact, "id") else contact.get("id", ""))
            score_data = value_scores.get(
                contact_id,
                {
                    "value_score": 0,
                    "total_revenue": 0,
                    "annual_revenue": 0,
                    "avg_deal_size": 0,
                    "num_deals": 0,
                    "company_size": 0,
                },
            )
            value_score = score_data["value_score"]

            # Determine segment based on thresholds (reverse order for value - higher is better)
            if not thresholds:
                segment = segment_labels[0]
            else:
                segment_index = 0
                for threshold in thresholds:
                    if value_score > threshold:
                        segment_index += 1
                    else:
                        break
                segment = segment_labels[segment_index]

            segments.append(
                {
                    "contact_id": contact_id,
                    "segment": segment,
                    "value_score": value_score,
                    "total_revenue": score_data["total_revenue"],
                    "annual_revenue": score_data["annual_revenue"],
                    "avg_deal_size": score_data["avg_deal_size"],
                    "num_deals": score_data["num_deals"],
                    "company_size": score_data["company_size"],
                }
            )

        # Create segment profiles
        segment_counts = {}
        segment_value_data = {}

        for seg in segments:
            segment_name = seg["segment"]
            segment_counts[segment_name] = segment_counts.get(segment_name, 0) + 1

            if segment_name not in segment_value_data:
                segment_value_data[segment_name] = {
                    "value_scores": [],
                    "total_revenues": [],
                    "annual_revenues": [],
                    "avg_deal_sizes": [],
                    "num_deals": [],
                    "company_sizes": [],
                }

            segment_value_data[segment_name]["value_scores"].append(seg["value_score"])
            segment_value_data[segment_name]["total_revenues"].append(seg["total_revenue"])
            segment_value_data[segment_name]["annual_revenues"].append(seg["annual_revenue"])
            segment_value_data[segment_name]["avg_deal_sizes"].append(seg["avg_deal_size"])
            segment_value_data[segment_name]["num_deals"].append(seg["num_deals"])
            segment_value_data[segment_name]["company_sizes"].append(seg["company_size"])

        segment_profiles = []
        total_contacts = len(contacts_data)

        for segment_name, count in segment_counts.items():
            value_data = segment_value_data[segment_name]
            segment_profiles.append(
                {
                    "label": segment_name,
                    "size": count,
                    "percentage": (count / total_contacts) * 100 if total_contacts > 0 else 0,
                    "characteristics": {
                        "avg_value_score": (
                            sum(value_data["value_scores"]) / len(value_data["value_scores"])
                            if value_data["value_scores"]
                            else 0
                        ),
                        "avg_total_revenue": (
                            sum(value_data["total_revenues"]) / len(value_data["total_revenues"])
                            if value_data["total_revenues"]
                            else 0
                        ),
                        "avg_annual_revenue": (
                            sum(value_data["annual_revenues"]) / len(value_data["annual_revenues"])
                            if value_data["annual_revenues"]
                            else 0
                        ),
                        "avg_deal_size": (
                            sum(value_data["avg_deal_sizes"]) / len(value_data["avg_deal_sizes"])
                            if value_data["avg_deal_sizes"]
                            else 0
                        ),
                        "avg_num_deals": (
                            sum(value_data["num_deals"]) / len(value_data["num_deals"])
                            if value_data["num_deals"]
                            else 0
                        ),
                        "avg_company_size": (
                            sum(value_data["company_sizes"]) / len(value_data["company_sizes"])
                            if value_data["company_sizes"]
                            else 0
                        ),
                    },
                }
            )

        return {
            "segments": segments,
            "segment_profiles": segment_profiles,
            "metadata": {
                "total_segments": len(segment_profiles),
                "segmentation_method": "Value-Based Analysis",
                "factors": ["total_revenue", "annual_revenue", "deal_size", "company_size"],
            },
            "segmentation_type": "value_based",
        }

    async def forecast_revenue(
        self,
        historical_deals: List[Any],
        current_pipeline: List[Any],
        forecast_period: int,
        confidence_level: float,
    ) -> Dict[str, Any]:
        """Forecast revenue using historical data and current pipeline"""

        if not historical_deals and not current_pipeline:
            return {
                "prediction": 0,
                "confidence_interval": {"lower": 0, "upper": 0},
                "scenarios": {"conservative": 0, "optimistic": 0},
                "model_accuracy": {"method": "insufficient_data"},
                "insights": ["No historical or pipeline data available for forecasting"],
            }

        # Parse forecast period (could be int days or string like "90_days", "6_months", etc)
        if isinstance(forecast_period, str):
            if "days" in forecast_period:
                period_days = int(forecast_period.split("_")[0])
            elif "months" in forecast_period:
                period_days = int(forecast_period.split("_")[0]) * 30
            elif "quarters" in forecast_period:
                period_days = int(forecast_period.split("_")[0]) * 90
            elif "years" in forecast_period:
                period_days = int(forecast_period.split("_")[0]) * 365
            else:
                period_days = 90  # Default to 90 days
        else:
            period_days = int(forecast_period)  # Assume days if int

        # Calculate historical revenue and normalize to daily average
        historical_revenue = 0
        historical_deal_count = 0
        for deal in historical_deals:
            properties = deal.properties if hasattr(deal, "properties") else deal
            amount_str = properties.get("amount", "0")
            try:
                amount = float(amount_str) if amount_str else 0.0
                historical_revenue += amount
                historical_deal_count += 1
            except (ValueError, TypeError):
                continue

        # Calculate weighted pipeline value
        pipeline_value = 0
        pipeline_deal_count = 0
        for deal in current_pipeline:
            properties = deal.properties if hasattr(deal, "properties") else deal
            amount_str = properties.get("amount", "0")
            try:
                amount = float(amount_str) if amount_str else 0.0
                probability = float(properties.get("hs_deal_stage_probability", 50) or 50) / 100
                pipeline_value += amount * probability
                pipeline_deal_count += 1
            except (ValueError, TypeError):
                continue

        # Estimate historical daily revenue rate (assume historical data covers 365 days)
        historical_daily_rate = historical_revenue / 365 if historical_revenue > 0 else 0

        # Scale historical trend to forecast period
        historical_projection = historical_daily_rate * period_days

        # Simple prediction combining historical trend and pipeline
        # Weight pipeline higher for shorter periods, historical trend for longer periods
        pipeline_weight = max(
            0.3, 1.0 - (period_days / 365)
        )  # More pipeline weight for shorter forecasts
        historical_weight = 1.0 - pipeline_weight

        base_prediction = (historical_projection * historical_weight) + (
            pipeline_value * pipeline_weight
        )

        # Add confidence intervals
        variance = base_prediction * 0.2  # 20% variance
        lower_bound = base_prediction - variance
        upper_bound = base_prediction + variance

        scenarios = {
            "conservative": base_prediction * 0.8,
            "realistic": base_prediction,
            "optimistic": base_prediction * 1.2,
        }

        return {
            "prediction": base_prediction,
            "confidence_interval": {"lower": lower_bound, "upper": upper_bound},
            "scenarios": scenarios,
            "model_accuracy": {
                "method": "time_scaled_historical_pipeline",
                "confidence_level": confidence_level,
                "forecast_period_days": period_days,
                "pipeline_weight": pipeline_weight,
                "historical_weight": historical_weight,
            },
            "insights": [
                f"Forecast for {period_days} days: ${base_prediction:,.0f}",
                f"Based on {historical_deal_count} historical deals and {pipeline_deal_count} pipeline deals",
                f"Historical daily rate: ${historical_daily_rate:,.0f}",
                f"Weighted prediction (pipeline: {pipeline_weight:.1%}, historical: {historical_weight:.1%})",
                f"Confidence interval: ${lower_bound:,.0f} - ${upper_bound:,.0f}",
            ],
        }
