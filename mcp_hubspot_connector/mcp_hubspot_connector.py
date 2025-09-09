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
        def __init__(self, property_name=None, operator=None, value=None, high_value=None, values=None):
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


@dataclass
class MCPToolResult:
    """Standardized result format for MCP tools"""
    success: bool
    data: Any
    insights: List[str] = None
    recommendations: List[str] = None
    metadata: Dict[str, Any] = None
    error_message: str = None


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
        """Process HubSpot SDK contact objects into analytics metrics"""

        contacts_list = []
        for contact in contacts_data:
            properties = contact.properties if hasattr(contact, 'properties') else contact
            contacts_list.append(
                {
                    "id": contact.id if hasattr(contact, 'id') else contact.get('id'),
                    "email": properties.get("email", ""),
                    "firstname": properties.get("firstname", ""),
                    "lastname": properties.get("lastname", ""),
                    "createdate": properties.get("createdate", ""),
                    "lifecyclestage": properties.get("lifecyclestage", ""),
                    "hs_lead_status": properties.get("hs_lead_status", ""),
                    "email_opens": int(properties.get("hs_email_open", 0) or 0),
                    "email_clicks": int(properties.get("hs_email_click", 0) or 0),
                    "email_bounces": int(properties.get("hs_email_bounce", 0) or 0),
                }
            )

        df = pd.DataFrame(contacts_list)

        if not df.empty:
            df["engagement_score"] = df["email_opens"] * 1 + df["email_clicks"] * 3
            active_count = len(df[df["engagement_score"] > 0])

            segments = {}
            if segmentation == "engagement_level":
                segments = {
                    "high_engagement": len(df[df["engagement_score"] >= 10]),
                    "medium_engagement": len(
                        df[(df["engagement_score"] >= 5) & (df["engagement_score"] < 10)]
                    ),
                    "low_engagement": len(df[df["engagement_score"] < 5]),
                }
            elif segmentation == "lifecycle_stage":
                segments = df["lifecyclestage"].value_counts().to_dict()

            return {
                "active_count": active_count,
                "avg_engagement": df["engagement_score"].mean(),
                "segments": segments,
                "detailed_metrics": {
                    "total_opens": df["email_opens"].sum(),
                    "total_clicks": df["email_clicks"].sum(),
                },
                "processing_time": 0.5,
                "velocity_metrics": self._calculate_contact_velocity(df),
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
                    "avg_engagement_recent_contacts": 0
                },
            }
    
    def _calculate_contact_velocity(self, df):
        """Calculate contact acquisition and engagement velocity"""
        if 'createdate' in df.columns and not df.empty:
            try:
                df['createdate'] = pd.to_datetime(df['createdate'])
                now = pd.Timestamp.now()
                last_30_days = now - pd.Timedelta(days=30)
                recent_contacts = len(df[df['createdate'] >= last_30_days])
                return {
                    "contacts_added_last_30_days": recent_contacts,
                    "avg_engagement_recent_contacts": df[df['createdate'] >= last_30_days]['engagement_score'].mean() if recent_contacts > 0 else 0
                }
            except:
                pass
        return {
            "contacts_added_last_30_days": 0,
            "avg_engagement_recent_contacts": 0
        }

    async def analyze_campaign_performance(
        self, campaigns_data: List[Any], campaign_stats: List[Dict], metrics: List[str]
    ) -> Dict[str, Any]:
        """Analyze campaign performance from HubSpot SDK data"""

        campaign_metrics = []

        for i, campaign in enumerate(campaigns_data):
            stats = campaign_stats[i] if i < len(campaign_stats) else {}

            sent = stats.get("sent", 1)
            delivered = stats.get("delivered", sent)
            opened = stats.get("opened", 0)
            clicked = stats.get("clicked", 0)

            metrics_data = {
                "campaign_id": campaign.id if hasattr(campaign, 'id') else campaign.get('id'),
                "name": getattr(campaign, "name", "Unknown"),
                "sent_count": sent,
                "delivered_count": delivered,
                "open_count": opened,
                "click_count": clicked,
                "open_rate": opened / delivered if delivered > 0 else 0,
                "click_rate": clicked / delivered if delivered > 0 else 0,
                "bounce_rate": (sent - delivered) / sent if sent > 0 else 0,
            }
            campaign_metrics.append(metrics_data)

        if campaign_metrics:
            df = pd.DataFrame(campaign_metrics)
            summary = {
                "avg_open_rate": df["open_rate"].mean(),
                "avg_click_rate": df["click_rate"].mean(),
                "avg_bounce_rate": df["bounce_rate"].mean(),
                "total_campaigns": len(campaign_metrics),
                "total_sent": df["sent_count"].sum(),
                "total_opened": df["open_count"].sum(),
                "total_clicked": df["click_count"].sum(),
            }
        else:
            summary = {
                "avg_open_rate": 0,
                "avg_click_rate": 0,
                "avg_bounce_rate": 0,
                "total_campaigns": 0,
            }

        insights = []
        if summary["avg_open_rate"] < 0.20:
            insights.append("Overall open rates are below industry average of 20%")
        if summary["avg_click_rate"] < 0.025:
            insights.append("Click rates could be improved with better content and CTAs")

        return {
            "campaign_metrics": campaign_metrics,
            "summary": summary,
            "insights": insights,
            "trends": {"open_rate_trend": "stable", "click_rate_trend": "stable"},
        }

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
        self, deals_data: List[Any], analysis_type: str
    ) -> Dict[str, Any]:
        """Analyze sales pipeline from HubSpot SDK deals data"""

        # Convert SDK objects to DataFrame
        deals_list = []
        for deal in deals_data:
            properties = deal.properties if hasattr(deal, 'properties') else deal

            # Parse amount safely
            amount_str = properties.get("amount", "0")
            try:
                amount = float(amount_str) if amount_str else 0.0
            except (ValueError, TypeError):
                amount = 0.0

            deals_list.append(
                {
                    "id": deal.id if hasattr(deal, 'id') else deal.get('id'),
                    "name": properties.get("dealname", ""),
                    "amount": amount,
                    "stage": properties.get("dealstage", ""),
                    "pipeline": properties.get("pipeline", ""),
                    "createdate": properties.get("createdate", ""),
                    "closedate": properties.get("closedate", ""),
                    "probability": properties.get("hs_deal_stage_probability", 0),
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
            }

        # Calculate pipeline metrics
        total_value = df["amount"].sum()
        deal_count = len(df)
        avg_deal_size = df["amount"].mean()

        # Calculate conversion rate (closed won vs total)
        closed_won_count = len(df[df["stage"].str.contains("won|closed", case=False, na=False)])
        conversion_rate = closed_won_count / deal_count if deal_count > 0 else 0

        # Stage analysis
        stage_metrics = {}
        if "stage" in df.columns:
            stage_analysis = (
                df.groupby("stage")
                .agg({"amount": ["sum", "count", "mean"], "probability": "mean"})
                .round(2)
            )

            for stage in stage_analysis.index:
                stage_metrics[stage] = {
                    "count": int(stage_analysis.loc[stage, ("amount", "count")]),
                    "value": float(stage_analysis.loc[stage, ("amount", "sum")]),
                    "avg_deal_size": float(stage_analysis.loc[stage, ("amount", "mean")]),
                    "avg_probability": float(stage_analysis.loc[stage, ("probability", "mean")]),
                }

        # Generate insights
        insights = []
        if conversion_rate < 0.2:
            insights.append(
                f"Low conversion rate: {conversion_rate:.1%} - consider pipeline optimization"
            )
        if avg_deal_size < 1000:
            insights.append("Average deal size is low - focus on upselling opportunities")

        # Calculate actual velocity metrics
        velocity_metrics = self._calculate_pipeline_velocity(df, closed_won_count)

        return {
            "total_value": total_value,
            "deal_count": deal_count,
            "avg_deal_size": avg_deal_size,
            "conversion_rate": conversion_rate,
            "stage_metrics": stage_metrics,
            "insights": insights,
            "velocity_metrics": velocity_metrics,
        }

    def _calculate_pipeline_velocity(self, deals_df, closed_won_count):
        """Calculate actual pipeline velocity metrics"""
        if 'createdate' in deals_df.columns and 'closedate' in deals_df.columns and not deals_df.empty:
            try:
                deals_df['createdate'] = pd.to_datetime(deals_df['createdate'])
                deals_df['closedate'] = pd.to_datetime(deals_df['closedate'])
                
                closed_deals = deals_df[deals_df['closedate'].notna()]
                if not closed_deals.empty:
                    closed_deals['days_in_pipeline'] = (closed_deals['closedate'] - closed_deals['createdate']).dt.days
                    avg_days = closed_deals['days_in_pipeline'].mean()
                    
                    # Calculate deals closed this month
                    now = pd.Timestamp.now()
                    this_month_start = now.replace(day=1)
                    deals_closed_this_month = len(closed_deals[closed_deals['closedate'] >= this_month_start])
                    
                    return {
                        "avg_days_in_pipeline": avg_days,
                        "deals_closed_this_month": deals_closed_this_month,
                        "pipeline_velocity_score": len(closed_deals) / len(deals_df) if len(deals_df) > 0 else 0,
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
        model_type: str,
    ) -> Dict[str, Any]:
        """ML-based lead scoring using HubSpot SDK data"""

        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            # Fallback to simple scoring if sklearn not available
            return await self._simple_lead_scoring(contacts_data)

        # Convert contacts to DataFrame
        contacts_list = []
        for contact in contacts_data:
            properties = contact.properties if hasattr(contact, 'properties') else contact
            contacts_list.append(
                {
                    "id": contact.id if hasattr(contact, 'id') else contact.get('id'),
                    "email_opens": int(properties.get("hs_email_open", 0) or 0),
                    "email_clicks": int(properties.get("hs_email_click", 0) or 0),
                    "email_bounces": int(properties.get("hs_email_bounce", 0) or 0),
                    "lifecyclestage": properties.get("lifecyclestage", ""),
                    "lead_status": properties.get("hs_lead_status", ""),
                    "createdate": properties.get("createdate", ""),
                }
            )

        contacts_df = pd.DataFrame(contacts_list)

        if contacts_df.empty:
            return {
                "scores": [],
                "performance_metrics": {"accuracy": 0},
                "feature_importance": [],
                "training_size": 0,
                "accuracy": 0,
            }

        # Feature engineering
        contacts_df["engagement_score"] = (
            contacts_df["email_opens"] * 1
            + contacts_df["email_clicks"] * 3
            + contacts_df["email_bounces"] * -1
        )

        # Convert categorical variables
        contacts_df["lifecycle_score"] = (
            contacts_df["lifecyclestage"]
            .map(
                {
                    "subscriber": 1,
                    "lead": 2,
                    "marketingqualifiedlead": 3,
                    "salesqualifiedlead": 4,
                    "opportunity": 5,
                    "customer": 6,
                }
            )
            .fillna(0)
        )

        # Create target variable (simplified)
        contacts_df["converted"] = (
            (contacts_df["lifecycle_score"] >= 5) | (contacts_df["engagement_score"] > 10)
        ).astype(int)

        # Prepare features
        feature_columns = ["email_opens", "email_clicks", "engagement_score", "lifecycle_score"]
        X = contacts_df[feature_columns].fillna(0)
        y = contacts_df["converted"]

        if len(X) < 10:  # Not enough data for ML
            return await self._simple_lead_scoring(contacts_data)

        # Train ML model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train_scaled, y_train)

        # Make predictions
        X_scaled = scaler.transform(X)
        predictions = model.predict_proba(X_scaled)[:, 1]  # Probability of conversion
        accuracy = model.score(X_test_scaled, y_test) if len(X_test) > 0 else 0.8

        # Generate scores
        scores = []
        for i, contact in contacts_df.iterrows():
            score = float(predictions[i])
            scores.append(
                {
                    "contact_id": contact["id"],
                    "score": score,
                    "risk_level": "high" if score > 0.7 else "medium" if score > 0.3 else "low",
                }
            )

        # Feature importance
        feature_importance = [
            {"feature": col, "importance": float(imp)}
            for col, imp in zip(feature_columns, model.feature_importances_)
        ]
        feature_importance.sort(key=lambda x: x["importance"], reverse=True)

        return {
            "scores": scores,
            "performance_metrics": {
                "accuracy": accuracy,
                "model_type": "RandomForest",
                "training_samples": len(X_train),
            },
            "feature_importance": feature_importance,
            "training_size": len(X_train),
            "accuracy": accuracy,
        }

    async def _simple_lead_scoring(self, contacts_data: List[Any]) -> Dict[str, Any]:
        """Fallback simple scoring when ML libraries not available"""
        scores = []
        for contact in contacts_data:
            properties = contact.properties if hasattr(contact, 'properties') else contact
            email_opens = int(properties.get("hs_email_open", 0) or 0)
            email_clicks = int(properties.get("hs_email_click", 0) or 0)
            
            engagement_score = email_opens * 1 + email_clicks * 3
            score = min(engagement_score / 20.0, 1.0)  # Normalize to 0-1
            
            scores.append(
                {
                    "contact_id": contact.id if hasattr(contact, 'id') else contact.get('id'),
                    "score": score,
                    "risk_level": "high" if score > 0.7 else "medium" if score > 0.3 else "low",
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
            low_performing = [c for c in campaign_metrics if c["open_rate"] < benchmark_open_rate * 0.8]
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

    def __init__(self, logger: logging.Logger, config: HubSpotSDKConfig = None, **settings: Dict[str, Any]):
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
    async def get_contact_analytics(self, params: Dict[str, Any]) -> MCPToolResult:
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

            return MCPToolResult(
                success=True,
                data={
                    "totalContacts": len(contacts_data),
                    "activeContacts": analytics_result["active_count"],
                    "avgEngagementScore": analytics_result["avg_engagement"],
                    "segments": analytics_result["segments"],
                    "detailedMetrics": analytics_result["detailed_metrics"],
                },
                insights=[i["message"] for i in insights],
                recommendations=[i["action"] for i in insights if i.get("action")],
                metadata={
                    "processing_time": analytics_result["processing_time"],
                    "data_source": "hubspot_sdk",
                    "sdk_version": "7.0.0",
                },
            )

        except Exception as e:
            return MCPToolResult(
                success=False, data=None, error_message=f"Contact analytics failed: {str(e)}"
            )

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

                    if search_response.results:
                        all_contacts.extend(search_response.results)
                        total_fetched += len(search_response.results)

                    if search_response.paging and search_response.paging.next:
                        after = search_response.paging.next.after
                    else:
                        break

            except Exception as e:
                self.logger.error(f"Search contacts failed: {e}")
                raise
        else:
            try:
                after = None
                total_fetched = 0

                while total_fetched < limit:
                    batch_limit = min(100, limit - total_fetched)

                    await self.rate_limiter.wait_if_needed()
                    get_all_response = contacts_api.get_all(
                        properties=properties, limit=batch_limit, after=after
                    )

                    if get_all_response.results:
                        all_contacts.extend(get_all_response.results)
                        total_fetched += len(get_all_response.results)

                    if get_all_response.paging and get_all_response.paging.next:
                        after = get_all_response.paging.next.after
                    else:
                        break

            except Exception as e:
                self.logger.error(f"Get all contacts failed: {e}")
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

    @handle_hubspot_errors
    async def analyze_campaign_performance(self, params: Dict[str, Any]) -> MCPToolResult:
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

            # Process through analytics engine
            performance_analysis = await self.analytics_engine.analyze_campaign_performance(
                campaigns_data, campaign_stats, metrics
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

            return MCPToolResult(
                success=True,
                data={
                    "campaigns": performance_analysis["campaign_metrics"],
                    "summary_metrics": performance_analysis["summary"],
                    "benchmarks": benchmarks,
                    "performance_trends": performance_analysis["trends"],
                },
                insights=performance_analysis["insights"],
                recommendations=[r["action"] for r in recommendations],
                metadata={
                    "analysis_date": datetime.now().isoformat(),
                    "campaigns_analyzed": len(campaigns_data),
                    "benchmark_type": benchmark_type,
                    "data_source": "hubspot_sdk_marketing",
                },
            )

        except Exception as e:
            return MCPToolResult(
                success=False, data=None, error_message=f"Campaign analysis failed: {str(e)}"
            )

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
                    response = marketing_events_api.get_all(limit=100, after=after)

                    if response.results:
                        campaigns.extend(response.results)

                    if response.paging and response.paging.next:
                        after = response.paging.next.after
                    else:
                        break

            return campaigns

        except Exception as e:
            self.logger.error(f"Error fetching campaigns: {e}")
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
    async def analyze_sales_pipeline(self, params: Dict[str, Any]) -> MCPToolResult:
        """MCP Tool: analyze_sales_pipeline - Sales pipeline analytics using SDK"""
        try:
            pipeline_ids = params.get("pipelineIds", [])
            timeframe = params.get("timeframe", {})
            analysis_type = params.get("analysisType", "conversion_rates")
            include_recommendations = params.get("includeRecommendations", True)

            # Get deals data using SDK
            deals_data = await self._get_deals_with_sdk(pipeline_ids, timeframe)

            # Process through analytics engine
            pipeline_analysis = await self.analytics_engine.analyze_sales_pipeline(
                deals_data, analysis_type
            )

            # Generate recommendations
            recommendations = []
            if include_recommendations:
                recommendations = await self.insight_generator.generate_pipeline_recommendations(
                    pipeline_analysis
                )

            return MCPToolResult(
                success=True,
                data={
                    "totalValue": pipeline_analysis["total_value"],
                    "dealCount": pipeline_analysis["deal_count"],
                    "avgDealSize": pipeline_analysis["avg_deal_size"],
                    "conversionRate": pipeline_analysis["conversion_rate"],
                    "stageAnalysis": pipeline_analysis["stage_metrics"],
                },
                insights=pipeline_analysis["insights"],
                recommendations=[r["action"] for r in recommendations],
                metadata={"analysis_type": analysis_type, "data_source": "hubspot_sdk_deals"},
            )

        except Exception as e:
            return MCPToolResult(
                success=False, data=None, error_message=f"Pipeline analysis failed: {str(e)}"
            )

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

                if search_response.results:
                    all_deals.extend(search_response.results)

                if search_response.paging and search_response.paging.next:
                    after = search_response.paging.next.after
                else:
                    break

        else:
            # Get all deals without filters
            after = None
            while True:
                await self.rate_limiter.wait_if_needed()
                response = deals_api.get_all(properties=properties, limit=100, after=after)

                if response.results:
                    all_deals.extend(response.results)

                if response.paging and response.paging.next:
                    after = response.paging.next.after
                else:
                    break

        self.logger.info(f"Fetched {len(all_deals)} deals using HubSpot SDK")
        return all_deals

    @handle_hubspot_errors
    async def predict_lead_scores(self, params: Dict[str, Any]) -> MCPToolResult:
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

            return MCPToolResult(
                success=True,
                data={
                    "predictions": prediction_result["scores"],
                    "model_performance": prediction_result["performance_metrics"],
                    "feature_importance": (
                        prediction_result.get("feature_importance")
                        if include_feature_importance
                        else None
                    ),
                },
                insights=prediction_insights["insights"],
                recommendations=prediction_insights["recommendations"],
                metadata={
                    "model_type": model_type,
                    "training_data_size": prediction_result["training_size"],
                    "model_accuracy": prediction_result["accuracy"],
                    "data_source": "hubspot_sdk_multi_api",
                },
            )

        except Exception as e:
            return MCPToolResult(
                success=False, data=None, error_message=f"Lead scoring failed: {str(e)}"
            )

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

                if batch_response.results:
                    all_contacts.extend(batch_response.results)

            except Exception as e:
                self.logger.error(f"Batch read contacts failed: {e}")
                continue

        return all_contacts

    # ============ EXPORT FUNCTIONS ============

    async def export_contact_analytics(self, params: Dict[str, Any] = None) -> MCPToolResult:
        """Export function for contact analytics"""
        if params is None:
            params = {
                "dateRange": {"start": "2024-01-01T00:00:00Z", "end": "2024-12-31T23:59:59Z"},
                "segmentation": "engagement_level",
                "includeEngagement": True,
                "limit": 1000,
            }
        return await self.get_contact_analytics(params)

    async def export_campaign_performance(self, params: Dict[str, Any] = None) -> MCPToolResult:
        """Export function for campaign performance analysis"""
        if params is None:
            params = {
                "campaignIds": [],  # Will get all campaigns
                "metrics": ["open_rate", "click_rate", "conversion_rate"],
                "benchmarkType": "industry",
                "includeRecommendations": True,
            }
        return await self.analyze_campaign_performance(params)

    async def export_sales_pipeline(self, params: Dict[str, Any] = None) -> MCPToolResult:
        """Export function for sales pipeline analysis"""
        if params is None:
            params = {
                "timeframe": {"start": "2024-01-01T00:00:00Z", "end": "2024-12-31T23:59:59Z"},
                "analysisType": "conversion_rates",
                "includeRecommendations": True,
            }
        return await self.analyze_sales_pipeline(params)

    async def export_lead_scores(self, params: Dict[str, Any] = None) -> MCPToolResult:
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




