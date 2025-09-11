"""
Insight Generator Module for MCP HubSpot Connector

This module contains the InsightGenerator class responsible for generating
business insights and actionable recommendations from analytics data.
"""

from typing import Any, Dict, List


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
