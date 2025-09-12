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

    async def analyze_revenue_forecast(self, forecast_result: Dict[str, Any]) -> Dict[str, List[str]]:
        """Analyze revenue forecast and generate insights"""
        insights = []
        recommendations = []
        
        prediction = forecast_result.get("prediction", 0)
        confidence_interval = forecast_result.get("confidence_interval", {})
        scenarios = forecast_result.get("scenarios", {})
        model_accuracy = forecast_result.get("model_accuracy", {})
        
        # Generate insights based on forecast data
        if prediction > 0:
            insights.append(f"Revenue forecast: ${prediction:,.0f}")
            
            # Confidence interval analysis
            lower = confidence_interval.get("lower", 0)
            upper = confidence_interval.get("upper", 0)
            uncertainty = (upper - lower) / prediction * 100 if prediction > 0 else 0
            
            if uncertainty > 50:
                insights.append("High forecast uncertainty - consider gathering more data")
                recommendations.append("Improve data quality by updating deal probabilities and amounts")
            elif uncertainty > 30:
                insights.append("Moderate forecast uncertainty")
                recommendations.append("Review pipeline deals for accuracy")
            else:
                insights.append("Forecast shows good confidence level")
        
        # Scenario analysis
        if scenarios:
            conservative = scenarios.get("conservative", 0)
            optimistic = scenarios.get("optimistic", 0)
            
            insights.append(f"Conservative scenario: ${conservative:,.0f}")
            insights.append(f"Optimistic scenario: ${optimistic:,.0f}")
            
            if optimistic > prediction * 1.5:
                recommendations.append("Significant upside potential - focus on deal acceleration")
            
            if conservative < prediction * 0.7:
                recommendations.append("Downside risk present - develop contingency plans")
        
        # Model accuracy insights
        forecast_period_days = model_accuracy.get("forecast_period_days", 0)
        pipeline_weight = model_accuracy.get("pipeline_weight", 0)
        
        if pipeline_weight > 0.7:
            insights.append("Forecast heavily weighted on current pipeline")
            recommendations.append("Focus on advancing existing deals to improve forecast accuracy")
        elif pipeline_weight < 0.3:
            insights.append("Forecast relies more on historical trends")
            recommendations.append("Build stronger pipeline to improve predictability")
        
        if forecast_period_days > 90:
            recommendations.append("Long-term forecast - monitor and update regularly")
        elif forecast_period_days < 30:
            recommendations.append("Short-term forecast - focus on deal closure activities")
        
        return {
            "insights": insights,
            "recommendations": recommendations
        }

    async def analyze_segmentation(self, segmentation_result: Dict[str, Any]) -> Dict[str, List[str]]:
        """Analyze customer segmentation results and generate insights"""
        insights = []
        recommendations = []
        
        segments = segmentation_result.get("segments", [])
        segment_profiles = segmentation_result.get("segment_profiles", [])
        segmentation_type = segmentation_result.get("segmentation_type", "unknown")
        quality_metrics = segmentation_result.get("quality_metrics", {})
        
        total_contacts = quality_metrics.get("total_contacts", len(segments))
        num_segments = len(segment_profiles)
        
        # General segmentation insights
        if total_contacts > 0:
            insights.append(f"Segmented {total_contacts:,} contacts into {num_segments} segments")
            
            # Segment size distribution analysis
            segment_sizes = [profile.get("size", 0) for profile in segment_profiles]
            if segment_sizes:
                largest_segment = max(segment_sizes)
                smallest_segment = min(segment_sizes)
                avg_segment_size = sum(segment_sizes) / len(segment_sizes)
                
                # Check for segment balance
                size_ratio = largest_segment / max(smallest_segment, 1)
                if size_ratio > 5:
                    insights.append("Highly unbalanced segments - consider adjusting segmentation criteria")
                    recommendations.append("Review segmentation parameters to create more balanced groups")
                elif size_ratio < 2:
                    insights.append("Well-balanced segment distribution")
                
                # Segment size insights
                largest_pct = (largest_segment / total_contacts) * 100
                if largest_pct > 60:
                    insights.append(f"One segment dominates ({largest_pct:.1f}% of contacts)")
                    recommendations.append("Consider increasing number of segments for better granularity")
        
        # Segmentation-specific insights
        if segmentation_type == "behavioral":
            insights.append("Behavioral segmentation based on engagement patterns")
            
            # Look for engagement patterns in segment profiles
            high_engagement_segments = []
            low_engagement_segments = []
            
            for profile in segment_profiles:
                label = profile.get("label", "")
                characteristics = profile.get("characteristics", {})
                avg_engagement = characteristics.get("avg_engagement", 0)
                
                if avg_engagement > 8:
                    high_engagement_segments.append(label)
                elif avg_engagement < 2:
                    low_engagement_segments.append(label)
            
            if high_engagement_segments:
                insights.append(f"High engagement segments: {', '.join(high_engagement_segments)}")
                recommendations.append("Focus retention campaigns on high-engagement segments")
            
            if low_engagement_segments:
                insights.append(f"Low engagement segments: {', '.join(low_engagement_segments)}")
                recommendations.append("Develop re-engagement campaigns for low-activity segments")
        
        elif segmentation_type == "rfm":
            insights.append("RFM segmentation based on recency, frequency, and monetary value")
            
            # Look for champion and at-risk segments
            for profile in segment_profiles:
                label = profile.get("label", "")
                size = profile.get("size", 0)
                percentage = profile.get("percentage", 0)
                characteristics = profile.get("characteristics", {})
                
                if "champion" in label.lower():
                    insights.append(f"Champions segment: {size} contacts ({percentage:.1f}%)")
                    recommendations.append("Maintain champion relationships with exclusive offers and personal attention")
                
                elif "at risk" in label.lower() or "risk" in label.lower():
                    insights.append(f"At-risk segment: {size} contacts ({percentage:.1f}%)")
                    recommendations.append("Implement immediate win-back campaigns for at-risk customers")
                
                elif "loyal" in label.lower():
                    recommendations.append("Cross-sell and upsell opportunities in loyal customer segment")
        
        elif segmentation_type == "lifecycle":
            insights.append("Lifecycle segmentation based on customer journey stages")
            
            # Analyze lifecycle distribution
            opportunity_segment = None
            lead_segment = None
            customer_segment = None
            
            for profile in segment_profiles:
                label = profile.get("label", "").lower()
                size = profile.get("size", 0)
                percentage = profile.get("percentage", 0)
                
                if "opportunity" in label:
                    opportunity_segment = {"size": size, "percentage": percentage}
                elif "lead" in label:
                    lead_segment = {"size": size, "percentage": percentage}
                elif "customer" in label:
                    customer_segment = {"size": size, "percentage": percentage}
            
            if opportunity_segment and opportunity_segment["percentage"] > 20:
                insights.append(f"High opportunity volume: {opportunity_segment['percentage']:.1f}% of contacts")
                recommendations.append("Focus sales resources on converting high-volume opportunities")
            
            if lead_segment and lead_segment["percentage"] > 40:
                insights.append(f"Large lead pipeline: {lead_segment['percentage']:.1f}% of contacts")
                recommendations.append("Implement lead nurturing campaigns to advance pipeline")
            
            if customer_segment and customer_segment["percentage"] < 15:
                insights.append(f"Low customer conversion rate: {customer_segment['percentage']:.1f}%")
                recommendations.append("Analyze conversion bottlenecks and improve sales process")
        
        elif segmentation_type == "value_based":
            insights.append("Value-based segmentation by customer lifetime value and revenue potential")
            
            # Look for high-value and enterprise segments
            for profile in segment_profiles:
                label = profile.get("label", "")
                size = profile.get("size", 0)
                percentage = profile.get("percentage", 0)
                characteristics = profile.get("characteristics", {})
                
                if "enterprise" in label.lower():
                    insights.append(f"Enterprise segment: {size} contacts ({percentage:.1f}%)")
                    recommendations.append("Assign dedicated account management for enterprise clients")
                
                elif "high value" in label.lower():
                    avg_revenue = characteristics.get("avg_total_revenue", 0)
                    insights.append(f"High-value segment: ${avg_revenue:,.0f} average revenue")
                    recommendations.append("Develop premium service offerings for high-value customers")
                
                elif "prospect" in label.lower():
                    if percentage > 30:
                        insights.append(f"Large prospect pool: {percentage:.1f}% of contacts")
                        recommendations.append("Implement systematic prospect qualification and nurturing")
        
        # Quality metrics insights
        if quality_metrics:
            silhouette_score = quality_metrics.get("silhouette_score", 0)
            if silhouette_score > 0.7:
                insights.append("High segmentation quality - distinct, well-separated groups")
            elif silhouette_score < 0.3:
                insights.append("Low segmentation quality - consider different approach or parameters")
                recommendations.append("Experiment with different segmentation methods or adjust parameters")
        
        # General recommendations based on segment count
        if num_segments < 3:
            recommendations.append("Consider increasing segments for more targeted marketing approaches")
        elif num_segments > 8:
            recommendations.append("Consider consolidating segments to avoid over-complexity in campaigns")
        
        # Actionability recommendations
        recommendations.append("Create segment-specific content and messaging strategies")
        recommendations.append("Set up automated workflows for each segment's typical customer journey")
        recommendations.append("Monitor segment migration patterns to identify growth opportunities")
        
        return {
            "insights": insights,
            "recommendations": recommendations
        }

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
