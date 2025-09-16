#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP HubSpot Connector - A comprehensive HubSpot CRM integration with advanced analytics

This package provides a modular, enterprise-ready connector for HubSpot CRM
with advanced analytics capabilities including ML-powered lead scoring,
customer segmentation, revenue forecasting, and executive reporting.

Main Components:
- MCPHubspotConnector: Main connector class
- AnalyticsEngine: Core analytics processing engine
- InsightGenerator: Business insights and recommendations generator
- RateLimiter: API rate limiting and retry management
- HubSpotSDKConfig: Configuration dataclass for HubSpot SDK
"""

# Import from the main mcp_hubspot_connector module
from .mcp_hubspot_connector import HubSpotSDKConfig, MCPHubspotConnector, handle_hubspot_errors

__version__ = "2.0.0"
__author__ = "bibow"

__all__ = [
    # Main Classes
    "MCPHubspotConnector",
    "AnalyticsEngine",
    "InsightGenerator",
    "RateLimiter",
    # Configuration
    "HubSpotSDKConfig",
    # Utilities
    "handle_hubspot_errors",
    # Metadata
    "__version__",
    "__author__",
]
