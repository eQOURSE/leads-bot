"""Phase 9 — Agent registry (dependency injection container).

Builds all source clients, agents, and sinks once at startup and re-uses them
across all three segment graph runs.
"""

from __future__ import annotations

from config.logging_config import setup_logging
from config.settings import Settings


class AgentRegistry:
    """Bundles all agents and clients with shared settings + lead_store.

    Builds once at startup; passed into all node functions.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.log = setup_logging("orchestrator.registry")

        # ---- Source clients (Phase 1) ----
        from sources.vibe_prospecting import VibeProspectingClient
        from sources.hunter_client import HunterClient
        from sources.scrapegraph_client import ScrapeGraphClient
        from sources.apify_client import ApifyMultiKeyClient
        from sources.serpapi_client import SerpAPIClient
        from sources.newsdata_client import NewsDataClient
        from sources.companies_api_client import CompaniesAPIClient
        from sources.rss_feeds import RSSFundingMonitor
        from sources.abstract_api_client import AbstractAPIClient
        from sources.smtp_verifier import SMTPVerifier

        self.vibe = VibeProspectingClient(settings)
        self.hunter = HunterClient(settings)
        self.scrapegraph = ScrapeGraphClient(settings)
        self.apify = ApifyMultiKeyClient(settings)
        self.serpapi = SerpAPIClient(settings)
        self.newsdata = NewsDataClient(settings)
        self.companies = CompaniesAPIClient(settings)
        self.rss = RSSFundingMonitor(settings)
        self.abstract = AbstractAPIClient(settings)
        self.smtp = SMTPVerifier(settings)

        # ---- Lead store (Phase 2) ----
        from sinks.sqlite_store import LeadStore
        self.lead_store = LeadStore(settings)

        # ---- Gemini agents (shared wrappers) ----
        from agents._gemini_wrapper import GeminiAgent
        self.gemini_flash = GeminiAgent(settings.GEMINI_MODEL_QUALIFIER, settings)
        self.gemini_personalizer = GeminiAgent(settings.GEMINI_MODEL_PERSONALIZER, settings)
        self.gemini_pro = GeminiAgent(settings.GEMINI_MODEL_WRITER, settings)
        self.gemini_validator = GeminiAgent(settings.GEMINI_MODEL_VALIDATOR, settings)

        # ---- Agents (Phases 2–8) ----
        from agents.icp_strategist import IcpStrategist
        from agents.company_hunter import CompanyHunter
        from agents.qualifier import Qualifier
        from agents.decision_maker_finder import DecisionMakerFinder
        from agents.contact_enricher import ContactEnricher
        from agents.personalizer import Personalizer
        from agents.message_writer import MessageWriter
        from agents.validator import Validator

        self.icp_strategist = IcpStrategist(settings)

        self.company_hunter = CompanyHunter(
            settings,
            self.icp_strategist,
            self.rss,
            self.serpapi,
            self.newsdata,
            self.companies,
            self.lead_store,
        )

        self.qualifier = Qualifier(
            settings,
            self.icp_strategist,
            self.gemini_flash,
            self.companies,
            self.lead_store,
        )

        self.dm_finder = DecisionMakerFinder(
            settings,
            self.icp_strategist,
            self.scrapegraph,
            self.apify,
            self.vibe,
            self.lead_store,
        )

        self.contact_enricher = ContactEnricher(
            settings,
            self.hunter,
            self.abstract,
            self.smtp,
            self.vibe,
            self.lead_store,
        )

        self.personalizer = Personalizer(
            settings,
            self.icp_strategist,
            self.gemini_personalizer,
            self.scrapegraph,
            self.newsdata,
            self.lead_store,
        )

        self.message_writer = MessageWriter(
            settings,
            self.icp_strategist,
            self.gemini_pro,
            self.lead_store,
        )

        self.validator = Validator(
            settings,
            self.lead_store,
            self.smtp,
            self.gemini_validator,
        )

        # ---- Sinks (Phase 8) ----
        from sinks.sqlite_writer import SQLiteWriter
        from sinks.google_sheets_sink import GoogleSheetsSink
        from sinks.telegram_sink import TelegramSink
        from sinks.sink_orchestrator import SinkOrchestrator

        self.sqlite_writer = SQLiteWriter(settings, self.lead_store)
        self.sheets_sink = GoogleSheetsSink(settings, self.lead_store)
        self.telegram_sink = TelegramSink(settings)
        self.sink_orchestrator = SinkOrchestrator(
            settings,
            self.sqlite_writer,
            self.sheets_sink,
            self.telegram_sink,
            self.lead_store,
        )

        self.log.info("AgentRegistry: all clients, agents, and sinks initialised")
