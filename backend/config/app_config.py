from typing import Dict, Any
from services.job_service import job_service
from services.translation_service import translation_service
from exceptions import *

# Application configuration
class AppConfig:
    """Application configuration and dependency container"""

    def __init__(self):
        self.job_service = job_service
        self.translation_service = translation_service
        self.pipeline_available = True

    def initialize_services(self) -> None:
        """Initialize all services"""
        # Import and initialize translators
        try:
            from shared_dependencies import get_translator
            # This will populate the translation service
            self.translation_service.add_translator("en-es", get_translator("en", "es"))
            self.translation_service.add_translator("es-en", get_translator("es", "en"))
            # Add more language pairs as needed
        except ImportError:
            pass

        # Check pipeline availability
        try:
            from modules.pipeline import VideoTranslationPipeline
            self.pipeline_available = True
        except ImportError:
            self.pipeline_available = False

        self.translation_service.pipeline_available = self.pipeline_available

# Global config instance
app_config = AppConfig()