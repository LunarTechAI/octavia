from typing import Dict, Any, Optional, Callable
import logging

logger = logging.getLogger(__name__)

class TranslationService:
    """Service for managing translation models and functions"""

    def __init__(self):
        self._translators: Dict[str, Callable] = {}
        self._pipeline_available = True

    def add_translator(self, key: str, translator_func: Callable) -> None:
        """Add a translator function"""
        self._translators[key] = translator_func
        logger.debug(f"Added translator for {key}")

    def get_translator(self, source_lang: str, target_lang: str) -> Optional[Callable]:
        """Get translator for language pair"""
        key = f"{source_lang}-{target_lang}"
        return self._translators.get(key)

    def has_translator(self, source_lang: str, target_lang: str) -> bool:
        """Check if translator exists for language pair"""
        return self.get_translator(source_lang, target_lang) is not None

    def get_all_translators(self) -> Dict[str, Callable]:
        """Get all available translators"""
        return self._translators.copy()

    def clear_cache(self) -> None:
        """Clear the translator cache"""
        self._translators.clear()
        logger.info("Cleared translator cache")

    @property
    def pipeline_available(self) -> bool:
        """Check if pipeline is available"""
        return self._pipeline_available

    @pipeline_available.setter
    def pipeline_available(self, available: bool) -> None:
        """Set pipeline availability"""
        self._pipeline_available = available
        logger.info(f"Pipeline availability set to {available}")

# Global instance
translation_service = TranslationService()