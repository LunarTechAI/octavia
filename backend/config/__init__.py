# Config package
import os

# Configuration constants
HELSINKI_MODELS = {
    "en-es": "Helsinki-NLP/opus-mt-en-es",
    "es-en": "Helsinki-NLP/opus-mt-es-en",
    "en-fr": "Helsinki-NLP/opus-mt-en-fr",
    "fr-en": "Helsinki-NLP/opus-mt-fr-en",
}

DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"
ENABLE_TEST_MODE = os.getenv("ENABLE_TEST_MODE", "false").lower() == "true"

POLAR_SERVER = "sandbox" if ENABLE_TEST_MODE else "live"