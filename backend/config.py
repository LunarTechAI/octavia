import os
from dotenv import load_dotenv
load_dotenv()

# DEMO_MODE: enable demo login without Supabase
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"

# Polar.sh configuration
POLAR_WEBHOOK_SECRET = os.getenv("POLAR_WEBHOOK_SECRET", "")
POLAR_ACCESS_TOKEN = os.getenv("POLAR_ACCESS_TOKEN")
POLAR_SERVER = os.getenv("POLAR_SERVER", "sandbox")
ENABLE_TEST_MODE = True

# Credit packages configuration with real Polar.sh product IDs
CREDIT_PACKAGES = {
    "starter_credits": {
        "name": "Starter Credits",
        "credits": 100,
        "price": 999,
        "polar_product_id": "68d54da0-c3ec-4215-9636-21457e57b3e6",
        "checkout_link": "https://sandbox-api.polar.sh/v1/checkout-links/polar_cl_ENF1TwWHLmhB809OfLQozk0UCGMLmYinMbfT14K8K2R/redirect",
        "description": "100 translation credits",
        "features": ["100 credits", "Standard processing", "Email support"],
        "popular": False
    },
    "pro_credits": {
        "name": "Pro Credits",
        "credits": 250,
        "price": 1999,
        "polar_product_id": "743297c6-eadb-4b96-a8d6-b4c815f0f1b5",
        "checkout_link": "https://sandbox-api.polar.sh/v1/checkout-links/polar_cl_SXDRYMs6nvN9dm8b5wK8Z3WcsowTEU7jYPXFe4XXHgm/redirect",
        "description": "250 translation credits",
        "features": ["250 credits", "Priority processing", "Priority support"],
        "popular": True
    },
    "premium_credits": {
        "name": "Premium Credits",
        "credits": 500,
        "price": 3499,
        "polar_product_id": "2dceabdb-d0f8-4ddd-9b68-af44f0c4ad96",
        "checkout_link": "https://sandbox-api.polar.sh/v1/checkout-links/polar_cl_QNmrgCNlflNXndg61t31JhwmQVIe5cthFDyAy2yb2ED/redirect",
        "description": "500 translation credits",
        "features": ["500 credits", "Express processing", "24/7 support", "Batch upload"],
        "popular": False
    }
}

# Helsinki NLP model mapping
HELSINKI_MODELS = {
    # English to other languages
    "en-es": "Helsinki-NLP/opus-mt-en-es",
    "en-fr": "Helsinki-NLP/opus-mt-en-fr",
    "en-de": "Helsinki-NLP/opus-mt-en-de",
    "en-it": "Helsinki-NLP/opus-mt-en-it",
    "en-ru": "Helsinki-NLP/opus-mt-en-ru",
    "en-ja": "Helsinki-NLP/opus-mt-en-jap",
    "en-ko": "Helsinki-NLP/opus-mt-en-ko",
    "en-zh": "Helsinki-NLP/opus-mt-en-zh",
    "en-ar": "Helsinki-NLP/opus-mt-en-ar",
    "en-hi": "Helsinki-NLP/opus-mt-en-hi",
    "en-pt": "Helsinki-NLP/opus-mt-en-pt",
    # Reverse translations
    "es-en": "Helsinki-NLP/opus-mt-es-en",
    "fr-en": "Helsinki-NLP/opus-mt-fr-en",
    "de-en": "Helsinki-NLP/opus-mt-de-en",
    "it-en": "Helsinki-NLP/opus-mt-it-en",
    "ru-en": "Helsinki-NLP/opus-mt-ru-en",
    # Between other languages
    "es-fr": "Helsinki-NLP/opus-mt-es-fr",
    "fr-es": "Helsinki-NLP/opus-mt-fr-es",
    "de-fr": "Helsinki-NLP/opus-mt-de-fr",
    "fr-de": "Helsinki-NLP/opus-mt-fr-de",
    "es-it": "Helsinki-NLP/opus-mt-es-it",
    "it-es": "Helsinki-NLP/opus-mt-it-es",
}
