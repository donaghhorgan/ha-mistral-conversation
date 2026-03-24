"""Constants for the Mistral AI Conversation integration."""

DOMAIN = "mistral_conversation"

# API Configuration
API_BASE_URL = "https://api.mistral.ai/v1"

# Configuration keys
CONF_API_KEY = "api_key"
CONF_MODEL = "model"
CONF_TEMPERATURE = "temperature"
CONF_MAX_TOKENS = "max_tokens"
CONF_PROMPT = "prompt"

# Default values
DEFAULT_MODEL = "mistral-small-latest"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1000
DEFAULT_PROMPT = """You are a helpful assistant integrated with Home Assistant.
You help users with their smart home queries and general questions.
The current Home Assistant instance is located at: {{ ha_name }}
Please be concise and helpful in your responses."""
