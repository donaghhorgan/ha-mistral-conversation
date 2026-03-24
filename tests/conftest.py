"""Pytest configuration for Mistral AI Conversation tests."""

import sys
from pathlib import Path

# Add the project root to Python path so custom_components can be imported
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Optional: Add custom_components to path directly
CUSTOM_COMPONENTS_PATH = PROJECT_ROOT / "custom_components"
sys.path.insert(0, str(CUSTOM_COMPONENTS_PATH))
