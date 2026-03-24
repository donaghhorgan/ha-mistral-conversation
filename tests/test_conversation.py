"""Tests for the Mistral AI conversation entity."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers import intent

from custom_components.mistral_conversation.const import (
    CONF_API_KEY,
    CONF_MAX_TOKENS,
    CONF_MODEL,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_PROMPT,
    DEFAULT_TEMPERATURE,
)
from custom_components.mistral_conversation.conversation import (
    MistralConversationEntity,
    async_setup_entry,
)


@pytest.mark.asyncio
async def test_async_setup_entry():
    """Test setup of conversation entry."""
    hass = MagicMock(spec=HomeAssistant)
    config_entry = MagicMock(spec=ConfigEntry)
    config_entry.data = {
        CONF_API_KEY: "test_api_key",
        CONF_MODEL: DEFAULT_MODEL,
    }
    config_entry.entry_id = "test_entry_id"

    mock_add_entities = MagicMock()

    await async_setup_entry(hass, config_entry, mock_add_entities)

    mock_add_entities.assert_called_once()
    added_entities = mock_add_entities.call_args[0][0]
    assert len(added_entities) == 1
    assert isinstance(added_entities[0], MistralConversationEntity)


@pytest.mark.asyncio
async def test_conversation_entity_initialization():
    """Test conversation entity initialization."""
    hass = MagicMock(spec=HomeAssistant)
    config_entry = MagicMock(spec=ConfigEntry)
    config_entry.data = {
        CONF_API_KEY: "test_api_key",
        CONF_MODEL: "test_model",
    }
    config_entry.entry_id = "test_entry_id"

    entity = MistralConversationEntity(hass, config_entry)

    assert entity._attr_name == "Mistral AI (test_model)"
    assert entity._attr_unique_id == "test_entry_id"
    assert entity._client is None


@pytest.mark.asyncio
async def test_supported_languages():
    """Test supported languages."""
    hass = MagicMock(spec=HomeAssistant)
    config_entry = MagicMock(spec=ConfigEntry)
    config_entry.data = {
        CONF_API_KEY: "test_api_key",
        CONF_MODEL: DEFAULT_MODEL,
    }
    config_entry.entry_id = "test_entry_id"

    entity = MistralConversationEntity(hass, config_entry)

    languages = entity.supported_languages
    assert languages == [conversation.MATCH_ALL]


@pytest.mark.asyncio
async def test_async_added_to_hass():
    """Test entity added to hass."""
    hass = MagicMock(spec=HomeAssistant)
    config_entry = MagicMock(spec=ConfigEntry)
    config_entry.data = {
        CONF_API_KEY: "test_api_key",
        CONF_MODEL: DEFAULT_MODEL,
        CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
        CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
    }
    config_entry.entry_id = "test_entry_id"

    entity = MistralConversationEntity(hass, config_entry)

    with patch(
        "custom_components.mistral_conversation.conversation.MistralAIClient"
    ) as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        await entity.async_added_to_hass()

        assert entity._client is not None
        mock_client_class.assert_called_once_with(
            hass,
            "test_api_key",
            DEFAULT_MODEL,
            DEFAULT_TEMPERATURE,
            DEFAULT_MAX_TOKENS,
        )


@pytest.mark.asyncio
async def test_async_process_success():
    """Test successful conversation processing."""
    hass = MagicMock(spec=HomeAssistant)
    hass.config = MagicMock()
    hass.config.location_name = "Test Home"
    hass.data = {}

    config_entry = MagicMock(spec=ConfigEntry)
    config_entry.data = {
        CONF_API_KEY: "test_api_key",
        CONF_MODEL: DEFAULT_MODEL,
        CONF_PROMPT: DEFAULT_PROMPT,
    }
    config_entry.entry_id = "test_entry_id"

    entity = MistralConversationEntity(hass, config_entry)

    # Mock the client
    mock_client = AsyncMock()
    mock_client.generate_response.return_value = "test response"
    entity._client = mock_client

    # Create conversation input
    user_input = conversation.ConversationInput(
        text="test question",
        conversation_id="test_conversation_id",
        language="en",
        context=None,
        device_id=None,
        satellite_id=None,
        agent_id=None,
    )

    result = await entity.async_process(user_input)

    assert result.conversation_id == "test_conversation_id"
    assert result.response.speech["plain"]["speech"] == "test response"

    # Verify client was called correctly
    # The prompt should be rendered with the actual location name
    expected_context = DEFAULT_PROMPT.replace("{{ ha_name }}", "Test Home")
    mock_client.generate_response.assert_awaited_once_with(
        "test question", context=expected_context
    )


@pytest.mark.asyncio
async def test_async_process_client_not_initialized():
    """Test conversation processing when client is not initialized."""
    hass = MagicMock(spec=HomeAssistant)
    config_entry = MagicMock(spec=ConfigEntry)
    config_entry.data = {
        CONF_API_KEY: "test_api_key",
        CONF_MODEL: DEFAULT_MODEL,
    }
    config_entry.entry_id = "test_entry_id"

    entity = MistralConversationEntity(hass, config_entry)
    entity._client = None

    user_input = conversation.ConversationInput(
        text="test question",
        conversation_id="test_conversation_id",
        language="en",
        context=None,
        device_id=None,
        satellite_id=None,
        agent_id=None,
    )

    with pytest.raises(ConfigEntryNotReady):
        await entity.async_process(user_input)


@pytest.mark.asyncio
async def test_async_process_with_llm_api():
    """Test conversation processing with LLM API."""
    hass = MagicMock(spec=HomeAssistant)
    hass.config = MagicMock()
    hass.config.location_name = "Test Home"
    hass.data = {"llm": {"test_llm": MagicMock()}}

    config_entry = MagicMock(spec=ConfigEntry)
    config_entry.data = {
        CONF_API_KEY: "test_api_key",
        CONF_MODEL: DEFAULT_MODEL,
        CONF_PROMPT: DEFAULT_PROMPT,
        CONF_LLM_HASS_API: "test_llm",
    }
    config_entry.entry_id = "test_entry_id"

    entity = MistralConversationEntity(hass, config_entry)

    # Mock the client
    mock_client = AsyncMock()
    mock_client.generate_response.return_value = "test response"
    entity._client = mock_client

    # Create conversation input
    user_input = conversation.ConversationInput(
        text="test question",
        conversation_id="test_conversation_id",
        language="en",
        context=None,
        device_id=None,
        satellite_id=None,
        agent_id=None,
    )

    result = await entity.async_process(user_input)

    assert result.conversation_id == "test_conversation_id"
    assert result.response.speech["plain"]["speech"] == "test response"


@pytest.mark.asyncio
async def test_async_process_with_template_error():
    """Test conversation processing with template error."""
    hass = MagicMock(spec=HomeAssistant)
    hass.config = MagicMock()
    hass.config.location_name = "Test Home"
    hass.data = {}

    config_entry = MagicMock(spec=ConfigEntry)
    config_entry.data = {
        CONF_API_KEY: "test_api_key",
        CONF_MODEL: DEFAULT_MODEL,
        CONF_PROMPT: "{{ invalid_template_syntax ",
    }
    config_entry.entry_id = "test_entry_id"

    entity = MistralConversationEntity(hass, config_entry)

    # Mock the client
    mock_client = AsyncMock()
    mock_client.generate_response.return_value = "test response"
    entity._client = mock_client

    # Create conversation input
    user_input = conversation.ConversationInput(
        text="test question",
        conversation_id="test_conversation_id",
        language="en",
        context=None,
        device_id=None,
        satellite_id=None,
        agent_id=None,
    )

    result = await entity.async_process(user_input)

    # Should fall back to default prompt when template error occurs
    assert result.conversation_id == "test_conversation_id"
    assert result.response.speech["plain"]["speech"] == "test response"

    # Verify client was called with default prompt
    mock_client.generate_response.assert_awaited_once_with(
        "test question", context=DEFAULT_PROMPT
    )


@pytest.mark.asyncio
async def test_async_process_with_client_error():
    """Test conversation processing with client error."""
    hass = MagicMock(spec=HomeAssistant)
    hass.config = MagicMock()
    hass.config.location_name = "Test Home"
    hass.data = {}

    config_entry = MagicMock(spec=ConfigEntry)
    config_entry.data = {
        CONF_API_KEY: "test_api_key",
        CONF_MODEL: DEFAULT_MODEL,
        CONF_PROMPT: DEFAULT_PROMPT,
    }
    config_entry.entry_id = "test_entry_id"

    entity = MistralConversationEntity(hass, config_entry)

    # Mock the client to raise an error
    mock_client = AsyncMock()
    mock_client.generate_response.side_effect = Exception("API error")
    entity._client = mock_client

    # Create conversation input
    user_input = conversation.ConversationInput(
        text="test question",
        conversation_id="test_conversation_id",
        language="en",
        context=None,
        device_id=None,
        satellite_id=None,
        agent_id=None,
    )

    result = await entity.async_process(user_input)

    # Should return error response
    assert result.conversation_id == "test_conversation_id"
    assert result.response.intent is None
    assert result.response.response_type == intent.IntentResponseType.ERROR


@pytest.mark.asyncio
async def test_async_process_without_conversation_id():
    """Test conversation processing without conversation ID."""
    hass = MagicMock(spec=HomeAssistant)
    hass.config = MagicMock()
    hass.config.location_name = "Test Home"
    hass.data = {}

    config_entry = MagicMock(spec=ConfigEntry)
    config_entry.data = {
        CONF_API_KEY: "test_api_key",
        CONF_MODEL: DEFAULT_MODEL,
        CONF_PROMPT: DEFAULT_PROMPT,
    }
    config_entry.entry_id = "test_entry_id"

    entity = MistralConversationEntity(hass, config_entry)

    # Mock the client
    mock_client = AsyncMock()
    mock_client.generate_response.return_value = "test response"
    entity._client = mock_client

    # Create conversation input without conversation_id
    user_input = conversation.ConversationInput(
        text="test question",
        conversation_id=None,
        language="en",
        context=None,
        device_id=None,
        satellite_id=None,
        agent_id=None,
    )

    result = await entity.async_process(user_input)

    # Should generate a new conversation ID
    assert result.conversation_id is not None
    assert len(result.conversation_id) > 0
    assert result.response.speech["plain"]["speech"] == "test response"


@pytest.mark.asyncio
async def test_attribution():
    """Test attribution property."""
    hass = MagicMock(spec=HomeAssistant)
    config_entry = MagicMock(spec=ConfigEntry)
    config_entry.data = {
        CONF_API_KEY: "test_api_key",
        CONF_MODEL: DEFAULT_MODEL,
    }
    config_entry.entry_id = "test_entry_id"

    entity = MistralConversationEntity(hass, config_entry)

    assert entity.attribution == "Powered by Mistral AI"
