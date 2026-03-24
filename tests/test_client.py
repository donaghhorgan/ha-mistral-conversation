"""Tests for the Mistral AI client."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryError

from custom_components.mistral_conversation.client import (
    MistralAIClient,
    _retry_with_exponential_backoff,
)


@pytest.mark.asyncio
async def test_retry_with_exponential_backoff_success():
    """Test successful retry with exponential backoff."""

    async def mock_func():
        return "success"

    result = await _retry_with_exponential_backoff(mock_func)
    assert result == "success"


@pytest.mark.asyncio
async def test_retry_with_exponential_backoff_retryable_error():
    """Test retry with exponential backoff for retryable errors."""
    call_count = 0

    async def mock_func():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConfigEntryError("503 Service Unavailable")
        return "success"

    result = await _retry_with_exponential_backoff(mock_func)
    assert result == "success"
    assert call_count == 3


@pytest.mark.asyncio
async def test_retry_with_exponential_backoff_non_retryable_error():
    """Test retry with exponential backoff for non-retryable errors."""

    async def mock_func():
        raise ConfigEntryError("400 Bad Request")

    with pytest.raises(ConfigEntryError):
        await _retry_with_exponential_backoff(mock_func)


@pytest.mark.asyncio
async def test_client_initialization():
    """Test client initialization."""
    hass = MagicMock(spec=HomeAssistant)
    hass.data = {}
    api_key = "test_api_key"
    model = "test_model"
    temperature = 0.5
    max_tokens = 500

    with patch(
        "custom_components.mistral_conversation.client.async_get_clientsession"
    ) as mock_session:
        mock_session.return_value = AsyncMock()
        client = MistralAIClient(hass, api_key, model, temperature, max_tokens)

        assert client.model == model


@pytest.mark.asyncio
async def test_test_connection_success():
    """Test successful connection test."""
    hass = MagicMock(spec=HomeAssistant)
    hass.data = {}
    api_key = "test_api_key"

    # Create a proper async context manager mock
    class MockResponse:
        def __init__(self, status):
            self.status = status

    class MockContextManager:
        def __init__(self, response):
            self.response = response

        async def __aenter__(self):
            return self.response

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    mock_response = MockResponse(200)

    with patch(
        "custom_components.mistral_conversation.client.async_get_clientsession"
    ) as mock_session:
        mock_client = MagicMock()
        mock_client.get.return_value = MockContextManager(mock_response)
        mock_session.return_value = mock_client

        client = MistralAIClient(hass, api_key)
        result = await client.test_connection()

        assert result is True


@pytest.mark.asyncio
async def test_test_connection_failure():
    """Test failed connection test."""
    hass = MagicMock(spec=HomeAssistant)
    api_key = "test_api_key"

    with patch(
        "custom_components.mistral_conversation.client.async_get_clientsession"
    ) as mock_session:
        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 401
        mock_client.get.return_value.__aenter__.return_value = mock_response
        mock_session.return_value = mock_client

        client = MistralAIClient(hass, api_key)
        result = await client.test_connection()

        assert result is False


@pytest.mark.asyncio
async def test_get_available_models_success():
    """Test successful retrieval of available models."""
    hass = MagicMock(spec=HomeAssistant)
    hass.data = {}
    api_key = "test_api_key"

    # Create a proper async context manager mock
    class MockResponse:
        def __init__(self, status, json_data):
            self.status = status
            self._json_data = json_data

        async def json(self):
            return self._json_data

    class MockContextManager:
        def __init__(self, response):
            self.response = response

        async def __aenter__(self):
            return self.response

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    mock_response = MockResponse(200, {"data": [{"id": "model1"}, {"id": "model2"}]})

    with patch(
        "custom_components.mistral_conversation.client.async_get_clientsession"
    ) as mock_session:
        mock_client = MagicMock()
        mock_client.get.return_value = MockContextManager(mock_response)
        mock_session.return_value = mock_client

        client = MistralAIClient(hass, api_key)
        models = await client.get_available_models()

        assert models == [{"id": "model1"}, {"id": "model2"}]


@pytest.mark.asyncio
async def test_get_available_models_failure():
    """Test failed retrieval of available models."""
    hass = MagicMock(spec=HomeAssistant)
    api_key = "test_api_key"

    with patch(
        "custom_components.mistral_conversation.client.async_get_clientsession"
    ) as mock_session:
        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 401
        mock_client.get.return_value.__aenter__.return_value = mock_response
        mock_session.return_value = mock_client

        client = MistralAIClient(hass, api_key)
        models = await client.get_available_models()

        assert models == []


@pytest.mark.asyncio
async def test_chat_completion_success():
    """Test successful chat completion."""
    hass = MagicMock(spec=HomeAssistant)
    hass.data = {}
    api_key = "test_api_key"

    # Create a proper async context manager mock
    class MockResponse:
        def __init__(self, status, json_data):
            self.status = status
            self._json_data = json_data

        async def json(self):
            return self._json_data

    class MockContextManager:
        def __init__(self, response):
            self.response = response

        async def __aenter__(self):
            return self.response

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    mock_response = MockResponse(
        200, {"choices": [{"message": {"content": "test response"}}]}
    )

    with patch(
        "custom_components.mistral_conversation.client.async_get_clientsession"
    ) as mock_session:
        mock_client = MagicMock()
        mock_client.post.return_value = MockContextManager(mock_response)
        mock_session.return_value = mock_client

        client = MistralAIClient(hass, api_key)
        messages = [{"role": "user", "content": "test message"}]
        response = await client.chat_completion(messages)

        assert response == {"choices": [{"message": {"content": "test response"}}]}


@pytest.mark.asyncio
async def test_chat_completion_error():
    """Test chat completion with error."""
    hass = MagicMock(spec=HomeAssistant)
    api_key = "test_api_key"

    with patch(
        "custom_components.mistral_conversation.client.async_get_clientsession"
    ) as mock_session:
        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 401
        mock_response.text.return_value = "Unauthorized"
        mock_client.post.return_value.__aenter__.return_value = mock_response
        mock_session.return_value = mock_client

        client = MistralAIClient(hass, api_key)
        messages = [{"role": "user", "content": "test message"}]

        with pytest.raises(ConfigEntryError):
            await client.chat_completion(messages)


@pytest.mark.asyncio
async def test_generate_response_success():
    """Test successful response generation."""
    hass = MagicMock(spec=HomeAssistant)
    hass.data = {}
    api_key = "test_api_key"

    # Create a proper async context manager mock
    class MockResponse:
        def __init__(self, status, json_data):
            self.status = status
            self._json_data = json_data

        async def json(self):
            return self._json_data

    class MockContextManager:
        def __init__(self, response):
            self.response = response

        async def __aenter__(self):
            return self.response

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    mock_response = MockResponse(
        200, {"choices": [{"message": {"content": "test response"}}]}
    )

    with patch(
        "custom_components.mistral_conversation.client.async_get_clientsession"
    ) as mock_session:
        mock_client = MagicMock()
        mock_client.post.return_value = MockContextManager(mock_response)
        mock_session.return_value = mock_client

        client = MistralAIClient(hass, api_key)
        response = await client.generate_response("test prompt")

        assert response == "test response"


@pytest.mark.asyncio
async def test_generate_response_error():
    """Test response generation with error."""
    hass = MagicMock(spec=HomeAssistant)
    hass.data = {}
    api_key = "test_api_key"

    # Create a proper async context manager mock
    class MockResponse:
        def __init__(self, status, text_data):
            self.status = status
            self._text_data = text_data

        async def text(self):
            return self._text_data

    class MockContextManager:
        def __init__(self, response):
            self.response = response

        async def __aenter__(self):
            return self.response

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    mock_response = MockResponse(401, "Unauthorized")

    with patch(
        "custom_components.mistral_conversation.client.async_get_clientsession"
    ) as mock_session:
        mock_client = MagicMock()
        mock_client.post.return_value = MockContextManager(mock_response)
        mock_session.return_value = mock_client

        client = MistralAIClient(hass, api_key)
        response = await client.generate_response("test prompt")

        assert "authentication issue" in response.lower()


@pytest.mark.asyncio
async def test_generate_response_invalid_format():
    """Test response generation with invalid response format."""
    hass = MagicMock(spec=HomeAssistant)
    hass.data = {}
    api_key = "test_api_key"

    # Create a proper async context manager mock
    class MockResponse:
        def __init__(self, status, json_data):
            self.status = status
            self._json_data = json_data

        async def json(self):
            return self._json_data

    class MockContextManager:
        def __init__(self, response):
            self.response = response

        async def __aenter__(self):
            return self.response

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    mock_response = MockResponse(200, {"invalid": "response"})

    with patch(
        "custom_components.mistral_conversation.client.async_get_clientsession"
    ) as mock_session:
        mock_client = MagicMock()
        mock_client.post.return_value = MockContextManager(mock_response)
        mock_session.return_value = mock_client

        client = MistralAIClient(hass, api_key)
        response = await client.generate_response("test prompt")

        assert "invalid response" in response.lower()
