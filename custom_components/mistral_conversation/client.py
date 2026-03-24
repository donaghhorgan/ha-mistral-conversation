"""Mistral AI client for Home Assistant conversation agent."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, TypedDict, cast

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    class MistralMessage(TypedDict, total=False):
        role: str
        content: str

    class MistralChoice(TypedDict, total=False):
        message: MistralMessage

    class MistralResponse(TypedDict, total=False):
        choices: list[MistralChoice]


import aiohttp
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryError
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import API_BASE_URL

_LOGGER = logging.getLogger(__name__)

# HTTP status codes for better error handling
HTTP_STATUS_TOO_MANY_REQUESTS = 429
HTTP_STATUS_SERVICE_UNAVAILABLE = 503
HTTP_STATUS_INTERNAL_SERVER_ERROR = 500
HTTP_STATUS_BAD_GATEWAY = 502
HTTP_STATUS_GATEWAY_TIMEOUT = 504

# Retry configuration
MAX_RETRY_ATTEMPTS = 3
INITIAL_RETRY_DELAY = 1.0  # seconds
MAX_RETRY_DELAY = 10.0  # seconds


async def _retry_with_exponential_backoff(
    func: Callable[..., Any],
    *args,
    **kwargs,
) -> Any:
    """Retry a function with exponential backoff for retryable errors."""
    last_error = None

    for attempt in range(MAX_RETRY_ATTEMPTS):
        try:
            return await func(*args, **kwargs)
        except ConfigEntryError as err:
            last_error = err
            error_message = str(err)

            # Check if this is a retryable error
            is_retryable = any(
                status_code in error_message
                for status_code in [
                    str(HTTP_STATUS_TOO_MANY_REQUESTS),
                    str(HTTP_STATUS_SERVICE_UNAVAILABLE),
                    str(HTTP_STATUS_INTERNAL_SERVER_ERROR),
                    str(HTTP_STATUS_BAD_GATEWAY),
                    str(HTTP_STATUS_GATEWAY_TIMEOUT),
                ]
            )

            if not is_retryable or attempt == MAX_RETRY_ATTEMPTS - 1:
                break

            # Calculate exponential backoff delay
            delay = min(INITIAL_RETRY_DELAY * (2**attempt), MAX_RETRY_DELAY)
            _LOGGER.warning(
                "Retryable error occurred (attempt %d/%d): %s. Retrying in %.1f seconds",
                attempt + 1,
                MAX_RETRY_ATTEMPTS,
                error_message,
                delay,
            )
            await asyncio.sleep(delay)
        except Exception as err:
            last_error = err
            break

    if last_error:
        raise last_error

    return None


class MistralAIClient:
    """Client for interacting with Mistral AI API."""

    def __init__(
        self,
        hass: HomeAssistant,
        api_key: str,
        model: str = "mistral-small-latest",
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> None:
        """Initialize the Mistral AI client."""
        self._hass = hass
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        # Use Home Assistant's shared ClientSession which already implements
        # connection pooling via aiohttp's TCPConnector
        self._session = async_get_clientsession(hass)

    @property
    def model(self) -> str:
        """Return the model name."""
        return self._model

    async def test_connection(self) -> bool:
        """Test the connection to Mistral AI API."""
        try:
            async with self._session.get(
                f"{API_BASE_URL}/models",
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                if response.status == 200:
                    return True
                _LOGGER.error(
                    "Failed to connect to Mistral AI API: %s", response.status
                )
                return False
        except Exception as err:
            _LOGGER.error("Error testing Mistral AI connection: %s", err)
            return False

    async def get_available_models(self) -> list[dict[str, Any]]:
        """Get list of available models from Mistral AI API."""
        try:
            async with self._session.get(
                f"{API_BASE_URL}/models",
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("data", [])
                _LOGGER.error("Failed to get models: %s", response.status)
                return []
        except Exception as err:
            _LOGGER.error("Error getting available models: %s", err)
            return []

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        stream: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> MistralResponse | AsyncIterator[MistralResponse]:
        """Send a chat completion request to Mistral AI API."""
        # Validate messages
        if not messages:
            raise ValueError("Messages list cannot be empty")

        if not isinstance(messages, list):
            raise ValueError("Messages must be a list")

        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValueError(f"Message {i} must be a dictionary")
            if "role" not in message or not isinstance(message["role"], str):
                raise ValueError(f"Message {i} must have a string 'role' field")
            if "content" not in message or not isinstance(message["content"], str):
                raise ValueError(f"Message {i} must have a string 'content' field")

        # Validate temperature
        if temperature is not None:
            if not isinstance(temperature, (int, float)):
                raise ValueError("Temperature must be a number")
            if not (0.0 <= temperature <= 2.0):
                raise ValueError("Temperature must be between 0.0 and 2.0")

        # Validate max_tokens
        if max_tokens is not None:
            if not isinstance(max_tokens, int):
                raise ValueError("Max tokens must be an integer")
            if not (max_tokens >= 1):
                raise ValueError("Max tokens must be at least 1")

        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature or self._temperature,
            "max_tokens": max_tokens or self._max_tokens,
            "stream": stream,
        }

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        # Define the actual completion function for retry logic
        async def _make_completion_request() -> (
            MistralResponse | AsyncIterator[MistralResponse]
        ):
            try:
                if stream:
                    return self._stream_completion(payload, headers)
                else:
                    return await self._single_completion(payload, headers)
            except Exception as err:
                _LOGGER.error("Error in chat completion: %s", err)
                raise ConfigEntryError(
                    f"Failed to get response from Mistral AI: {err}"
                ) from err

        # Use retry logic for non-streaming requests
        if stream:
            return await _make_completion_request()
        else:
            return await _retry_with_exponential_backoff(_make_completion_request)

    async def _single_completion(
        self, payload: dict[str, Any], headers: dict[str, str]
    ) -> MistralResponse:
        """Handle non-streaming completion."""
        async with self._session.post(
            f"{API_BASE_URL}/chat/completions",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as response:
            if response.status == 200:
                return await response.json()

            error_text = await response.text()

            # Handle specific HTTP status codes with appropriate error messages
            if response.status == 401:
                error_msg = "Unauthorized: Invalid API key or authentication failed"
            elif response.status == 403:
                error_msg = "Forbidden: API key does not have sufficient permissions"
            elif response.status == HTTP_STATUS_TOO_MANY_REQUESTS:
                error_msg = (
                    "Rate limit exceeded: Too many requests in a short time period"
                )
            elif response.status == HTTP_STATUS_SERVICE_UNAVAILABLE:
                error_msg = (
                    "Service unavailable: Mistral AI API is temporarily unavailable"
                )
            elif response.status >= 500:
                error_msg = f"Server error: Mistral AI API encountered an internal error ({response.status})"
            else:
                error_msg = f"API request failed with status {response.status}"

            _LOGGER.error("Mistral AI API error %s: %s", response.status, error_text)
            raise ConfigEntryError(f"{error_msg}: {error_text}")

    async def _stream_completion(
        self, payload: dict[str, Any], headers: dict[str, str]
    ) -> AsyncIterator[MistralResponse]:
        """Handle streaming completion."""
        async with self._session.post(
            f"{API_BASE_URL}/chat/completions",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                _LOGGER.error(
                    "Mistral AI API error: %s - %s", response.status, error_text
                )
                raise ConfigEntryError(
                    f"Mistral AI API error: {response.status} - {error_text}"
                )

            async for line_bytes in response.content:
                line = line_bytes.decode("utf-8").strip()
                if line.startswith("data: "):
                    data = line[6:]  # Remove "data: " prefix
                    if data == "[DONE]":
                        break
                    try:
                        yield json.loads(data)
                    except (ValueError, json.JSONDecodeError):
                        continue

    async def generate_response(
        self,
        prompt: str,
        conversation_id: str | None = None,
        context: str | None = None,
    ) -> str:
        """Generate a response for the given prompt."""
        # Validate prompt
        if not prompt:
            raise ValueError("Prompt cannot be empty")

        if not isinstance(prompt, str):
            raise ValueError("Prompt must be a string")

        # Validate context if provided
        if context is not None and not isinstance(context, str):
            raise ValueError("Context must be a string")

        # Validate conversation_id if provided
        if conversation_id is not None and not isinstance(conversation_id, str):
            raise ValueError("Conversation ID must be a string")

        messages: list[MistralMessage] = []

        # Add system context if provided
        if context:
            messages.append({"role": "system", "content": context})

        # Add user message
        messages.append({"role": "user", "content": prompt})

        try:
            # Cast messages to the expected type for chat_completion
            messages_list = [dict(msg) for msg in messages]
            response = await self.chat_completion(messages_list)

            # Ensure response is a dict, not an AsyncIterator
            if not isinstance(response, dict):
                _LOGGER.error("Expected dict response, got: %s", type(response))
                return "I apologize, but I couldn't generate a proper response."

            # Validate and extract response with proper type narrowing
            try:
                # Type validation: response should be a MistralResponse
                if not isinstance(response, dict):
                    raise ValueError("Response is not a dictionary")

                mistral_response = cast("MistralResponse", response)

                # Validate choices field
                choices = mistral_response.get("choices")
                if not isinstance(choices, list):
                    raise ValueError("Response choices is not a list")

                if not choices:
                    raise ValueError("No choices in response")

                choice = choices[0]
                if not isinstance(choice, dict):
                    raise ValueError("Choice is not a dictionary")

                # Validate message field
                message = choice.get("message")
                if not isinstance(message, dict):
                    raise ValueError("Message is not a dictionary")

                # Validate content field
                content = message.get("content")
                if not isinstance(content, str):
                    raise ValueError("Content is not a string")

                return content.strip()

            except (ValueError, KeyError, TypeError) as err:
                _LOGGER.error(
                    "Invalid response format from Mistral AI: %s. Response: %s",
                    err,
                    response,
                )
                return "I apologize, but I received an invalid response from the AI service."

        except ConfigEntryError as err:
            _LOGGER.error("Error generating response: %s", err)
            # Provide user-friendly error messages for common issues
            error_msg = str(err)
            if "Unauthorized" in error_msg:
                return "I apologize, but there's an authentication issue with the AI service."
            elif "Forbidden" in error_msg:
                return (
                    "I apologize, but the AI service access is not properly configured."
                )
            elif "Rate limit exceeded" in error_msg:
                return "I apologize, but the AI service is temporarily busy. Please try again shortly."
            elif "Service unavailable" in error_msg:
                return "I apologize, but the AI service is currently unavailable. Please try again later."
            elif "Server error" in error_msg:
                return "I apologize, but the AI service encountered an internal error."
            else:
                return f"I apologize, but I encountered an error: {error_msg}"
        except Exception as err:
            _LOGGER.error("Unexpected error generating response: %s", err)
            return "I apologize, but I encountered an unexpected error while processing your request."

    async def close(self) -> None:
        """Close the client session."""
        # The session is managed by Home Assistant, so we don't close it here
        pass
