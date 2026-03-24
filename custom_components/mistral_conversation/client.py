"""Mistral AI client for Home Assistant conversation agent."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, TypedDict, cast

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

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
            _LOGGER.error("Mistral AI API error: %s - %s", response.status, error_text)
            raise ConfigEntryError(
                f"Mistral AI API error: {response.status} - {error_text}"
            )

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
        messages = []

        # Add system context if provided
        if context:
            messages.append({"role": "system", "content": context})

        # Add user message
        messages.append({"role": "user", "content": prompt})

        try:
            response = await self.chat_completion(messages)

            # Ensure response is a dict, not an AsyncIterator
            if not isinstance(response, dict):
                _LOGGER.error("Expected dict response, got: %s", type(response))
                return "I apologize, but I couldn't generate a proper response."

            # Type narrowing for response dict
            if isinstance(response, dict):
                # Cast to MistralResponse type for proper type checking
                mistral_response = cast("MistralResponse", response)
                choices = mistral_response.get("choices")
                if isinstance(choices, list) and choices:
                    choice = choices[0]
                    if isinstance(choice, dict):
                        message = choice.get("message")
                        if isinstance(message, dict):
                            content = message.get("content")
                            if isinstance(content, str):
                                return content.strip()

            _LOGGER.error("Unexpected response format from Mistral AI: %s", response)
            return "I apologize, but I couldn't generate a proper response."

        except Exception as err:
            _LOGGER.error("Error generating response: %s", err)
            return f"I encountered an error: {err}"

    async def close(self) -> None:
        """Close the client session."""
        # The session is managed by Home Assistant, so we don't close it here
        pass
