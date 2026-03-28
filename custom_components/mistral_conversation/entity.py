"""Base entity for Mistral AI Conversation integration."""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

import mistralai
import voluptuous as vol
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import llm
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.json import json_dumps
from mistralai.client import Mistral
from voluptuous_openapi import convert

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

from .const import (
    CONF_API_KEY,
    CONF_MAX_TOKENS,
    CONF_MODEL,
    CONF_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)

# Max number of back and forth with the LLM to generate a response
MAX_TOOL_ITERATIONS = 10


def _format_tool(
    tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None = None
) -> dict[str, Any]:
    """Format a Home Assistant tool for Mistral AI API."""
    tool_spec: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
        },
    }

    # Add parameters if tool has a schema
    if hasattr(tool, "parameters") and tool.parameters:
        properties: dict[str, dict[str, str]] = {}
        required: list[str] = []

        for param_name, param_schema in tool.parameters.schema.items():
            param_info: dict[str, str] = {"type": "string"}  # Default to string
            if hasattr(param_schema, "description"):
                param_info["description"] = param_schema.description
            if param_name in param_schema.required:
                required.append(param_name)

            properties[param_name] = param_info

        if properties:
            parameters: dict[str, Any] = {"type": "object", "properties": properties}
            if required:
                parameters["required"] = required
            tool_spec["function"]["parameters"] = parameters

    return tool_spec


def _convert_content_to_mistral(
    content: conversation.Content,
) -> dict[str, Any]:
    """Convert Home Assistant chat content to Mistral AI message format."""
    if isinstance(content, conversation.UserContent):
        return {"role": "user", "content": content.content}
    elif isinstance(content, conversation.AssistantContent):
        message: dict[str, Any] = {"role": "assistant", "content": content.content or ""}
        if content.tool_calls:
            message["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.tool_name,
                        "arguments": json_dumps(tool_call.tool_args),
                    },
                }
                for tool_call in content.tool_calls
            ]
        return message
    elif isinstance(content, conversation.SystemContent):
        return {"role": "system", "content": content.content}
    elif isinstance(content, conversation.ToolResultContent):
        return {
            "role": "tool",
            "name": content.tool_name,
            "content": json_dumps(content.tool_result),
            "tool_call_id": content.tool_call_id,
        }
    else:
        raise HomeAssistantError(f"Unsupported content type: {type(content)}")


async def _transform_stream(
    response_stream: AsyncGenerator[Any],
) -> AsyncGenerator[conversation.AssistantContentDeltaDict]:
    """Transform Mistral AI streaming response for Home Assistant chat log."""
    async for stream_response in response_stream:
        # Handle different response formats from Mistral AI
        if hasattr(stream_response, 'choices') and stream_response.choices:
            choice = stream_response.choices[0]
            if hasattr(choice, 'delta'):
                delta = choice.delta
                if hasattr(delta, 'content') and delta.content:
                    yield {"content": delta.content}
                if hasattr(delta, 'tool_calls') and delta.tool_calls:
                    tool_calls = []
                    for tool_call in delta.tool_calls:
                        if hasattr(tool_call, 'function') and tool_call.function:
                            tool_calls.append(
                                llm.ToolInput(
                                    id=tool_call.id,
                                    tool_name=tool_call.function.name,
                                    tool_args=tool_call.function.arguments or {},
                                )
                            )
                    if tool_calls:
                        yield {"tool_calls": tool_calls}
        elif hasattr(stream_response, 'data'):
            # Handle alternative response format
            data = stream_response.data
            if hasattr(data, 'choices') and data.choices:
                choice = data.choices[0]
                if hasattr(choice, 'delta'):
                    delta = choice.delta
                    if hasattr(delta, 'content') and delta.content:
                        yield {"content": delta.content}


class MistralBaseLLMEntity(Entity):
    """Mistral AI base LLM entity."""

    _attr_has_entity_name = True
    _attr_name: str | None = None

    def __init__(self, entry: ConfigEntry, subentry: conversation.ConfigSubentry | None = None) -> None:
        """Initialize the entity."""
        self.entry = entry
        self.subentry = subentry
        self._attr_unique_id = subentry.subentry_id if subentry else entry.entry_id

        model = subentry.data.get(CONF_MODEL, DEFAULT_MODEL) if subentry else DEFAULT_MODEL
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, self._attr_unique_id)},
            name=subentry.title if subentry else f"Mistral AI ({model})",
            manufacturer="Mistral AI",
            model=model,
            entry_type=dr.DeviceEntryType.SERVICE,
        )

        self._client: Mistral | None = None

    async def async_added_to_hass(self) -> None:
        """When entity is added to hass."""
        await super().async_added_to_hass()

        # Initialize Mistral client
        def _init_client():
            return Mistral(api_key=self.entry.data[CONF_API_KEY])

        self._client = await self.hass.async_add_executor_job(_init_client)

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from hass."""
        if self._client:
            # Clean up client if needed
            pass
        await super().async_will_remove_from_hass()

    async def _async_handle_chat_log(
        self,
        chat_log: conversation.ChatLog,
        structure: vol.Schema | None = None,
        max_iterations: int = MAX_TOOL_ITERATIONS,
    ) -> None:
        """Generate an answer for the chat log."""
        if self._client is None:
            raise HomeAssistantError("Mistral AI client not initialized")

        options = self.subentry.data if self.subentry else self.entry.data

        # Convert chat log content to Mistral format
        messages = [_convert_content_to_mistral(content) for content in chat_log.content]

        # Prepare API parameters
        api_params = {
            "model": options.get(CONF_MODEL, DEFAULT_MODEL),
            "messages": messages,
            "temperature": options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE),
            "max_tokens": options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS),
        }

        # Add tools if available
        tools = None
        if chat_log.llm_api:
            tools = [_format_tool(tool, chat_log.llm_api.custom_serializer) for tool in chat_log.llm_api.tools]
            if tools:
                api_params["tools"] = tools

        # Handle structured output if requested
        if structure and chat_log.llm_api:
            output_format = convert(
                structure,
                custom_serializer=chat_log.llm_api.custom_serializer,
            )
            # Mistral doesn't have direct structured output, so we use tool approach
            if tools is None:
                tools = []
            tools.append({
                "type": "function",
                "function": {
                    "name": "structured_output",
                    "description": "Use this tool to provide structured output",
                    "parameters": output_format,
                },
            })
            api_params["tools"] = tools

        # Tool calling iteration loop
        for _iteration in range(max_iterations):
            try:
                # Make the API call
                response = await self._client.chat.complete_async(**api_params)

                # Process the response
                if not response.choices:
                    raise HomeAssistantError("No response from Mistral AI API")

                choice = response.choices[0]
                message = choice.message

                # Add assistant response to chat log
                assistant_content = conversation.AssistantContent(
                    agent_id=self.unique_id,
                    content=message.content or "",
                )

                # Add tool calls if any
                if message.tool_calls:
                    assistant_content.tool_calls = [
                        conversation.ToolCall(
                            id=tool_call.id,
                            tool_name=tool_call.function.name,
                            tool_args=tool_call.function.arguments or {},
                        )
                        for tool_call in message.tool_calls
                    ]

                chat_log.async_add_assistant_content(assistant_content)

                # Handle tool calls
                if message.tool_calls and chat_log.llm_api:
                    for tool_call in message.tool_calls:
                        tool_name = tool_call.function.name
                        tool_args = tool_call.function.arguments or {}

                        # Find and call the corresponding tool
                        for tool in chat_log.llm_api.tools:
                            if tool.name == tool_name:
                                try:
                                    tool_result = await tool.async_call(
                                        self.hass,
                                        tool_input=tool_args,
                                        llm_context=chat_log.llm_context,
                                    )

                                    # Add tool result to chat log
                                    chat_log.async_add_tool_result(
                                        conversation.ToolResult(
                                            tool_name=tool_name,
                                            tool_call_id=tool_call.id,
                                            tool_args=tool_args,
                                            result=tool_result,
                                        )
                                    )

                                    # Add tool result to messages for next iteration
                                    messages.append({
                                        "role": "tool",
                                        "name": tool_name,
                                        "content": json_dumps(tool_result),
                                        "tool_call_id": tool_call.id,
                                    })

                                except Exception as err:
                                    _LOGGER.error("Error calling tool %s: %s", tool_name, err)
                                    chat_log.async_add_tool_result(
                                        conversation.ToolResult(
                                            tool_name=tool_name,
                                            tool_call_id=tool_call.id,
                                            tool_args=tool_args,
                                            error=str(err),
                                        )
                                    )

                                    # Add error response to messages
                                    messages.append({
                                        "role": "tool",
                                        "name": tool_name,
                                        "content": f"Error: {str(err)}",
                                        "tool_call_id": tool_call.id,
                                    })

                if not chat_log.unresponded_tool_results:
                    break

            except mistralai.APIError as err:
                _LOGGER.error("Mistral AI API error: %s", err)
                raise HomeAssistantError(f"Mistral AI API error: {err}") from err
            except mistralai.AuthenticationError as err:
                _LOGGER.error("Mistral AI authentication error: %s", err)
                self.entry.async_start_reauth(self.hass)
                raise HomeAssistantError(
                    "Authentication error with Mistral AI API, reauthentication required"
                ) from err
            except mistralai.RateLimitError as err:
                _LOGGER.error("Mistral AI rate limit error: %s", err)
                raise HomeAssistantError("Mistral AI rate limit exceeded") from err
            except Exception as err:
                _LOGGER.error("Unexpected error from Mistral AI: %s", err)
                raise HomeAssistantError(f"Unexpected error from Mistral AI: {err}") from err

    async def _async_handle_chat_log_streaming(
        self,
        chat_log: conversation.ChatLog,
        structure: vol.Schema | None = None,
    ) -> None:
        """Generate an answer for the chat log with streaming support."""
        if self._client is None:
            raise HomeAssistantError("Mistral AI client not initialized")

        options = self.subentry.data if self.subentry else self.entry.data

        # Convert chat log content to Mistral format
        messages = [_convert_content_to_mistral(content) for content in chat_log.content]

        # Prepare API parameters
        api_params = {
            "model": options.get(CONF_MODEL, DEFAULT_MODEL),
            "messages": messages,
            "temperature": options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE),
            "max_tokens": options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS),
            "stream": True,
        }

        # Add tools if available
        tools = None
        if chat_log.llm_api:
            tools = [_format_tool(tool, chat_log.llm_api.custom_serializer) for tool in chat_log.llm_api.tools]
            if tools:
                api_params["tools"] = tools

        try:
            # Make the streaming API call
            response_stream = await self._client.chat.complete_stream_async(**api_params)

            # Process the streaming response
            async for delta_content in chat_log.async_add_delta_content_stream(
                self.entity_id,
                _transform_stream(response_stream)
            ):
                pass  # Content is added to chat log by the stream processor

        except Exception as err:
            _LOGGER.error("Error in Mistral AI streaming: %s", err)
            raise HomeAssistantError(f"Mistral AI streaming error: {err}") from err
