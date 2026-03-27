"""Mistral AI conversation agent for Home Assistant."""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from homeassistant.components import conversation
from homeassistant.components.conversation import chat_log
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady, TemplateError
from homeassistant.helpers import chat_session as chat_session_helper
from homeassistant.helpers import intent, template
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.util import ulid
from mistralai.client import Mistral

from .const import (
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


# Type definitions for better type hints
class MistralTool(TypedDict):
    """Type definition for Mistral AI tool format."""

    type: str
    function: dict[str, Any]


class MistralMessage(TypedDict):
    """Type definition for Mistral AI message format."""

    role: str
    content: str


class MistralMessageWithToolCalls(MistralMessage, total=False):
    """Type definition for Mistral AI message with tool calls."""

    tool_calls: list[dict[str, Any]]


class MistralToolMessage(TypedDict):
    """Type definition for Mistral AI tool message format."""

    role: str
    name: str
    content: str


class MistralParameters(TypedDict, total=False):
    """Type definition for Mistral AI tool parameters."""

    type: str
    properties: dict[str, dict[str, str]]
    required: list[str]


def _format_tool(tool: Any) -> MistralTool:
    """Format a Home Assistant tool for Mistral AI API."""
    tool_schema: MistralTool = {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "strict": True,  # Always use strict mode for tool calls
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
            parameters: MistralParameters = {"type": "object", "properties": properties}
            if required:
                parameters["required"] = required
            tool_schema["function"]["parameters"] = parameters

    return tool_schema


def _convert_content(content: chat_log.Content) -> MistralMessage:
    """Convert Home Assistant chat content to Mistral AI message format."""
    content_text = content.content or ""  # type: ignore[unresolved-attr]
    if content.role == "user":
        return {"role": "user", "content": content_text}
    elif content.role == "assistant":
        return {"role": "assistant", "content": content_text}
    else:
        # Handle system messages or other roles
        return {"role": "system", "content": content_text}


def _transform_stream(response: Any) -> MistralMessage:
    """Transform Mistral AI streaming response for Home Assistant chat log."""
    # For now, return a simple structure that can be handled by the chat log
    # This would need to be enhanced for actual streaming support
    return {"content": str(response), "role": "assistant"}


_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Mistral AI conversation platform."""
    agent = MistralConversationEntity(hass, config_entry)
    async_add_entities([agent])


class MistralConversationEntity(conversation.ConversationEntity):
    """Mistral AI conversation agent."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the agent."""
        self.hass = hass
        self.entry = entry
        self._client: Mistral | None = None
        self._attr_name = f"Mistral AI ({entry.options.get(CONF_MODEL, DEFAULT_MODEL)})"
        self._attr_unique_id = entry.entry_id

    @property
    def supported_languages(self) -> list[str]:
        """Return a list of supported languages."""
        return [conversation.MATCH_ALL]

    async def async_added_to_hass(self) -> None:
        """When entity is added to hass."""
        await super().async_added_to_hass()

        # Initialize Mistral client in executor to avoid blocking the event loop
        def _init_client():
            return Mistral(api_key=self.entry.data[CONF_API_KEY])

        self._client = await self.hass.async_add_executor_job(_init_client)

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        """Process a sentence."""
        if self._client is None:
            raise ConfigEntryNotReady("Mistral AI client not initialized")

        # Validate user input
        if not user_input.text:
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                "Sorry, I received empty input. Please provide some text to process.",
            )
            return conversation.ConversationResult(
                response=intent_response,
                conversation_id=user_input.conversation_id,
            )

        if not isinstance(user_input.text, str):
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                "Sorry, I received invalid input format. Please provide text input.",
            )
            return conversation.ConversationResult(
                response=intent_response,
                conversation_id=user_input.conversation_id,
            )

        # Validate conversation ID if provided
        if user_input.conversation_id is not None and not isinstance(
            user_input.conversation_id, str
        ):
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                "Sorry, there's an issue with the conversation ID.",
            )
            return conversation.ConversationResult(
                response=intent_response,
                conversation_id=user_input.conversation_id,
            )

        # Use chat log for conversation history management
        conversation_id = user_input.conversation_id or ulid.ulid()
        chat_session = chat_session_helper.ChatSession(
            conversation_id=conversation_id,
        )

        raw_prompt = self.entry.data.get(CONF_PROMPT, DEFAULT_PROMPT)
        llm_api = self.entry.options.get(CONF_LLM_HASS_API)

        if llm_api:
            try:
                llm_api = self.hass.data["llm"][llm_api]
            except KeyError:
                _LOGGER.error("LLM API %s not found", llm_api)
            except Exception as err:
                _LOGGER.error("Error formatting tools: %s", err)

        if raw_prompt:
            try:
                prompt_template = template.Template(raw_prompt, self.hass)
                prompt = prompt_template.async_render(
                    {
                        "ha_name": self.hass.config.location_name,
                    },
                    parse_result=False,
                )
            except TemplateError as err:
                _LOGGER.error("Error rendering prompt template: %s", err)
                prompt = DEFAULT_PROMPT
        else:
            prompt = DEFAULT_PROMPT

        _LOGGER.debug("Prompt: %s", prompt)
        _LOGGER.debug("User input: %s", user_input.text)

        try:
            # Use Home Assistant's recommended LLM API integration
            with chat_log.async_get_chat_log(
                self.hass, chat_session, user_input
            ) as chat_log_instance:
                # Provide LLM data using the standard Home Assistant method
                try:
                    await chat_log_instance.async_provide_llm_data(
                        user_input.as_llm_context("mistral_conversation"),
                        self.entry.options.get(CONF_LLM_HASS_API),
                        self.entry.options.get(CONF_PROMPT),
                        user_input.extra_system_prompt,
                    )
                except conversation.ConverseError as err:
                    return err.as_conversation_result()

                # Extract conversation history from chat log for Mistral API
                conversation_history = []
                for content in chat_log_instance.content:
                    if content.role in ("user", "assistant") and hasattr(
                        content, "content"
                    ):
                        conversation_history.append(
                            {"role": content.role, "content": content.content}
                        )

                # Generate response from Mistral AI using the official client
                messages = []

                # Add system context if provided
                if prompt:
                    messages.append({"role": "system", "content": prompt})

                # Add conversation history if provided (respecting token limits)
                if conversation_history:
                    # Add recent history messages until we hit a reasonable limit
                    history_messages_to_add = []

                    # Add most recent messages first until we reach the limit
                    for msg in reversed(conversation_history):
                        # Simple character-based approximation for now
                        if len(msg["content"]) < 1000:  # Reasonable message size
                            history_messages_to_add.insert(
                                0, {"role": msg["role"], "content": msg["content"]}
                            )
                            # Stop if we have enough history
                            if len(history_messages_to_add) >= 10:
                                break

                    # Add history messages after system prompt
                    for msg in history_messages_to_add:
                        messages.insert(1, msg)

                # Add user message (only if it's not already in the conversation history)
                if (
                    not conversation_history
                    or conversation_history[-1].get("content") != user_input.text
                ):
                    messages.append({"role": "user", "content": user_input.text})

                # Format tools for Mistral AI API if available
                tools = None
                if chat_log_instance.llm_api:
                    tools = [
                        _format_tool(tool) for tool in chat_log_instance.llm_api.tools
                    ]

                # Tool calling iteration loop (up to 10 iterations)
                for _iteration in range(10):
                    # Make the API call using the official Mistral client
                    chat_response = await self._client.chat.complete_async(
                        model=self.entry.data.get(CONF_MODEL, DEFAULT_MODEL),
                        messages=messages,
                        temperature=self.entry.data.get(
                            CONF_TEMPERATURE, DEFAULT_TEMPERATURE
                        ),
                        max_tokens=self.entry.data.get(
                            CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS
                        ),
                        tools=tools if tools else None,
                    )

                    # Extract the response content
                    response = chat_response.choices[0].message.content
                    tool_calls = getattr(
                        chat_response.choices[0].message, "tool_calls", None
                    )

                    # Add assistant response to chat log
                    chat_log_instance.async_add_assistant_content(
                        chat_log.AssistantContent(
                            agent_id=self.unique_id or self.entry.entry_id,
                            content=response,
                        )
                    )

                    # Handle tool calls if any
                    if tool_calls and chat_log_instance.llm_api:
                        for tool_call in tool_calls:
                            tool_name = tool_call.function.name
                            tool_args = tool_call.function.arguments

                            # Find and call the corresponding tool
                            for tool in chat_log_instance.llm_api.tools:
                                if tool.name == tool_name:
                                    try:
                                        # Call the tool
                                        tool_result = await tool.async_call(
                                            self.hass,
                                            tool_input=tool_args,
                                            llm_context=user_input.as_llm_context(
                                                "mistral_conversation"
                                            ),
                                        )

                                        # Add tool result to chat log
                                        chat_log_instance.async_add_tool_result(  # type: ignore[attr-defined]
                                            chat_log.ToolResult(  # type: ignore[attr-defined]
                                                tool_name=tool_name,
                                                tool_args=tool_args,
                                                result=tool_result,
                                            )
                                        )

                                        # Add tool call and result to messages for next iteration
                                        messages.append(
                                            {
                                                "role": "assistant",
                                                "content": response,
                                                "tool_calls": [
                                                    {
                                                        "id": tool_call.id,
                                                        "type": "function",
                                                        "function": {
                                                            "name": tool_name,
                                                            "arguments": tool_args,
                                                        },
                                                    }
                                                ],
                                            }
                                        )

                                        messages.append(
                                            {
                                                "role": "tool",
                                                "name": tool_name,
                                                "content": str(tool_result),
                                            }
                                        )

                                    except Exception as err:
                                        _LOGGER.error(
                                            "Error calling tool %s: %s", tool_name, err
                                        )
                                        chat_log_instance.async_add_tool_result(  # type: ignore[attr-defined]
                                            chat_log.ToolResult(  # type: ignore[attr-defined]
                                                tool_name=tool_name,
                                                tool_args=tool_args,
                                                error=str(err),
                                            )
                                        )

                                        # Add error response to messages
                                        messages.append(
                                            {
                                                "role": "tool",
                                                "name": tool_name,
                                                "content": f"Error: {str(err)}",
                                            }
                                        )

                    if not chat_log_instance.unresponded_tool_results:
                        break

                # Store conversation ID for reference
                _LOGGER.debug("Conversation %s updated with response", conversation_id)

                intent_response = intent.IntentResponse(language=user_input.language)
                intent_response.async_set_speech(response)

                return conversation.ConversationResult(
                    response=intent_response,
                    conversation_id=conversation_id,
                    continue_conversation=True,
                )
        except Exception as err:
            _LOGGER.error("Error generating response: %s", err)
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Sorry, I had a problem talking to Mistral AI: {err}",
            )
            return conversation.ConversationResult(
                response=intent_response,
                conversation_id=conversation_id,
            )

        return conversation.ConversationResult(
            response=intent_response,
            conversation_id=conversation_id,
            continue_conversation=True,
        )

    @property
    def attribution(self) -> str:
        """Return the attribution."""
        return "Powered by Mistral AI"
