"""Mistral AI conversation agent for Home Assistant."""

from __future__ import annotations

import logging

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

from .client import MistralAIClient
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
        self._client: MistralAIClient | None = None
        self._attr_name = f"Mistral AI ({entry.data[CONF_MODEL]})"
        self._attr_unique_id = entry.entry_id

    @property
    def supported_languages(self) -> list[str]:
        """Return a list of supported languages."""
        return [conversation.MATCH_ALL]

    async def async_added_to_hass(self) -> None:
        """When entity is added to hass."""
        await super().async_added_to_hass()
        self._client = MistralAIClient(
            self.hass,
            self.entry.data[CONF_API_KEY],
            self.entry.data.get(CONF_MODEL, DEFAULT_MODEL),
            self.entry.data.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE),
            self.entry.data.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS),
        )

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
        llm_api = self.entry.data.get(CONF_LLM_HASS_API)

        if llm_api:
            try:
                llm_api = self.hass.data["llm"][llm_api]
                # Future: implement tool support
                # tools = [
                #     {
                #         "type": "function",
                #         "function": tool.to_openai_function(),
                #     }
                #     for tool in llm_api.tools
                # ]
            except KeyError:
                _LOGGER.error("LLM API %s not found", llm_api)

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

                # Generate response from Mistral AI
                response = await self._client.generate_response(
                    user_input.text,
                    context=prompt,
                    conversation_history=conversation_history,
                )

                # Add assistant response to chat log
                chat_log_instance.async_add_assistant_content_without_tools(
                    chat_log.AssistantContent(
                        agent_id=self.unique_id or self.entry.entry_id,
                        content=response,
                    )
                )

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
