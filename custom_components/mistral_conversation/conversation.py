"""Mistral AI conversation agent for Home Assistant."""

from __future__ import annotations

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import CONF_MODEL, CONF_PROMPT, DEFAULT_MODEL, DOMAIN
from .entity import MistralBaseLLMEntity


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Mistral AI conversation platform."""
    agent = MistralConversationEntity(hass, config_entry)
    async_add_entities([agent])


class MistralConversationEntity(
    conversation.ConversationEntity,
    conversation.AbstractConversationAgent,
    MistralBaseLLMEntity,
):
    """Mistral AI conversation agent."""

    _attr_supports_streaming = True

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the agent."""
        # Initialize base entity
        super().__init__(entry)
        self.hass = hass
        # Set the entity name
        model = entry.options.get(CONF_MODEL, DEFAULT_MODEL)
        self._attr_name = f"Mistral AI ({model})"

    @property
    def supported_languages(self) -> list[str] | str:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def async_added_to_hass(self) -> None:
        """When entity is added to hass."""
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from hass."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Process the user input and call the API."""
        options = self.entry.data

        try:
            # Use the standard Home Assistant LLM prompt system
            # This supports custom prompts while falling back to llm.DEFAULT_INSTRUCTIONS_PROMPT
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                options.get(CONF_LLM_HASS_API),
                options.get(CONF_PROMPT),  # Support custom prompts like reference integrations
                user_input.extra_system_prompt,
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        # Use streaming if supported, otherwise fall back to non-streaming
        if self._attr_supports_streaming:
            await self._async_handle_chat_log_streaming(chat_log)
        else:
            await self._async_handle_chat_log(chat_log)

        return conversation.async_get_result_from_chat_log(user_input, chat_log)

    @property
    def attribution(self) -> str:
        """Return the attribution."""
        return "Powered by Mistral AI"
