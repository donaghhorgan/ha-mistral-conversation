"""Config flow for Mistral AI Conversation integration."""

from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TemplateSelector,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)

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
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_API_KEY): TextSelector(
            TextSelectorConfig(type=TextSelectorType.PASSWORD, autocomplete="off")
        ),
    }
)

OPTIONS_SCHEMA = vol.Schema(
    {
        vol.Optional(CONF_MODEL, default=DEFAULT_MODEL): SelectSelector(
            SelectSelectorConfig(
                options=[],
                mode=SelectSelectorMode.DROPDOWN,
            )
        ),
        vol.Optional(CONF_TEMPERATURE, default=DEFAULT_TEMPERATURE): NumberSelector(
            NumberSelectorConfig(
                min=0.0,
                max=2.0,
                step=0.1,
                mode=NumberSelectorMode.SLIDER,
            )
        ),
        vol.Optional(CONF_MAX_TOKENS, default=DEFAULT_MAX_TOKENS): NumberSelector(
            NumberSelectorConfig(
                min=1,
                max=4000,
                step=1,
                mode=NumberSelectorMode.BOX,
            )
        ),
        vol.Optional(CONF_PROMPT, default=DEFAULT_PROMPT): TemplateSelector(),
        vol.Optional(CONF_LLM_HASS_API): SelectSelector(
            SelectSelectorConfig(
                options=[],
                mode=SelectSelectorMode.DROPDOWN,
            )
        ),
    }
)


async def validate_input(hass: HomeAssistant, data: dict[str, Any]) -> dict[str, Any]:
    """Validate the user input allows us to connect."""
    client = MistralAIClient(
        hass,
        data[CONF_API_KEY],
        data.get(CONF_MODEL, DEFAULT_MODEL),
    )

    if not await client.test_connection():
        raise InvalidAuth

    # Return info that you want to store in the config entry.
    return {"title": f"Mistral AI ({data.get(CONF_MODEL, DEFAULT_MODEL)})"}


class ConfigFlow(config_entries.ConfigFlow):
    """Handle a config flow for Mistral AI Conversation."""

    VERSION = 1
    DOMAIN = DOMAIN

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        """Handle the initial step."""
        if user_input is None:
            return self.async_show_form(
                step_id="user", data_schema=STEP_USER_DATA_SCHEMA
            )

        errors = {}

        try:
            info = await validate_input(self.hass, user_input)
        except CannotConnect:
            errors["base"] = "cannot_connect"
        except InvalidAuth:
            errors["base"] = "invalid_auth"
        except Exception:  # pylint: disable=broad-except
            _LOGGER.exception("Unexpected exception")
            errors["base"] = "unknown"
        else:
            return self.async_create_entry(title=info["title"], data=user_input)

        return self.async_show_form(
            step_id="user", data_schema=STEP_USER_DATA_SCHEMA, errors=errors
        )

    @staticmethod
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> OptionsFlow:
        """Create the options flow."""
        return OptionsFlow(config_entry)


async def get_available_models(hass: HomeAssistant, api_key: str) -> list[str]:
    """Get available models from Mistral AI API."""
    try:
        client = MistralAIClient(hass, api_key)
        models_data = await client.get_available_models()
        return [model["id"] for model in models_data if model.get("id")]
    except Exception as err:
        _LOGGER.error("Error fetching available models: %s", err)
        return []


class OptionsFlow(config_entries.OptionsFlow):
    """Mistral AI config flow options handler."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        self.config_entry = config_entry
        self.available_models: list[str] = []

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        """Manage the options."""
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        # Fetch available models from API
        api_key = self.config_entry.data.get(CONF_API_KEY)
        if api_key:
            self.available_models = await get_available_models(self.hass, api_key)

        # Create dynamic schema with available models
        dynamic_schema = vol.Schema(
            {
                vol.Optional(
                    CONF_MODEL,
                    default=self.config_entry.options.get(CONF_MODEL, DEFAULT_MODEL),
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=self.available_models
                        if self.available_models
                        else [DEFAULT_MODEL],
                        mode=SelectSelectorMode.DROPDOWN,
                    )
                ),
                vol.Optional(
                    CONF_TEMPERATURE,
                    default=self.config_entry.options.get(
                        CONF_TEMPERATURE, DEFAULT_TEMPERATURE
                    ),
                ): NumberSelector(
                    NumberSelectorConfig(
                        min=0.0,
                        max=2.0,
                        step=0.1,
                        mode=NumberSelectorMode.SLIDER,
                    )
                ),
                vol.Optional(
                    CONF_MAX_TOKENS,
                    default=self.config_entry.options.get(
                        CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS
                    ),
                ): NumberSelector(
                    NumberSelectorConfig(
                        min=1,
                        max=4000,
                        step=1,
                        mode=NumberSelectorMode.BOX,
                    )
                ),
                vol.Optional(
                    CONF_PROMPT,
                    default=self.config_entry.options.get(CONF_PROMPT, DEFAULT_PROMPT),
                ): TemplateSelector(),
                vol.Optional(CONF_LLM_HASS_API): SelectSelector(
                    SelectSelectorConfig(
                        options=[],
                        mode=SelectSelectorMode.DROPDOWN,
                    )
                ),
            }
        )

        return self.async_show_form(
            step_id="init",
            data_schema=dynamic_schema,
        )


class CannotConnect(Exception):
    """Error to indicate we cannot connect."""


class InvalidAuth(Exception):
    """Error to indicate there is invalid auth."""
