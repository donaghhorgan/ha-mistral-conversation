"""Config flow for Mistral AI Conversation integration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import aiohttp
import voluptuous as vol
from homeassistant import config_entries
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TemplateSelector,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)
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
    DOMAIN,
)

if TYPE_CHECKING:
    from types import MappingProxyType

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
    # Validate API key
    api_key = data.get(CONF_API_KEY)
    if not api_key:
        raise InvalidAuth("API key is required")

    if not isinstance(api_key, str):
        raise InvalidAuth("API key must be a string")

    # Validate model
    model = data.get(CONF_MODEL, DEFAULT_MODEL)
    if not isinstance(model, str):
        raise InvalidAuth("Model must be a string")

    if not model.strip():
        raise InvalidAuth("Model cannot be empty")

    # Validate temperature if provided
    temperature = data.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
    if temperature is not None:
        if not isinstance(temperature, (int, float)):
            raise InvalidAuth("Temperature must be a number")
        if not (0.0 <= temperature <= 2.0):
            raise InvalidAuth("Temperature must be between 0.0 and 2.0")

    # Validate max_tokens if provided
    max_tokens = data.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)
    if max_tokens is not None:
        if not isinstance(max_tokens, int):
            raise InvalidAuth("Max tokens must be an integer")
        if not (max_tokens >= 1):
            raise InvalidAuth("Max tokens must be at least 1")

    # Validate prompt if provided
    prompt = data.get(CONF_PROMPT, DEFAULT_PROMPT)
    if prompt is not None and not isinstance(prompt, str):
        raise InvalidAuth("Prompt must be a string")

    try:
        client = Mistral(api_key=data[CONF_API_KEY])

        # Test connection by listing available models
        try:
            models = await client.list_models()
            if not models.data:
                raise CannotConnect(
                    "Failed to connect to Mistral AI API. Please check your API key and network connection."
                ) from None
        except Exception as err:
            raise CannotConnect(f"Failed to connect to Mistral AI API: {err}") from err

    except aiohttp.ClientError as err:
        if "timeout" in str(err).lower():
            raise CannotConnect(
                "Connection timed out. Please check your network connection."
            ) from err
        elif "ssl" in str(err).lower():
            raise CannotConnect(
                "SSL certificate error. Please check your system date/time and SSL certificates."
            ) from err
        else:
            raise CannotConnect(f"Network error: {str(err)}") from err
    except Exception as err:
        _LOGGER.error("Unexpected error during validation: %s", err)
        raise CannotConnect(f"Unexpected error: {str(err)}") from err

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
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> OptionsFlow:
        """Create the options flow."""
        return OptionsFlow(config_entry)


async def get_available_models(hass: HomeAssistant, api_key: str) -> list[str]:
    """Get available models from Mistral AI API."""
    try:
        client = Mistral(api_key=api_key)
        models = await client.list_models()
        return [model.id for model in models.data if hasattr(model, "id")]
    except Exception as err:
        _LOGGER.error("Error fetching available models: %s", err)
        return []


class OptionsFlow(config_entries.OptionsFlow):
    """Mistral AI config flow options handler."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        self.config_entry = config_entry
        self.available_models: list[str] = []

    @callback
    def async_get_options_schema(
        self, options: MappingProxyType[str, Any]
    ) -> vol.Schema:
        """Return the options schema."""
        # Get available LLM APIs
        apis: list[SelectOptionDict] = [
            SelectOptionDict(
                label=api.name,
                value=api.id,
            )
            for api in llm.async_get_apis(self.hass)
        ]

        # Fetch available models from API
        api_key = self.config_entry.data.get(CONF_API_KEY)
        if api_key:
            # This will be populated in async_step_init
            self.available_models = []

        return vol.Schema(
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
                vol.Optional(
                    CONF_LLM_HASS_API,
                    description={"suggested_value": options.get(CONF_LLM_HASS_API)},
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=apis,
                        mode=SelectSelectorMode.DROPDOWN,
                    )
                ),
            }
        )

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

        return self.async_show_form(
            step_id="init",
            data_schema=self.async_get_options_schema(self.config_entry.options),
        )


class CannotConnect(HomeAssistantError):
    """Error to indicate we cannot connect."""


class InvalidAuth(HomeAssistantError):
    """Error to indicate there is invalid auth."""
