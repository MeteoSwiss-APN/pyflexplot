"""
This module (taken from MeteoSwiss Python Commons library) introduces a custom class that leverages Pydantic's capabilities: a custom BaseSettings class BaseServiceSettings.

This custom classes inherit Pydantic's core functionalities. This includes data validation, parsing,
serialization, error handling, and behavior customization.
"""
import logging
import os
from typing import Union, Any, Type

import yaml
from pydantic import BaseModel, Extra
from pydantic.fields import FieldInfo
from pydantic.v1.utils import deep_update
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict, InitSettingsSource

logger = logging.getLogger(__name__)



class BaseServiceSettings(BaseSettings):
    """
    The Custom BaseSettings class is a derivative of Pydantic's BaseSettings class. It introduces the ability to read
    settings values from a series of YAML files, providing an additional source for configuration data.
    """
    model_config = SettingsConfigDict(env_nested_delimiter='__', extra=Extra.allow)

    def __init__(self, settings_file_names: Union[str, list[str]], settings_dirname: str, **values: Any):
        """
        Initializes the service settings by loading values from the specified YAML filename(s).
        The YAML settings will be loaded in the same order as given in the list (or as a single settings file),
        with later files taking precedence.
        :param settings_file_names: Names of YAML files containing settings.
        :param settings_dirname: Directory where the settings YAML files are located.
        :param values: Additional settings values (if provided). Refer to Pydantic's BaseSettings for details.
        :raises FileNotFoundError: If none of the specified YAML settings files exist.
        """

        if isinstance(settings_file_names, str):
            settings_file_names = [settings_file_names]

        # we want to harmlessly be able to pass files that do not exist in reality
        existing_settings_file_names = []
        for fname in settings_file_names:
            if os.path.exists(os.path.join(settings_dirname, fname)):
                existing_settings_file_names.append(fname)
            else:
                logger.warning('Given YAML settings file "%s" does not exist.', fname)

        if not existing_settings_file_names:
            raise FileNotFoundError('Could not find the specified YAML settings file(s).')

        super().__init__(_settings_filenames=existing_settings_file_names, _settings_dirname=settings_dirname, **values)

    # pylint: disable=too-many-arguments
    @classmethod
    def settings_customise_sources(
            cls,
            settings_cls: Type[BaseSettings],
            init_settings: InitSettingsSource,  # type: ignore # the init_settings is always a InitSettingsSource
            env_settings: PydanticBaseSettingsSource,
            dotenv_settings: PydanticBaseSettingsSource,
            file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, PydanticBaseSettingsSource, PydanticBaseSettingsSource,
    PydanticBaseSettingsSource]:
        load_yaml_config = _YamlConfigSource(settings_cls=settings_cls,
                                             filenames=init_settings.init_kwargs.pop('_settings_filenames'),
                                             dirname=init_settings.init_kwargs.pop('_settings_dirname'))

        # environment variables have highest precedence, then yaml values etc.
        return env_settings, load_yaml_config, init_settings, file_secret_settings


class _YamlConfigSource(PydanticBaseSettingsSource):
    _yaml_content_dict: dict[str, Any]

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        # this method needs to be implemented but is not used since the whole parsed
        # yaml file is returned as dictionary without inspecting the values or fields
        raise NotImplementedError()

    def __init__(self, filenames: list[str], dirname: str, settings_cls: type[BaseSettings]):
        self._yaml_content_dict = {}

        for filename in filenames:
            with open(os.path.join(dirname, filename), encoding='utf-8') as file:
                self._yaml_content_dict = deep_update(self._yaml_content_dict, yaml.safe_load(file))

        super().__init__(settings_cls)

    def __call__(self) -> dict[str, Any]:
        return self._yaml_content_dict
