from typing import *

from pydantic import BaseModel, Field


class CurrentWeather(BaseModel):
    """
    None model

    """

    time: str = Field(alias="time")

    temperature: float = Field(alias="temperature")

    wind_speed: float = Field(alias="wind_speed")

    wind_direction: float = Field(alias="wind_direction")

    weather_code: float = Field(alias="weather_code")
