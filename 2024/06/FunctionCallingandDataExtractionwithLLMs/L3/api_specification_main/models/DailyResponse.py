from typing import *

from pydantic import BaseModel, Field


class DailyResponse(BaseModel):
    """
    None model

    """

    time: List[str] = Field(alias="time")

    temperature_2m_max: Optional[List[float]] = Field(alias="temperature_2m_max", default=None)

    temperature_2m_min: Optional[List[float]] = Field(alias="temperature_2m_min", default=None)

    apparent_temperature_max: Optional[List[float]] = Field(alias="apparent_temperature_max", default=None)

    apparent_temperature_min: Optional[List[float]] = Field(alias="apparent_temperature_min", default=None)

    precipitation_sum: Optional[List[float]] = Field(alias="precipitation_sum", default=None)

    precipitation_hours: Optional[List[float]] = Field(alias="precipitation_hours", default=None)

    weather_code: Optional[List[float]] = Field(alias="weather_code", default=None)

    sunrise: Optional[List[float]] = Field(alias="sunrise", default=None)

    sunset: Optional[List[float]] = Field(alias="sunset", default=None)

    wind_speed_10m_max: Optional[List[float]] = Field(alias="wind_speed_10m_max", default=None)

    wind_gusts_10m_max: Optional[List[float]] = Field(alias="wind_gusts_10m_max", default=None)

    wind_direction_10m_dominant: Optional[List[float]] = Field(alias="wind_direction_10m_dominant", default=None)

    shortwave_radiation_sum: Optional[List[float]] = Field(alias="shortwave_radiation_sum", default=None)

    uv_index_max: Optional[List[float]] = Field(alias="uv_index_max", default=None)

    uv_index_clear_sky_max: Optional[List[float]] = Field(alias="uv_index_clear_sky_max", default=None)

    et0_fao_evapotranspiration: Optional[List[float]] = Field(alias="et0_fao_evapotranspiration", default=None)
