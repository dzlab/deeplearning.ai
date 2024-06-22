import json
from typing import *

import httpx

from ..api_config import APIConfig, HTTPException
from ..models import *


async def get_v1forecast(
    latitude: float,
    longitude: float,
    hourly: Optional[List[str]] = None,
    daily: Optional[List[str]] = None,
    current_weather: Optional[bool] = None,
    temperature_unit: Optional[str] = None,
    wind_speed_unit: Optional[str] = None,
    timeformat: Optional[str] = None,
    timezone: Optional[str] = None,
    past_days: Optional[int] = None,
    api_config_override: Optional[APIConfig] = None,
) -> Dict[str, Any]:
    api_config = api_config_override if api_config_override else APIConfig()

    base_path = api_config.base_path
    path = f"/v1/forecast"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer { api_config.get_access_token() }",
    }
    query_params: Dict[str, Any] = {
        "hourly": hourly,
        "daily": daily,
        "latitude": latitude,
        "longitude": longitude,
        "current_weather": current_weather,
        "temperature_unit": temperature_unit,
        "wind_speed_unit": wind_speed_unit,
        "timeformat": timeformat,
        "timezone": timezone,
        "past_days": past_days,
    }

    query_params = {key: value for (key, value) in query_params.items() if value is not None}

    async with httpx.AsyncClient(base_url=base_path, verify=api_config.verify) as client:
        response = await client.request(
            "get",
            httpx.URL(path),
            headers=headers,
            params=query_params,
        )

    if response.status_code != 200:
        raise HTTPException(response.status_code, f" failed with status code: {response.status_code}")

    return response.json()
