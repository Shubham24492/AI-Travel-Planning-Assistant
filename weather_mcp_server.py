import httpx
from mcp.server.fastmcp import FastMCP


mcp = FastMCP("Singapore Weather Server")


@mcp.tool(description="Find the latitude and longitude of a city.")
def find_city(city: str) -> dict:
    response = httpx.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1, "language": "en", "format": "json"},
        timeout=10,
    )
    response.raise_for_status()
    results = response.json().get("results", [])
    if not results:
        return {"error": f"City not found: {city}"}

    result = results[0]
    return {
        "name": result["name"],
        "country": result.get("country"),
        "latitude": result["latitude"],
        "longitude": result["longitude"],
    }


@mcp.tool(description="Get current conditions and a daily forecast for latitude and longitude.")
def get_weather(latitude: float, longitude: float, forecast_days: int = 3) -> dict:
    forecast_days = max(1, min(forecast_days, 16))
    response = httpx.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "forecast_days": forecast_days,
            "timezone": "Asia/Singapore",
        },
        timeout=10,
    )
    response.raise_for_status()
    payload = response.json()
    return {"current": payload.get("current", {}), "daily": payload.get("daily", {})}


if __name__ == "__main__":
    mcp.run()