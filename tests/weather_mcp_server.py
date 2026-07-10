"""MCP 天气服务器 — 基于 wttr.in（免费，无需 API Key）。

4 个工具供 Agent 调用:
  - get_weather:    实时天气（温度、湿度、风、能见度、气压、紫外线）
  - get_forecast:   未来 3 天预报
  - get_hourly:     今天逐小时预报
  - get_astro:      日出/日落/月出/月落/月相
"""

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types
from datetime import datetime
from pathlib import Path
import httpx
import json

log_path = Path(__file__).resolve().parent.parent / "mcp_weather_startup.txt"


async def _fetch_json(city: str) -> dict | None:
    """从 wttr.in 获取 JSON 天气数据。"""
    url = f"https://wttr.in/{city}?format=j1&lang=zh"
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(url)
    if r.status_code != 200:
        return None
    return r.json()


async def main() -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(f"Weather MCP started at: {datetime.now().isoformat()}\n")

    server = Server("weather-mcp")

    @server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="get_weather",
                description="查询城市实时天气：温度、体感温度、湿度、风速风向、能见度、气压、紫外线指数",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "城市名，中文或英文，如：北京、武汉、Tokyo",
                        },
                    },
                    "required": ["city"],
                },
            ),
            types.Tool(
                name="get_forecast",
                description="查询城市未来 3 天天气预报：每天最高/最低温度、天气状况、降水概率",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "城市名，中文或英文",
                        },
                    },
                    "required": ["city"],
                },
            ),
            types.Tool(
                name="get_hourly",
                description="查询城市今天逐小时天气预报：每个小时的温度、天气状况、降水概率、风速",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "城市名，中文或英文",
                        },
                    },
                    "required": ["city"],
                },
            ),
            types.Tool(
                name="get_astro",
                description="查询城市日出、日落、月出、月落时间及月相",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "城市名，中文或英文",
                        },
                    },
                    "required": ["city"],
                },
            ),
        ]

    @server.call_tool()
    async def handle_call_tool(
        name: str, arguments: dict | None
    ) -> list[types.TextContent]:
        try:
            city = (arguments or {}).get("city", "")
            if not city:
                return [types.TextContent(type="text", text="请提供城市名称")]

            data = await _fetch_json(city)
            if data is None:
                return [types.TextContent(type="text", text=f"查询失败，无法连接到天气服务或城市不存在: {city}")]

            area = data.get("nearest_area", [{}])[0]
            loc = area.get("areaName", [{}])[0].get("value", city)
            region = area.get("region", [{}])[0].get("value", "")
            country = area.get("country", [{}])[0].get("value", "")

            if name == "get_weather":
                cc = data.get("current_condition", [{}])[0]
                uv = cc.get("uvIndex", "?")
                try:
                    uv_val = float(uv)
                    uv_desc = "低" if uv_val <= 2 else ("中" if uv_val <= 5 else ("高" if uv_val <= 7 else "很高"))
                except (ValueError, TypeError):
                    uv_desc = "?"
                lines = [
                    f"📍 {loc}, {region}, {country}",
                    f"🌡️  温度: {cc.get('temp_C', '?')}°C (体感 {cc.get('FeelsLikeC', '?')}°C)",
                    f"☁️  天气: {cc.get('weatherDesc', [{}])[0].get('value', '?')}",
                    f"💧 湿度: {cc.get('humidity', '?')}%",
                    f"💨 风速: {cc.get('windspeedKmph', '?')} km/h ({cc.get('winddir16Point', '?')})",
                    f"👁️  能见度: {cc.get('visibility', '?')} km",
                    f"📊 气压: {cc.get('pressure', '?')} mb",
                    f"☀️  紫外线指数: {uv} ({uv_desc})",
                ]
                return [types.TextContent(type="text", text="\n".join(lines))]

            elif name == "get_forecast":
                lines = [f"📅 {loc}, {region}, {country} 未来 3 天预报:", ""]
                for day in data.get("weather", [])[:3]:
                    date = day.get("date", "?")
                    maxt = day.get("maxtempC", "?")
                    mint = day.get("mintempC", "?")
                    h0 = day.get("hourly", [{}])[0]
                    desc = h0.get("weatherDesc", [{}])[0].get("value", "?")
                    rain = h0.get("chanceofrain", "?")
                    sun = h0.get("sunHour", "?")
                    lines.append(f"  {date}: {desc}, 🌡️ {mint}~{maxt}°C, 🌧️ {rain}%, ☀️ {sun}h")
                return [types.TextContent(type="text", text="\n".join(lines))]

            elif name == "get_hourly":
                today = data.get("weather", [{}])[0]
                date = today.get("date", "今天")
                lines = [f"🕐 {loc} {date} 逐小时预报:", ""]
                for h in today.get("hourly", []):
                    time = h.get("time", "0")
                    try:
                        hour = int(time) // 100
                    except ValueError:
                        hour = 0
                    temp = h.get("tempC", "?")
                    desc = h.get("weatherDesc", [{}])[0].get("value", "?")
                    rain = h.get("chanceofrain", "?")
                    wind = h.get("windspeedKmph", "?")
                    lines.append(f"  {hour:02d}:00  {temp}°C  {desc}  降水{rain}%  风速{wind}km/h")
                return [types.TextContent(type="text", text="\n".join(lines))]

            elif name == "get_astro":
                today = data.get("weather", [{}])[0]
                astro = today.get("astronomy", [{}])[0]
                lines = [
                    f"🌙 {loc}, {region}, {country} 天文信息 ({today.get('date', '?')})",
                    f"  🌅 日出: {astro.get('sunrise', '?')}",
                    f"  🌇 日落: {astro.get('sunset', '?')}",
                    f"  🌙 月出: {astro.get('moonrise', '?')}",
                    f"  🌑 月落: {astro.get('moonset', '?')}",
                    f"  🌓 月相: {astro.get('moon_phase', {}).get('value', '?')}",
                ]
                return [types.TextContent(type="text", text="\n".join(lines))]

            return [types.TextContent(type="text", text=f"未知工具: {name}")]

        except Exception as e:
            return [types.TextContent(type="text", text=f"异常: {type(e).__name__}: {e}")]

    async with mcp.server.stdio.stdio_server() as (read, write):
        await server.run(read, write, InitializationOptions(
            server_name="weather-mcp",
            server_version="0.2.0",
            capabilities=server.get_capabilities(
                notification_options=NotificationOptions(),
                experimental_capabilities={},
            ),
        ))


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
