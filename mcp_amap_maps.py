"""高德地图 MCP Server — Python 版

提供地理编码、逆地理编码、IP定位、天气查询、路径规划、POI搜索等功能。
环境变量 AMAP_MAPS_API_KEY 必填。
"""

import os
import sys
import json
from urllib.parse import urlencode
from urllib.request import urlopen

from mcp.server import Server
from mcp.server.stdio import StdioServerTransport
from mcp.types import (
    GetToolsResult,
    CallToolResult,
    TextContent,
    Tool,
)


def get_api_key() -> str:
    api_key = os.environ.get("AMAP_MAPS_API_KEY")
    if not api_key:
        print("AMAP_MAPS_API_KEY 环境变量未设置", file=sys.stderr)
        sys.exit(1)
    return api_key


API_KEY = get_api_key()
BASE_PARAMS = {"key": API_KEY, "source": "ts_mcp"}


def _build_url(base: str, params: dict) -> str:
    """拼接 URL + 公共参数。"""
    merged = {**BASE_PARAMS, **params}
    return f"{base}?{urlencode(merged)}"


def _get_json(url: str) -> dict:
    """GET 请求并解析 JSON。"""
    with urlopen(url, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ── 工具定义 ──────────────────────────────────────────────────────────────

TOOLS = [
    Tool(
        name="maps_regeocode",
        description="将经纬度坐标转换为行政区划地址信息",
        inputSchema={
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "经纬度，格式：经度,纬度"}
            },
            "required": ["location"],
        },
    ),
    Tool(
        name="maps_geo",
        description="将结构化地址转换为经纬度坐标，支持地标名胜、建筑物名称",
        inputSchema={
            "type": "object",
            "properties": {
                "address": {"type": "string", "description": "地址信息"},
                "city": {"type": "string", "description": "指定查询城市（可选）"},
            },
            "required": ["address"],
        },
    ),
    Tool(
        name="maps_ip_location",
        description="根据 IP 地址定位所在位置",
        inputSchema={
            "type": "object",
            "properties": {
                "ip": {"type": "string", "description": "IP 地址"}
            },
            "required": ["ip"],
        },
    ),
    Tool(
        name="maps_weather",
        description="查询指定城市的天气",
        inputSchema={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称或 adcode"}
            },
            "required": ["city"],
        },
    ),
    Tool(
        name="maps_bicycling",
        description="骑行路径规划，最大支持 500km",
        inputSchema={
            "type": "object",
            "properties": {
                "origin": {"type": "string", "description": "出发点 经度,纬度"},
                "destination": {"type": "string", "description": "目的地 经度,纬度"},
            },
            "required": ["origin", "destination"],
        },
    ),
    Tool(
        name="maps_direction_walking",
        description="步行路径规划，100km 以内",
        inputSchema={
            "type": "object",
            "properties": {
                "origin": {"type": "string", "description": "出发点 经度,纬度"},
                "destination": {"type": "string", "description": "目的地 经度,纬度"},
            },
            "required": ["origin", "destination"],
        },
    ),
    Tool(
        name="maps_direction_driving",
        description="驾车路径规划",
        inputSchema={
            "type": "object",
            "properties": {
                "origin": {"type": "string", "description": "出发点 经度,纬度"},
                "destination": {"type": "string", "description": "目的地 经度,纬度"},
            },
            "required": ["origin", "destination"],
        },
    ),
    Tool(
        name="maps_direction_transit_integrated",
        description="公交路径规划（含火车、公交、地铁），跨城必传城市",
        inputSchema={
            "type": "object",
            "properties": {
                "origin": {"type": "string", "description": "出发点 经度,纬度"},
                "destination": {"type": "string", "description": "目的地 经度,纬度"},
                "city": {"type": "string", "description": "起点城市"},
                "cityd": {"type": "string", "description": "终点城市"},
            },
            "required": ["origin", "destination", "city", "cityd"],
        },
    ),
    Tool(
        name="maps_distance",
        description="测量两个经纬度坐标之间的距离",
        inputSchema={
            "type": "object",
            "properties": {
                "origins": {"type": "string", "description": "起点 经度,纬度（多个用分号分隔）"},
                "destination": {"type": "string", "description": "终点 经度,纬度"},
                "type": {"type": "string", "description": "1=驾车 0=直线 3=步行"},
            },
            "required": ["origins", "destination"],
        },
    ),
    Tool(
        name="maps_text_search",
        description="关键词搜索 POI",
        inputSchema={
            "type": "object",
            "properties": {
                "keywords": {"type": "string", "description": "搜索关键词"},
                "city": {"type": "string", "description": "查询城市（可选）"},
                "types": {"type": "string", "description": "POI 类型（可选）"},
            },
            "required": ["keywords"],
        },
    ),
    Tool(
        name="maps_around_search",
        description="周边搜索，根据坐标和半径搜索 POI",
        inputSchema={
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "中心点 经度,纬度"},
                "radius": {"type": "string", "description": "搜索半径（米）"},
                "keywords": {"type": "string", "description": "搜索关键词（可选）"},
            },
            "required": ["location"],
        },
    ),
    Tool(
        name="maps_search_detail",
        description="查询 POI 详细信息",
        inputSchema={
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "POI ID"}
            },
            "required": ["id"],
        },
    ),
]


# ── 工具处理函数 ──────────────────────────────────────────────────────────

async def handle_regeocode(location: str) -> CallToolResult:
    data = _get_json(_build_url("https://restapi.amap.com/v3/geocode/regeo", {"location": location}))
    if data.get("status") != "1":
        return _error(f"逆地理编码失败: {data.get('info', data.get('infocode', ''))}")
    comp = data["regeocode"]["addressComponent"]
    return _ok({"province": comp["province"], "city": comp["city"], "district": comp["district"]})


async def handle_geo(address: str, city: str = "") -> CallToolResult:
    params = {"address": address}
    if city:
        params["city"] = city
    data = _get_json(_build_url("https://restapi.amap.com/v3/geocode/geo", params))
    if data.get("status") != "1":
        return _error(f"地理编码失败: {data.get('info', data.get('infocode', ''))}")
    geocodes = data.get("geocodes", [])
    result = [
        {"location": g["location"], "level": g.get("level", ""),
         "province": g.get("province", ""), "city": g.get("city", ""),
         "district": g.get("district", ""), "adcode": g.get("adcode", "")}
        for g in geocodes
    ]
    return _ok({"results": result})


async def handle_ip_location(ip: str) -> CallToolResult:
    data = _get_json(_build_url("https://restapi.amap.com/v3/ip", {"ip": ip}))
    if data.get("status") != "1":
        return _error(f"IP 定位失败: {data.get('info', data.get('infocode', ''))}")
    return _ok({"province": data["province"], "city": data["city"],
                "adcode": data.get("adcode", ""), "rectangle": data.get("rectangle", "")})


async def handle_weather(city: str) -> CallToolResult:
    data = _get_json(_build_url("https://restapi.amap.com/v3/weather/weatherInfo",
                                 {"city": city, "extensions": "all"}))
    if data.get("status") != "1":
        return _error(f"天气查询失败: {data.get('info', data.get('infocode', ''))}")
    fc = data["forecasts"][0]
    return _ok({"city": fc["city"], "forecasts": fc["casts"]})


async def handle_bicycling(origin: str, destination: str) -> CallToolResult:
    data = _get_json(_build_url("https://restapi.amap.com/v4/direction/bicycling",
                                 {"origin": origin, "destination": destination}))
    if data.get("errcode") != 0:
        return _error(f"骑行规划失败: {data.get('info', data.get('infocode', ''))}")
    return _ok(_simplify_paths(data["data"]))


async def handle_walking(origin: str, destination: str) -> CallToolResult:
    data = _get_json(_build_url("https://restapi.amap.com/v3/direction/walking",
                                 {"origin": origin, "destination": destination}))
    if data.get("status") != "1":
        return _error(f"步行规划失败: {data.get('info', data.get('infocode', ''))}")
    return _ok({"route": _simplify_route(data["route"])})


async def handle_driving(origin: str, destination: str) -> CallToolResult:
    data = _get_json(_build_url("https://restapi.amap.com/v3/direction/driving",
                                 {"origin": origin, "destination": destination}))
    if data.get("status") != "1":
        return _error(f"驾车规划失败: {data.get('info', data.get('infocode', ''))}")
    return _ok({"route": _simplify_route(data["route"])})


async def handle_transit_integrated(origin: str, destination: str,
                                     city: str, cityd: str) -> CallToolResult:
    data = _get_json(_build_url("https://restapi.amap.com/v3/direction/transit/integrated",
                                 {"origin": origin, "destination": destination,
                                  "city": city, "cityd": cityd}))
    if data.get("status") != "1":
        return _error(f"公交规划失败: {data.get('info', data.get('infocode', ''))}")
    return _ok({"route": {"origin": data["route"]["origin"], "destination": data["route"]["destination"],
                          "distance": data["route"].get("distance", ""),
                          "transits": data["route"].get("transits", [])}})


async def handle_distance(origins: str, destination: str, type_: str = "1") -> CallToolResult:
    data = _get_json(_build_url("https://restapi.amap.com/v3/distance",
                                 {"origins": origins, "destination": destination, "type": type_}))
    if data.get("status") != "1":
        return _error(f"距离测量失败: {data.get('info', data.get('infocode', ''))}")
    return _ok({"results": [{"origin_id": r["origin_id"], "dest_id": r["dest_id"],
                              "distance": r["distance"], "duration": r["duration"]}
                             for r in data.get("results", [])]})


async def handle_text_search(keywords: str, city: str = "", types: str = "") -> CallToolResult:
    params = {"keywords": keywords, "city": city, "citylimit": "false"}
    if types:
        params["types"] = types
    data = _get_json(_build_url("https://restapi.amap.com/v3/place/text", params))
    if data.get("status") != "1":
        return _error(f"关键词搜索失败: {data.get('info', data.get('infocode', ''))}")
    return _ok({
        "suggestion": {"keywords": data.get("suggestion", {}).get("keywords", ""),
                       "cities": data.get("suggestion", {}).get("cities", [])},
        "pois": [{"id": p["id"], "name": p["name"], "address": p.get("address", ""),
                  "typecode": p.get("typecode", "")} for p in data.get("pois", [])],
    })


async def handle_around_search(location: str, radius: str = "1000",
                                keywords: str = "") -> CallToolResult:
    params = {"location": location, "radius": radius}
    if keywords:
        params["keywords"] = keywords
    data = _get_json(_build_url("https://restapi.amap.com/v3/place/around", params))
    if data.get("status") != "1":
        return _error(f"周边搜索失败: {data.get('info', data.get('infocode', ''))}")
    return _ok({
        "pois": [{"id": p["id"], "name": p["name"], "address": p.get("address", ""),
                  "typecode": p.get("typecode", "")} for p in data.get("pois", [])],
    })


async def handle_search_detail(id: str) -> CallToolResult:
    data = _get_json(_build_url("https://restapi.amap.com/v3/place/detail", {"id": id}))
    if data.get("status") != "1":
        return _error(f"POI 详情查询失败: {data.get('info', data.get('infocode', ''))}")
    poi = data["pois"][0]
    return _ok({
        "id": poi["id"], "name": poi["name"], "location": poi.get("location", ""),
        "address": poi.get("address", ""), "city": poi.get("cityname", ""),
        "type": poi.get("type", ""),
    })


# ── 辅助函数 ──────────────────────────────────────────────────────────────

def _ok(data: dict) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=json.dumps(data, ensure_ascii=False, indent=2))])


def _error(msg: str) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=msg)],
        isError=True,
    )


def _simplify_paths(data: dict) -> dict:
    return {
        "origin": data.get("origin", ""),
        "destination": data.get("destination", ""),
        "paths": [
            {"distance": p["distance"], "duration": p["duration"],
             "steps": [{"instruction": s["instruction"], "road": s.get("road", ""),
                        "distance": s["distance"], "duration": s.get("duration", "")}
                       for s in p.get("steps", [])]}
            for p in data.get("paths", [])
        ],
    }


def _simplify_route(route: dict) -> dict:
    return {
        "origin": route.get("origin", ""),
        "destination": route.get("destination", ""),
        "paths": [
            {"distance": p["distance"], "duration": p["duration"],
             "steps": [{"instruction": s["instruction"], "road": s.get("road", ""),
                        "distance": s["distance"], "duration": s.get("duration", "")}
                       for s in p.get("steps", [])]}
            for p in route.get("paths", [])
        ],
    }


# ── Server 启动 ───────────────────────────────────────────────────────────

_HANDLERS = {
    "maps_regeocode": handle_regeocode,
    "maps_geo": handle_geo,
    "maps_ip_location": handle_ip_location,
    "maps_weather": handle_weather,
    "maps_bicycling": handle_bicycling,
    "maps_direction_walking": handle_walking,
    "maps_direction_driving": handle_driving,
    "maps_direction_transit_integrated": handle_transit_integrated,
    "maps_distance": handle_distance,
    "maps_text_search": handle_text_search,
    "maps_around_search": handle_around_search,
    "maps_search_detail": handle_search_detail,
}


async def main():
    server = Server("mcp-server/amap-maps")

    async def list_tools() -> GetToolsResult:
        return GetToolsResult(tools=TOOLS)

    async def call_tool(name: str, arguments: dict) -> CallToolResult:
        handler = _HANDLERS.get(name)
        if not handler:
            return _error(f"未知工具: {name}")
        try:
            # 把 arguments 中的 type 转换为 type_（Python 关键字兼容）
            if "type" in arguments and name == "maps_distance":
                arguments = {**arguments, "type_": arguments.pop("type")}
            return await handler(**arguments)
        except Exception as e:
            return _error(f"执行失败: {e}")

    server.set_tools(list_tools, call_tool)

    transport = StdioServerTransport()
    await server.connect(transport)
    print("高德地图 MCP Server 已在 stdio 上运行", file=sys.stderr)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
