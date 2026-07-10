"""供配置托管型智能应用使用的标准清单规范格式。

这份清单刻意设计为具备完整描述性。各类安装程序仍各自独立适配，
而该通信协议为 WebUI 以及后续各类注册中心提供一套精简统一的
规范词汇，用于描述功能权限、信任校验以及经过核验的安装/卸载流程方案。
"""

from __future__ import annotations

from typing import Any

APP_PROTOCOL_SCHEMA = "agent-app.v1"


def compact_dict(values: dict[str, Any]) -> dict[str, Any]:
    """移除字典中为空的可选值，同时保留显式设置的布尔值和零值。"""
    return {
        key: value
        for key, value in values.items()
        if value is not None and value != "" and value != [] and value != {}
    }


def app_manifest(
    *,
    app_id: str,
    display_name: str,
    description: str,
    category: str,
    source: str,
    capabilities: list[dict[str, Any]],
    install: dict[str, Any],
    remove: dict[str, Any],
    trust: dict[str, Any],
    version: str | None = None,
    logo_url: str | None = None,
    brand_color: str | None = None,
    docs_url: str | None = None,
) -> dict[str, Any]:
    """构建稳定的应用清单字典。"""
    return compact_dict({
        "schema": APP_PROTOCOL_SCHEMA,
        "id": app_id,
        "display_name": display_name,
        "version": version,
        "description": description,
        "category": category,
        "source": source,
        "logo_url": logo_url,
        "brand_color": brand_color,
        "docs_url": docs_url,
        "capabilities": capabilities,
        "install": install,
        "remove": remove,
        "trust": trust,
    })
