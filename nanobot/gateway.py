"""MyNanobot 网关：前端 + API + WebSocket。"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path

import aiohttp
from aiohttp import web

from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import InboundMessage

logger = logging.getLogger(__name__)

_GATEWAY_TOKEN = "mynanobot-dev-token"


class Gateway:
    """MyNanobot 网关。

    职责：
    - 前端静态文件服务
    - /webui/bootstrap 和 WebSocket
    - 所有 /api/* 和 /webui/* 返回 JSON 存根，避免 SPA 回退吞掉请求
    """

    def __init__(self, agent: AgentLoop, host: str = "0.0.0.0", port: int = 8765):
        self.agent = agent
        self.host = host
        self.port = port
        self.dist_dir = Path(__file__).resolve().parent.parent / "webui" / "dist"
        self._app = web.Application(
            middlewares=[self._cache_middleware],
        )
        self._setup_routes()

    def _setup_routes(self):
        """注册路由。

        顺序很重要：
        1. API 路由先注册（精确匹配）
        2. 静态文件路由
        3. API 兜底路由（/api/* 和 /webui/* 返回 JSON）
        4. SPA 回退路由（/{tail:.*} 返回 index.html）
        """
        # ---- 健康检查和 WebSocket ----
        self._app.router.add_get("/health", self._handle_health)
        self._app.router.add_get("/test", self._handle_test_page)
        self._app.router.add_get("/webui/bootstrap", self._handle_bootstrap)
        self._app.router.add_get("/ws", self._handle_websocket)

        # ---- 前端需要的 API 存根 ----
        self._app.router.add_get("/api/sessions", self._handle_sessions)
        self._app.router.add_get("/api/sessions/{key}/webui-thread", self._handle_webui_thread)
        self._app.router.add_get("/api/sessions/{key}/delete", self._handle_api_stub)
        self._app.router.add_get("/api/sessions/{key}/file-preview", self._handle_api_stub)
        self._app.router.add_get("/api/sessions/{key}/automations", self._handle_api_stub)
        self._app.router.add_get("/api/settings", self._handle_settings)
        self._app.router.add_get("/api/settings/usage", self._handle_api_stub)
        self._app.router.add_get("/api/settings/version-check", self._handle_api_stub)
        self._app.router.add_get("/api/settings/update", self._handle_api_stub)
        self._app.router.add_get("/api/settings/provider/update", self._handle_api_stub)
        self._app.router.add_get("/api/settings/mcp-presets", self._handle_api_stub)
        self._app.router.add_get("/api/settings/cli-apps", self._handle_api_stub)
        self._app.router.add_get("/api/settings/web-search/update", self._handle_api_stub)
        self._app.router.add_get("/api/settings/network-safety/update", self._handle_api_stub)
        self._app.router.add_get("/api/settings/image-generation/update", self._handle_api_stub)
        self._app.router.add_get("/api/settings/transcription/update", self._handle_api_stub)
        self._app.router.add_get("/api/settings/model-configurations/create", self._handle_api_stub)
        self._app.router.add_get("/api/settings/model-configurations/update", self._handle_api_stub)
        self._app.router.add_get("/api/settings/provider-models", self._handle_api_stub)
        self._app.router.add_get("/api/settings/provider/oauth-login", self._handle_api_stub)
        self._app.router.add_get("/api/settings/provider/oauth-logout", self._handle_api_stub)
        self._app.router.add_get("/api/webui/skills", self._handle_skills)
        self._app.router.add_get("/api/webui/skills/{name}", self._handle_api_stub)
        self._app.router.add_get("/api/webui/sidebar-state", self._handle_sidebar_state)
        self._app.router.add_get("/api/webui/sidebar-state/update", self._handle_api_stub)
        self._app.router.add_get("/api/workspaces", self._handle_api_stub)
        self._app.router.add_get("/api/commands", self._handle_commands)
        self._app.router.add_get("/api/commands/palette", self._handle_api_stub)

        # ---- 前端静态文件 ----
        if self.dist_dir.exists():
            self._app.router.add_get("/", self._serve_frontend)
            assets_dir = self.dist_dir / "assets"
            if assets_dir.exists():
                self._app.router.add_static(
                    "/assets/", path=str(assets_dir), name="assets",
                )
            brand_dir = self.dist_dir / "brand"
            if brand_dir.exists():
                self._app.router.add_static(
                    "/brand/", path=str(brand_dir), name="brand",
                )

        # ---- API 兜底：所有未匹配的 /api/* 和 /webui/* 返回 JSON ----
        self._app.router.add_get("/api/{tail:.*}", self._handle_api_stub)
        self._app.router.add_get("/webui/{tail:.*}", self._handle_api_stub)

        # ---- SPA 回退 ----
        if self.dist_dir.exists():
            self._app.router.add_get("/{tail:.*}", self._serve_frontend)

    # ---- 前端静态文件处理器 ----

    @web.middleware
    async def _cache_middleware(self, request: web.Request, handler):
        """全局 middleware：API 加 CORS，静态文件禁止缓存。"""
        resp = await handler(request)
        path = request.path
        if path.startswith("/api") or path.startswith("/webui"):
            resp.headers["Access-Control-Allow-Origin"] = "*"
        if not path.startswith("/api") and not path.startswith("/health") and not path.startswith("/ws"):
            resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return resp

    async def _serve_frontend(self, request: web.Request) -> web.StreamResponse:
        index_path = self.dist_dir / "index.html"
        if not index_path.exists():
            return web.json_response({"error": "frontend not built"}, status=503)
        return web.FileResponse(
            index_path,
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    # ---- API 处理器 ----

    async def _handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def _handle_test_page(self, request: web.Request) -> web.Response:
        """返回一个简单的测试页面。"""
        html = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>MyNanobot Test</title>
<style>body{font-family:system-ui;max-width:600px;margin:40px auto;padding:20px;background:#111;color:#eee}
input,button{font:inherit;padding:8px 12px;border-radius:6px;border:none}
input{flex:1;background:#222;color:#eee}
button{background:#4f46e5;color:white;cursor:pointer}
button:hover{background:#6366f1}
#output{margin-top:20px;white-space:pre-wrap;font-family:monospace}
.pass{color:#22c55e}.fail{color:#ef4444}</style></head>
<body>
<h2>MyNanobot 连通性测试</h2>
<div style="display:flex;gap:8px">
  <input id="msg" value="你好" placeholder="输入消息">
  <button onclick="test()">发送</button>
</div>
<div id="output"></div>
<script>
async function log(tag,ok,detail){
  const d=document.getElementById('output');
  d.innerHTML+=`<span class="${ok?'pass':'fail'}">${ok?'OK':'FAIL'}</span> ${tag}: ${detail}\n`;
}
async function test(){
  document.getElementById('output').innerHTML='';
  try{
    const r1=await fetch('/webui/bootstrap');
    const b=await r1.json();
    log('bootstrap',r1.ok,`token=${b.token?.slice(0,10)}... ws_path=${b.ws_path}`);
    
    const r2=await fetch('/api/settings');
    log('settings',r2.ok,`agent=${(await r2.json()).agent?.model||'?'}`);
    
    const ws=new WebSocket(`ws://${location.host}${b.ws_path}`);
    await new Promise((resolve,reject)=>{
      ws.onopen=()=>{log('websocket',true,'connected');resolve();};
      ws.onerror=()=>{log('websocket',false,'error');reject();};
      setTimeout(()=>{log('websocket',false,'timeout');reject();},5000);
    });
    
    ws.send(JSON.stringify({event:'user_message',text:document.getElementById('msg').value}));
    await new Promise((resolve)=>{
      ws.onmessage=(ev)=>{
        const d=JSON.parse(ev.data);
        if(d.event==='delta_end'){log('reply',true,'received end marker');resolve();}
        else if(d.event==='delta')console.log('stream:',d.text);
      };
      setTimeout(()=>{log('reply',false,'timeout waiting');resolve();},15000);
    });
    ws.close();
    log('done',true,'All tests passed');
  }catch(e){
    log('error',false,e.message);
  }
}
test();
</script></body></html>"""
        return web.Response(text=html, content_type="text/html")

    async def _handle_bootstrap(self, request: web.Request) -> web.Response:
        """返回前端启动所需信息。"""
        return web.json_response({
            "token": _GATEWAY_TOKEN,
            "ws_path": "/ws",
            "ws_url": None,
            "expires_in": 3600,
            "model_name": self.agent.model,
            "runtime_surface": "browser",
            "runtime_capabilities": {},
            "version": "0.1.0",
        })

    async def _handle_sessions(self, request: web.Request) -> web.Response:
        return web.json_response({"sessions": []})

    async def _handle_webui_thread(self, request: web.Request) -> web.Response:
        return web.json_response({"messages": []})

    async def _handle_settings(self, request: web.Request) -> web.Response:
        """返回前端设置页面所需数据结构。"""
        return web.json_response({
            "surface": "browser",
            "runtime_surface": "browser",
            "runtime_capabilities": {},
            "apply_state": {"status": "idle", "sections": []},
            "agent": {
                "model": self.agent.model,
                "provider": type(self.agent.provider).__name__,
                "resolved_provider": None,
                "has_api_key": True,
                "model_preset": None,
                "max_tokens": 8192,
                "context_window_tokens": 65536,
                "temperature": 0.1,
                "reasoning_effort": None,
                "timezone": "Asia/Shanghai",
                "bot_name": "MyNanobot",
                "bot_icon": "\U0001f916",
                "tool_hint_max_length": 40,
            },
            "model_presets": [],
            "providers": [],
            "provider_settings": {},
            "transcription": {"enabled": False, "provider": "", "model": "", "language": "", "max_duration_sec": 120, "max_upload_mb": 25},
            "image_generation": {"enabled": False, "provider": "", "model": "", "default_aspect_ratio": "1:1", "default_image_size": "1024x1024", "max_images_per_turn": 4},
            "web_search": {"provider": "", "api_key": None, "base_url": None, "max_results": 8, "timeout": 0, "use_jina_reader": False},
            "security": {"webui_allow_local_service_access": True, "webui_default_access_mode": "allow"},
            "tools_config": {},
            "usage": {"total_tokens": 0, "total_prompt_tokens": 0, "total_completion_tokens": 0, "daily": []},
            "workspace": {"default": "/home/huhu/.nanobot/workspace"},
            "workspace_scope": None,
        })

    async def _handle_skills(self, request: web.Request) -> web.Response:
        return web.json_response({"skills": []})

    async def _handle_commands(self, request: web.Request) -> web.Response:
        return web.json_response({"commands": []})

    async def _handle_sidebar_state(self, request: web.Request) -> web.Response:
        return web.json_response({})

    async def _handle_api_stub(self, request: web.Request) -> web.Response:
        """未实现的 API 端点返回空 JSON，避免 SPA 回退。"""
        return web.json_response({})

    # ---- WebSocket 处理器 ----

    async def _handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        logger.info("WebSocket 客户端已连接")
        client_id = request.query.get("client_id", f"ws-{uuid.uuid4().hex[:8]}")
        chat_id = f"ws:{client_id}"

        async def _send_responses():
            try:
                while True:
                    outbound = await self.agent.bus.consume_outbound()
                    if outbound.content:
                        await ws.send_str(json.dumps({
                            "event": "delta",
                            "text": outbound.content,
                        }))
                    await ws.send_str(json.dumps({"event": "delta_end"}))
            except Exception:
                pass

        send_task = asyncio.create_task(_send_responses())

        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    event = data.get("event", "")

                    if event == "user_message":
                        content = data.get("text", "")
                        logger.info("收到前端消息: %s", content[:50])
                        inbound = InboundMessage(
                            channel="webui",
                            sender_id=client_id,
                            chat_id=chat_id,
                            content=content,
                        )
                        await self.agent.bus.publish_inbound(inbound)
                    elif event == "abort":
                        logger.info("客户端 %s 取消请求", client_id)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error("WebSocket 错误: %s", ws.exception())
        except Exception as e:
            logger.warning("WebSocket 断开: %s", e)
        finally:
            send_task.cancel()
            try:
                await send_task
            except asyncio.CancelledError:
                pass
            logger.info("WebSocket 客户端已断开: %s", client_id)

        return ws

    # ---- 启动 ----

    async def start(self):
        logger.info("MyNanobot 网关启动于 http://%s:%s", self.host, self.port)
        runner = web.AppRunner(self._app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        return runner

    @classmethod
    async def run(cls, agent: AgentLoop, host: str = "0.0.0.0", port: int = 8765):
        gateway = cls(agent, host=host, port=port)
        await agent.start()
        runner = await gateway.start()
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            pass
        finally:
            await agent.stop()
            await runner.cleanup()
