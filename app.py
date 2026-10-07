import json
import logging
import os

from aiohttp import web, WSMsgType

from ai_core import SymbolicAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
INDEX_FILE = os.path.join(STATIC_DIR, "index.html")

ai = SymbolicAI()
clients = set()


def process_message(data):
    """Process one incoming message (dict) and return a dict reply."""
    msg_type = data.get("type", "chat")

    if msg_type == "chat":
        result = ai.think(str(data.get("message", "")))
        return {"type": "chat", "response": result.text, "confidence": result.conf,
                "source": result.source, "meta": result.meta or {}}
    if msg_type == "teach":
        return {"type": "teach",
                "result": ai.teach(str(data.get("question", "")), str(data.get("answer", "")))}
    if msg_type == "train_file":
        return {"type": "train_file",
                "result": ai.train_from_text(str(data.get("content", "")),
                                             source=str(data.get("filename", "unknown")))}
    if msg_type == "generate":
        return {"type": "generate",
                "result": ai.generate_creative(str(data.get("seed", "")), bool(data.get("poem", False)))}
    if msg_type == "analogy":
        return {"type": "analogy",
                "result": ai.solve_analogy(str(data.get("a", "")), str(data.get("b", "")),
                                           str(data.get("c", "")))}
    if msg_type == "reason":
        facts = data.get("facts", [])
        if not isinstance(facts, list):
            facts = [str(facts)]
        return {"type": "reason", "result": ai.reason([str(f) for f in facts]),
                "all_facts": ai.get_facts()}
    if msg_type == "stats":
        return {"type": "stats", "result": ai.get_stats()}
    if msg_type == "rules":
        return {"type": "rules", "result": ai.get_rules()}
    return {"type": "error", "message": f"Unknown type: {msg_type}"}


async def ws_route(request):
    # heartbeat keeps Render's proxy from dropping idle sockets
    ws = web.WebSocketResponse(heartbeat=25, max_msg_size=8 * 1024 * 1024)
    await ws.prepare(request)
    clients.add(ws)
    logger.info("Client connected. Total: %d", len(clients))
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    if not isinstance(data, dict):
                        raise ValueError("expected a JSON object")
                except ValueError:
                    await ws.send_json({"type": "error", "message": "Invalid JSON"})
                    continue
                try:
                    await ws.send_json(process_message(data))
                except Exception as e:  # one bad message must not kill the socket
                    logger.exception("Error processing message")
                    await ws.send_json({"type": "error", "message": str(e)})
            elif msg.type == WSMsgType.ERROR:
                logger.error("WebSocket error: %s", ws.exception())
    finally:
        clients.discard(ws)
        logger.info("Client disconnected. Total: %d", len(clients))
    return ws


async def index_handler(request):
    if not os.path.exists(INDEX_FILE):
        return web.Response(text="static/index.html not found", status=404)
    return web.FileResponse(INDEX_FILE)


async def health_handler(request):
    return web.json_response({"status": "ok"})


async def stats_handler(request):
    return web.json_response(ai.get_stats())


async def read_json(request):
    try:
        data = await request.json()
    except Exception:
        raise web.HTTPBadRequest(text=json.dumps({"error": "Invalid JSON"}),
                                 content_type="application/json")
    if not isinstance(data, dict):
        raise web.HTTPBadRequest(text=json.dumps({"error": "Expected a JSON object"}),
                                 content_type="application/json")
    return data


async def chat_handler(request):
    data = await read_json(request)
    try:
        result = ai.think(str(data.get("message", "")))
        return web.json_response({"response": result.text, "confidence": result.conf,
                                  "source": result.source})
    except Exception as e:
        logger.exception("chat failed")
        return web.json_response({"error": str(e)}, status=500)


async def train_handler(request):
    data = await read_json(request)
    try:
        if "file" in data:
            result = ai.train_from_text(str(data["file"]), source=str(data.get("filename", "upload")))
        else:
            result = ai.teach(str(data.get("question", "")), str(data.get("answer", "")))
        return web.json_response(result)
    except Exception as e:
        logger.exception("train failed")
        return web.json_response({"error": str(e)}, status=500)


# ---- CORS (replaces aiohttp-cors; its per-route loop is fragile with static/ws routes) ----
async def options_handler(request):
    return web.Response(status=204)


async def add_cors_headers(request, response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"


app = web.Application(client_max_size=8 * 1024 * 1024)
app.on_response_prepare.append(add_cors_headers)

app.router.add_get("/", index_handler)
app.router.add_get("/health", health_handler)
app.router.add_get("/stats", stats_handler)
app.router.add_post("/chat", chat_handler)
app.router.add_post("/train", train_handler)
app.router.add_get("/ws", ws_route)
if os.path.isdir(STATIC_DIR):
    app.router.add_static("/static/", path=STATIC_DIR, name="static")
app.router.add_route("OPTIONS", "/{tail:.*}", options_handler)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info("Starting server on port %d", port)
    web.run_app(app, host="0.0.0.0", port=port)
