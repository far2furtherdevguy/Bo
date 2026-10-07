import asyncio
import json
import os
import logging
from aiohttp import web
import aiohttp_cors
from ai_core import SymbolicAI

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize AI
ai = SymbolicAI()

# Connected clients
clients = set()

async def websocket_handler(request):
    """Handle WebSocket connections"""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    clients.add(ws)
    logger.info(f"Client connected. Total: {len(clients)}")
    
    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    response = await process_message(data)
                    await ws.send_json(response)
                except json.JSONDecodeError:
                    await ws.send_json({
                        "type": "error",
                        "message": "Invalid JSON"
                    })
            elif msg.type == web.WSMsgType.ERROR:
                logger.error(f'WebSocket error: {ws.exception()}')
    finally:
        clients.discard(ws)
        logger.info(f"Client disconnected. Total: {len(clients)}")
    
    return ws

async def process_message(data):
    """Process incoming messages"""
    msg_type = data.get("type", "chat")
    
    if msg_type == "chat":
        query = data.get("message", "")
        result = ai.think(query)
        return {
            "type": "chat",
            "response": result.text,
            "confidence": result.conf,
            "source": result.source,
            "meta": result.meta or {}
        }
    
    elif msg_type == "teach":
        question = data.get("question", "")
        answer = data.get("answer", "")
        result = ai.teach(question, answer)
        return {"type": "teach", "result": result}
    
    elif msg_type == "train_file":
        content = data.get("content", "")
        filename = data.get("filename", "unknown")
        result = ai.train_from_text(content, source=filename)
        return {"type": "train_file", "result": result}
    
    elif msg_type == "generate":
        seed = data.get("seed", "")
        poem = data.get("poem", False)
        result = ai.generate_creative(seed, poem)
        return {"type": "generate", "result": result}
    
    elif msg_type == "analogy":
        a = data.get("a", "")
        b = data.get("b", "")
        c = data.get("c", "")
        result = ai.solve_analogy(a, b, c)
        return {"type": "analogy", "result": result}
    
    elif msg_type == "reason":
        facts = data.get("facts", [])
        result = ai.reason(facts)
        return {
            "type": "reason",
            "result": result,
            "all_facts": ai.get_facts()
        }
    
    elif msg_type == "stats":
        return {"type": "stats", "result": ai.get_stats()}
    
    elif msg_type == "rules":
        return {"type": "rules", "result": ai.get_rules()}
    
    else:
        return {"type": "error", "message": f"Unknown type: {msg_type}"}

# HTTP handlers
async def index_handler(request):
    """Serve the HTML interface"""
    return web.FileResponse('./static/index.html')

async def stats_handler(request):
    """REST API for stats"""
    return web.json_response(ai.get_stats())

async def chat_handler(request):
    """REST API for chat"""
    try:
        data = await request.json()
        query = data.get("message", "")
        result = ai.think(query)
        return web.json_response({
            "response": result.text,
            "confidence": result.conf,
            "source": result.source
        })
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

async def train_handler(request):
    """REST API for training"""
    try:
        data = await request.json()
        
        if "file" in data:
            content = data["file"]
            filename = data.get("filename", "upload")
            result = ai.train_from_text(content, source=filename)
        else:
            question = data.get("question", "")
            answer = data.get("answer", "")
            result = ai.teach(question, answer)
        
        return web.json_response(result)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

async def ws_route(request):
    """WebSocket route handler"""
    return await websocket_handler(request)

# Create app
app = web.Application()

# CORS
cors = aiohttp_cors.setup(app, defaults={
    "*": aiohttp_cors.ResourceOptions(
        allow_credentials=True,
        expose_headers="*",
        allow_headers="*",
        allow_methods="*"
    )
})

# Routes
app.router.add_get('/', index_handler)
app.router.add_get('/stats', stats_handler)
app.router.add_post('/chat', chat_handler)
app.router.add_post('/train', train_handler)
app.router.add_get('/ws', ws_route)  # WebSocket endpoint
app.router.add_static('/static/', path='./static', name='static')

# Apply CORS to all routes
for route in list(app.router.routes()):
    cors.add(route)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    web.run_app(app, host="0.0.0.0", port=port)
