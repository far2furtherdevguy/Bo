import asyncio
import websockets
import json
import os
from aiohttp import web
import aiohttp_cors
from ai_core import SymbolicAI
import base64

# Initialize AI
ai = SymbolicAI()

# Connected clients
clients = set()

async def websocket_handler(websocket, path):
    """Handle WebSocket connections"""
    clients.add(websocket)
    print(f"Client connected. Total: {len(clients)}")
    
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                response = await process_message(data)
                await websocket.send(json.dumps(response))
            except json.JSONDecodeError:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON"
                }))
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        clients.remove(websocket)
        print(f"Client disconnected. Total: {len(clients)}")

async def process_message(data: Dict) -> Dict:
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
            "meta": result.meta
        }
    
    elif msg_type == "teach":
        question = data.get("question", "")
        answer = data.get("answer", "")
        result = ai.teach(question, answer)
        return {
            "type": "teach",
            "result": result
        }
    
    elif msg_type == "train_file":
        content = data.get("content", "")
        filename = data.get("filename", "unknown")
        result = ai.train_from_text(content, source=filename)
        return {
            "type": "train_file",
            "result": result
        }
    
    elif msg_type == "generate":
        seed = data.get("seed", "")
        poem = data.get("poem", False)
        result = ai.generate_creative(seed, poem)
        return {
            "type": "generate",
            "result": result
        }
    
    elif msg_type == "analogy":
        a = data.get("a", "")
        b = data.get("b", "")
        c = data.get("c", "")
        result = ai.solve_analogy(a, b, c)
        return {
            "type": "analogy",
            "result": result
        }
    
    elif msg_type == "reason":
        facts = data.get("facts", [])
        result = ai.reason(facts)
        return {
            "type": "reason",
            "result": result,
            "all_facts": ai.get_facts()
        }
    
    elif msg_type == "stats":
        return {
            "type": "stats",
            "result": ai.get_stats()
        }
    
    elif msg_type == "rules":
        return {
            "type": "rules",
            "result": ai.get_rules()
        }
    
    else:
        return {
            "type": "error",
            "message": f"Unknown type: {msg_type}"
        }

# HTTP handlers
async def index_handler(request):
    """Serve the HTML interface"""
    return web.FileResponse('./static/index.html')

async def stats_handler(request):
    """REST API for stats"""
    return web.json_response(ai.get_stats())

async def chat_handler(request):
    """REST API for chat"""
    data = await request.json()
    query = data.get("message", "")
    result = ai.think(query)
    return web.json_response({
        "response": result.text,
        "confidence": result.conf,
        "source": result.source
    })

async def train_handler(request):
    """REST API for training"""
    data = await request.json()
    
    if "file" in data:
        # File upload
        content = data["file"]
        filename = data.get("filename", "upload")
        result = ai.train_from_text(content, source=filename)
    else:
        # Q&A training
        question = data.get("question", "")
        answer = data.get("answer", "")
        result = ai.teach(question, answer)
    
    return web.json_response(result)

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
app.router.add_static('/static/', path='./static', name='static')

# Apply CORS
for route in list(app.router.routes()):
    cors.add(route)

# Start servers
async def main():
    # Start WebSocket server
    ws_server = await websockets.serve(websocket_handler, "0.0.0.0", 8765)
    print("WebSocket server started on ws://localhost:8765")
    
    # Start HTTP server
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", int(os.environ.get("PORT", 8000)))
    await site.start()
    print(f"HTTP server started on http://localhost:{os.environ.get('PORT', 8000)}")
    
    # Keep running
    await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
