import os
import json
import base64
import asyncio
import logging
import websockets
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.websockets import WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from dotenv import load_dotenv

load_dotenv()

# Configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_ENDPOINT = os.getenv("AZURE_OPENAI_API_ENDPOINT")
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX")
AZURE_SEARCH_SEMANTIC_CONFIGURATION = os.getenv("AZURE_SEARCH_SEMANTIC_CONFIGURATION")
PORT = int(os.getenv("PORT", 8000))
VOICE = "alloy"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Azure Search Client Setup
credential = AzureKeyCredential(AZURE_SEARCH_KEY)
search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT, index_name=AZURE_SEARCH_INDEX, credential=credential
)

# FastAPI App
app = FastAPI()

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=JSONResponse)
async def index_page():
    return {"message": "Exotel AI Voice Agent is running!"}

@app.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    """Handle Exotel incoming call webhook"""
    # Extract Exotel params if needed
    params = await request.form()
    caller_id = params.get("CallFrom", "Unknown")
    call_sid = params.get("CallSid", "Unknown")
    
    logger.info(f"Incoming call from {caller_id} with SID {call_sid}")
    
    # Get the host with port number for WebSocket URL
    host = request.url.hostname
    port = request.url.port or PORT
    ws_url = f"wss://{host}:{port}/media-stream"
    logger.info(f"Configuring WebSocket URL: {ws_url}")
    
    # Return Exotel AppML response
    exoml_response = {
        "appml": {
            "version": "1.0",
            "call": {
                "actions": [
                    {
                        "say": "Please wait while we connect your call."
                    },
                    {
                        "wait": {"time": 1}
                    },
                    {
                        "say": "You can start talking now!"
                    },
                    {
                        "connect": {
                            "websocket": {
                                "url": ws_url,
                                "content-type": "audio/mulaw;rate=8000"
                            }
                        }
                    }
                ]
            }
        }
    }
    
    return JSONResponse(content=exoml_response)

@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    logger.info("WebSocket connection opened.")
    try:
        await websocket.accept()
        logger.info("WebSocket connection accepted successfully.")

        stream_sid = None
        
        # Constants for Exotel audio chunking
        EXOTEL_MIN_CHUNK_SIZE = 3200
        EXOTEL_MAX_CHUNK_SIZE = 100000
        EXOTEL_CHUNK_MULTIPLE = 320

        logger.info(f"Attempting to connect to Azure OpenAI at {AZURE_OPENAI_API_ENDPOINT}")
        async with websockets.connect(
            AZURE_OPENAI_API_ENDPOINT,
            extra_headers={"api-key": AZURE_OPENAI_API_KEY},
        ) as openai_ws:
            logger.info("Successfully connected to Azure OpenAI WebSocket")
            await initialize_session(openai_ws)
            logger.info("OpenAI session initialized successfully")

            async def receive_from_exotel():
                nonlocal stream_sid
                try:
                    async for message in websocket.iter_text():
                        data = json.loads(message)
                        logger.debug(f"Received Exotel event: {data['event']}")
                        
                        if data["event"] == "media":
                            audio_append = {
                                "type": "input_audio_buffer.append",
                                "audio": data["media"]["payload"],
                            }
                            await openai_ws.send(json.dumps(audio_append))
                            
                        elif data["event"] == "start":
                            stream_sid = data["stream_sid"]
                            logger.info(f"Stream started with SID: {stream_sid}")
                            
                        elif data["event"] == "stop":
                            logger.info(f"Stream stopped with SID: {stream_sid}")
                            
                except WebSocketDisconnect:
                    logger.warning("WebSocket disconnected by client.")
                    if openai_ws.open:
                        await openai_ws.close()
                except Exception as e:
                    logger.error(f"Error in receive_from_exotel: {str(e)}")
                    logger.exception(e)

            async def send_to_exotel():
                try:
                    async for openai_message in openai_ws:
                        response = json.loads(openai_message)
                        logger.debug(f"Received OpenAI response type: {response.get('type')}")

                        if response.get("type") == "response.audio.delta" and "delta" in response:
                            raw_audio = base64.b64decode(response["delta"])
                            
                            # Chunk audio according to Exotel requirements
                            valid_chunk_size = max(
                                EXOTEL_MIN_CHUNK_SIZE, 
                                min(
                                    EXOTEL_MAX_CHUNK_SIZE, 
                                    (len(raw_audio) // EXOTEL_CHUNK_MULTIPLE) * EXOTEL_CHUNK_MULTIPLE
                                )
                            )
                            
                            chunked_payloads = [
                                raw_audio[i:i + valid_chunk_size] 
                                for i in range(0, len(raw_audio), valid_chunk_size)
                            ]
                            
                            # Send each chunk with appropriate metadata
                            for chunk in chunked_payloads:
                                audio_payload = base64.b64encode(chunk).decode("ascii")
                                audio_delta = {
                                    "event": "media",
                                    "stream_sid": stream_sid,
                                    "media": {"payload": audio_payload}
                                }
                                
                                await websocket.send_json(audio_delta)
                                logger.debug(f"Sent audio chunk to Exotel, size: {len(chunk)}")

                        # Handle function calls for RAG
                        if response.get("type") == "response.function_call_arguments.done":
                            function_name = response["name"]
                            if function_name == "get_additional_context":
                                query = json.loads(response["arguments"]).get("query", "")
                                search_results = azure_search_rag(query)
                                logger.info(f"RAG Results: {search_results}")
                                await send_function_output(openai_ws, response["call_id"], search_results)

                        # Process committed audio input for RAG
                        if response.get("type") == "input_audio_buffer.committed":
                            query = response.get("text", "").strip()
                            if query:
                                logger.info(f"Received query: {query}")
                            
                except Exception as e:
                    logger.error(f"Error in send_to_exotel: {str(e)}")
                    logger.exception(e)

            await asyncio.gather(receive_from_exotel(), send_to_exotel())
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {str(e)}")
        logger.exception(e)
        if not websocket.client_state.is_connected:
            logger.error("WebSocket connection failed to establish")
        raise


async def initialize_session(openai_ws):
    """Initialize the OpenAI session with instructions and tools."""
    session_update = {
        "type": "session.update",
        "session": {
            "turn_detection": {"type": "server_vad"},
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
            "voice": VOICE,
            "instructions": (
                "You are an AI assistant providing factual answers ONLY from the search. "
                "If USER says hello Always respond with with Hello, I am Rose from Insurance Company. How can I help you today? "
                "Use the `get_additional_context` function to retrieve relevant information."
                "Keep all your responses very concise and straight to point and not more than 15 words"
                "If USER says Thank You, Always respond with with You are welcome, Is there anything else I can help you with?"
            ),
            "tools": [
                {
                    "type": "function",
                    "name": "get_additional_context",
                    "description": "Fetch context from Azure Search based on a user query.",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                }
            ],
        },
    }
    await openai_ws.send(json.dumps(session_update))


async def send_function_output(openai_ws, call_id, output):
    """Send RAG results back to OpenAI."""
    response = {
        "type": "conversation.item.create",
        "item": {
            "type": "function_call_output",
            "call_id": call_id,
            "output": output,
        },
    }
    await openai_ws.send(json.dumps(response))

    # Prompt OpenAI to continue processing
    await openai_ws.send(json.dumps({"type": "response.create"}))


def azure_search_rag(query):
    """Perform Azure Cognitive Search and return results."""
    try:
        logger.info(f"Querying Azure Search with: {query}")
        results = search_client.search(
            search_text=query,
            top=2,
            query_type="semantic",
            semantic_configuration_name=AZURE_SEARCH_SEMANTIC_CONFIGURATION,
        )
        summarized_results = [doc.get("chunk", "No content available") for doc in results]
        if not summarized_results:
            return "No relevant information found in Azure Search."
        return "\n".join(summarized_results)
    except Exception as e:
        logger.error(f"Error in Azure Search: {e}")
        return "Error retrieving data from Azure Search."


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)