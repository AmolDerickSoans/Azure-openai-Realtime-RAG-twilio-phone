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
    logger.info("Starting new WebSocket connection handler with enhanced logging")
    try:
        logger.info("Attempting to accept WebSocket connection with detailed tracking")
        await websocket.accept()
        logger.info("WebSocket connection accepted successfully, ready for data flow")

        stream_sid = None
        
        # Constants for Exotel audio chunking
        EXOTEL_MIN_CHUNK_SIZE = 3200
        EXOTEL_MAX_CHUNK_SIZE = 100000
        EXOTEL_CHUNK_MULTIPLE = 320

        logger.info(f"Attempting to connect to Azure OpenAI at {AZURE_OPENAI_API_ENDPOINT} with enhanced security")
        async with websockets.connect(
            AZURE_OPENAI_API_ENDPOINT,
            extra_headers={"api-key": AZURE_OPENAI_API_KEY},
        ) as openai_ws:
            logger.info("Successfully established secure connection to Azure OpenAI WebSocket")
            logger.info("Initializing OpenAI session with custom configuration parameters")
            await initialize_session(openai_ws)
            logger.info("OpenAI session initialized successfully with all parameters set")

            async def receive_from_exotel():
                nonlocal stream_sid
                try:
                    logger.info("Starting Exotel receive loop with enhanced monitoring")
                    async for message in websocket.iter_text():
                        data = json.loads(message)
                        logger.info(f"Received Exotel event type: {data['event']} with detailed tracking")
                        
                        if data["event"] == "media":
                            logger.debug("Processing media event from Exotel with payload validation")
                            audio_append = {
                                "type": "input_audio_buffer.append",
                                "audio": data["media"]["payload"],
                            }
                            await openai_ws.send(json.dumps(audio_append))
                            logger.debug(f"Audio data successfully forwarded to OpenAI, size: {len(data['media']['payload'])}")
                            
                        elif data["event"] == "start":
                            stream_sid = data["stream_sid"]
                            logger.info(f"Stream initialized with SID: {stream_sid}, ready for audio processing")
                            
                        elif data["event"] == "stop":
                            logger.info(f"Stream termination received for SID: {stream_sid}, cleaning up resources")
                            
                except WebSocketDisconnect:
                    logger.warning("WebSocket disconnected by Exotel client, initiating cleanup")
                    if openai_ws.open:
                        logger.info("Gracefully closing OpenAI WebSocket connection")
                        await openai_ws.close()
                except Exception as e:
                    logger.error(f"Critical error in receive_from_exotel: {str(e)}")
                    logger.exception("Detailed error traceback for debugging:")

            async def send_to_exotel():
                try:
                    logger.info("Starting OpenAI receive loop with enhanced monitoring")
                    async for openai_message in openai_ws:
                        response = json.loads(openai_message)
                        logger.info(f"Received OpenAI response type: {response.get('type')} with detailed tracking")

                        if response.get("type") == "response.audio.delta" and "delta" in response:
                            logger.debug("Processing audio response from OpenAI for Exotel delivery")
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
                            
                            logger.debug(f"Audio chunked into {len(chunked_payloads)} parts for Exotel streaming")
                            
                            # Send each chunk with appropriate metadata
                            for chunk in chunked_payloads:
                                audio_payload = base64.b64encode(chunk).decode("ascii")
                                audio_delta = {
                                    "event": "media",
                                    "stream_sid": stream_sid,
                                    "media": {"payload": audio_payload}
                                }
                                
                                await websocket.send_json(audio_delta)
                                logger.debug(f"Successfully sent audio chunk to Exotel, size: {len(chunk)} bytes")

                        # Handle function calls for RAG
                        if response.get("type") == "response.function_call_arguments.done":
                            logger.info("Processing RAG function call with enhanced tracking")
                            function_name = response["name"]
                            if function_name == "get_additional_context":
                                query = json.loads(response["arguments"]).get("query", "")
                                logger.info(f"Executing RAG query with parameters: {query}")
                                search_results = azure_search_rag(query)
                                logger.info(f"RAG search completed with results length: {len(search_results)}")
                                await send_function_output(openai_ws, response["call_id"], search_results)
                                logger.debug("Successfully sent RAG results back to OpenAI")

                        # Process committed audio input for RAG
                        if response.get("type") == "input_audio_buffer.committed":
                            query = response.get("text", "").strip()
                            if query:
                                logger.info(f"Received and processed transcribed query: {query}")
                            
                except Exception as e:
                    logger.error(f"Critical error in send_to_exotel function: {str(e)}")
                    logger.exception("Detailed error traceback for debugging:")

            logger.info("Starting WebSocket communication tasks")
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