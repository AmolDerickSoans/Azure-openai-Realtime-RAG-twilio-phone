import os
import json
import base64
import asyncio
import websockets
import logging
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.websockets import WebSocketDisconnect
from twilio.twiml.voice_response import VoiceResponse, Connect
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
PORT = int(os.getenv("PORT", 5050))
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


@app.get("/", response_class=JSONResponse)
async def index_page():
    return {"message": "Application is running!"}


@app.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    response = VoiceResponse()
    response.say("Please wait while we connect your call.")
    response.pause(length=1)
    response.say("You can start talking now!")
    host = request.url.hostname
    connect = Connect()
    connect.stream(url=f"wss://{host}/media-stream")
    response.append(connect)
    return HTMLResponse(content=str(response), media_type="application/xml")


@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    logger.info("WebSocket connection opened.")
    await websocket.accept()

    stream_sid = None
    input_audio_queue = asyncio.Queue()
    audio_queue = asyncio.Queue()
    
    # Constants for audio buffering
    BUFFER_SIZE = 20 * 160  # 0.4 seconds of audio at 8kHz

    async with websockets.connect(
        AZURE_OPENAI_API_ENDPOINT,
        additional_headers={"api-key": AZURE_OPENAI_API_KEY},
    ) as openai_ws:
        await initialize_session(openai_ws)

        async def receive_from_twilio():
            nonlocal stream_sid
            # Buffers for audio processing
            inbuffer = bytearray(b'')
            outbuffer = bytearray(b'')
            inbound_chunks_started = False
            latest_inbound_timestamp = 0
            
            try:
                async for message in websocket.iter_text():
                    data = json.loads(message)
                    if data["event"] == "media":
                        # Decode the audio payload
                        chunk = base64.b64decode(data["media"]["payload"])
                        media_timestamp = int(data["media"].get("timestamp", 0))
                        
                        # Handle silence filling for dropped packets
                        if inbound_chunks_started:
                            if latest_inbound_timestamp + 20 < media_timestamp:
                                bytes_to_fill = 8 * (media_timestamp - (latest_inbound_timestamp + 20))
                                inbuffer.extend(b'\xff' * bytes_to_fill)
                        else:
                            inbound_chunks_started = True
                            latest_inbound_timestamp = media_timestamp
                        
                        latest_inbound_timestamp = media_timestamp
                        inbuffer.extend(chunk)
                        
                        # Process buffered audio
                        while len(inbuffer) >= BUFFER_SIZE:
                            # Store the input audio chunk for processing
                            input_audio_queue.put_nowait(inbuffer[:BUFFER_SIZE])
                            
                            # Format for OpenAI
                            audio_append = {
                                "type": "input_audio_buffer.append",
                                "audio": base64.b64encode(inbuffer[:BUFFER_SIZE]).decode('ascii'),
                            }
                            await openai_ws.send(json.dumps(audio_append))
                            
                            # Clear processed buffer
                            inbuffer = inbuffer[BUFFER_SIZE:]
                    elif data["event"] == "start":
                        # Handle stream_sid extraction with proper error handling
                        try:
                            stream_sid = data.get("streamSid") or data.get("start", {}).get("streamSid")
                            if not stream_sid:
                                logger.warning("Could not find streamSid in the data")
                            else:
                                logger.info(f"Stream started with SID: {stream_sid}")
                        except Exception as e:
                            logger.error(f"Error extracting streamSid: {e}")
            except WebSocketDisconnect:
                logger.warning("WebSocket disconnected by client.")
                if openai_ws.open:
                    await openai_ws.close()

        async def send_to_twilio():
            try:
                async for openai_message in openai_ws:
                    response = json.loads(openai_message)

                    if response.get("type") == "response.audio.delta" and "delta" in response:
                        chunk = base64.b64decode(response["delta"])
                        
                        # Process audio according to Exotel requirements
                        exotel_audio = np.frombuffer(chunk, dtype=np.uint8)
                        exotel_audio_bytes = exotel_audio.tobytes()
                        
                        # Chunk the audio as per Exotel requirements
                        EXOTEL_MIN_CHUNK_SIZE = 3200
                        EXOTEL_MAX_CHUNK_SIZE = 100000
                        EXOTEL_CHUNK_MULTIPLE = 320
                        valid_chunk_size = max(
                            EXOTEL_MIN_CHUNK_SIZE, 
                            min(
                                EXOTEL_MAX_CHUNK_SIZE, 
                                (len(exotel_audio_bytes) // EXOTEL_CHUNK_MULTIPLE) * EXOTEL_CHUNK_MULTIPLE
                            )
                        )
                        
                        chunked_payloads = [
                            exotel_audio_bytes[i:i + valid_chunk_size] 
                            for i in range(0, len(exotel_audio_bytes), valid_chunk_size)
                        ]
                        
                        # Send each chunk with appropriate metadata
                        for chunk in chunked_payloads:
                            audio_payload = base64.b64encode(chunk).decode("ascii")
                            audio_delta = {
                                "event": "media",
                                "stream_sid": stream_sid,
                                "media": {
                                    "payload": audio_payload
                                }
                            }
                            
                            await websocket.send_text(json.dumps(audio_delta))

                    # Handle function calls for RAG
                    if response.get("type") == "response.function_call_arguments.done":
                        function_name = response["name"]
                        if function_name == "get_additional_context":
                            query = json.loads(response["arguments"]).get("query", "")
                            search_results = azure_search_rag(query)
                            logger.info(f"RAG Results: {search_results}")
                            await send_function_output(openai_ws, response["call_id"], search_results)
            except Exception as e:
                logger.error(f"Error in send_to_twilio: {e}")

        await asyncio.gather(receive_from_twilio(), send_to_twilio())


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
                "Keep all your responses very consise and straight to point and not more than 15 words"
                "If USER says Thank You,  Always respond with with You are welcome, Is there anything else I can help you with?"
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


async def trigger_rag_search(openai_ws, query):
    """Trigger RAG search for a specific query."""
    search_function_call = {
        "type": "conversation.item.create",
        "item": {
            "type": "function_call",
            "name": "get_additional_context",
            "arguments": {"query": query},
        },
    }
    await openai_ws.send(json.dumps(search_function_call))


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