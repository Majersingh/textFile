"""
Slack Personal Assistant - Reply on Your Behalf
Uses User Token (xoxp-) to act as YOU
Monitors your DMs and replies in your style using AI
"""

import os
import hmac
import hashlib
import time
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, Request, HTTPException, Header, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import httpx
from dotenv import load_dotenv
from collections import deque

# LangChain for AI responses
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Application configuration"""
    # USER Token (xoxp-...) - acts as YOU
    SLACK_USER_TOKEN = os.getenv("SLACK_USER_TOKEN")  # xoxp-...
    SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET")
    
    # Your Slack User ID (to identify your messages)
    MY_USER_ID = os.getenv("MY_USER_ID")  # U1234567890
    
    # OpenAI for response generation
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    PORT = int(os.getenv("PORT", 8000))
    HOST = os.getenv("HOST", "0.0.0.0")
    
    # Analysis settings
    ANALYZE_MESSAGE_COUNT = 50  # Number of your past messages to analyze


# ============================================================================
# SLACK API CLIENT
# ============================================================================

class SlackUserClient:
    """Client for making requests using User Token (acts as you)"""
    
    def __init__(self, token: str):
        self.token = token
        self.base_url = "https://slack.com/api"
    
    async def get_conversation_history(
        self,
        channel: str,
        limit: int = 100,
        oldest: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get conversation history from a channel/DM"""
        url = f"{self.base_url}/conversations.history"
        
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        
        params = {
            "channel": channel,
            "limit": limit
        }
        if oldest:
            params["oldest"] = oldest
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, params=params)
            return response.json()
    
    async def post_message_as_user(
        self,
        channel: str,
        text: str,
        thread_ts: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Post a message AS YOU (not as bot)
        Uses chat.postMessage with user token
        """
        url = f"{self.base_url}/chat.postMessage"
        
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "channel": channel,
            "text": text,
            "as_user": True  # Important: post as authenticated user
        }
        
        if thread_ts:
            payload["thread_ts"] = thread_ts
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers)
            return response.json()
    
    async def get_user_info(self, user_id: str) -> Dict[str, Any]:
        """Get user information"""
        url = f"{self.base_url}/users.info"
        
        headers = {
            "Authorization": f"Bearer {self.token}"
        }
        
        params = {"user": user_id}
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, headers=headers)
            return response.json()
    
    async def get_conversations_list(
        self,
        types: str = "im",
        limit: int = 100
    ) -> Dict[str, Any]:
        """Get list of conversations (DMs, channels, etc.)"""
        url = f"{self.base_url}/conversations.list"
        
        headers = {
            "Authorization": f"Bearer {self.token}"
        }
        
        params = {
            "types": types,
            "limit": limit
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, params=params)
            return response.json()


# Initialize Slack client
slack_client = SlackUserClient(Config.SLACK_USER_TOKEN)

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    api_key=Config.OPENAI_API_KEY
)

# ============================================================================
# MESSAGE STYLE ANALYZER
# ============================================================================

class MessageStyleAnalyzer:
    """Analyzes your message history to learn your communication style"""
    
    def __init__(self):
        self.style_profile = None
        self.my_messages_cache = deque(maxlen=100)
    
    async def analyze_my_style(self, channel: str, my_user_id: str) -> str:
        """
        Analyze your past messages to understand your communication style
        """
        print(f"🔍 Analyzing your message style in channel {channel}...")
        
        # Get conversation history
        history = await slack_client.get_conversation_history(
            channel=channel,
            limit=Config.ANALYZE_MESSAGE_COUNT * 2  # Get more to filter yours
        )
        
        if not history.get("ok"):
            print(f"❌ Failed to get history: {history.get('error')}")
            return "casual and friendly"
        
        # Filter only YOUR messages
        my_messages = [
            msg["text"] for msg in history.get("messages", [])
            if msg.get("user") == my_user_id and msg.get("text") and not msg.get("bot_id")
        ][:Config.ANALYZE_MESSAGE_COUNT]
        
        print(f"✅ Found {len(my_messages)} of your messages")
        
        if not my_messages:
            return "casual and friendly"
        
        # Cache for future use
        self.my_messages_cache.extend(my_messages)
        
        # Use AI to analyze style
        analysis_prompt = ChatPromptTemplate.from_template(
            """You are analyzing someone's Slack message writing style.

Here are their recent messages:
{messages}

Analyze their communication style and describe it in 2-3 sentences covering:
1. Tone (formal/casual/friendly/direct)
2. Message length preference (brief/detailed)
3. Use of emojis, punctuation, greetings
4. Any unique patterns or phrases they use

Be specific and concise."""
        )
        
        chain = analysis_prompt | llm | StrOutputParser()
        
        messages_text = "\n---\n".join(my_messages[:20])  # Use first 20 for analysis
        
        try:
            style_description = await chain.ainvoke({"messages": messages_text})
            print(f"✅ Style analysis: {style_description[:100]}...")
            self.style_profile = style_description
            return style_description
        except Exception as e:
            print(f"❌ Style analysis failed: {e}")
            return "casual and friendly"
    
    def get_style_profile(self) -> str:
        """Get cached style profile"""
        return self.style_profile or "casual and friendly"


# Initialize analyzer
style_analyzer = MessageStyleAnalyzer()

# ============================================================================
# AI RESPONSE GENERATOR
# ============================================================================

async def generate_response_in_my_style(
    incoming_message: str,
    sender_name: str,
    channel: str,
    my_user_id: str,
    conversation_context: List[str] = None
) -> str:
    """
    Generate a response that sounds like YOU
    """
    
    # Analyze your style if not done yet
    if not style_analyzer.style_profile:
        await style_analyzer.analyze_my_style(channel, my_user_id)
    
    style_profile = style_analyzer.get_style_profile()
    
    # Get recent conversation context
    context_text = ""
    if conversation_context:
        context_text = "\n".join([f"- {msg}" for msg in conversation_context[-5:]])
    
    # Build example messages from your history
    examples_text = ""
    if style_analyzer.my_messages_cache:
        examples = list(style_analyzer.my_messages_cache)[:5]
        examples_text = "\n".join([f"Example {i+1}: {msg}" for i, msg in enumerate(examples)])
    
    # Generate response
    response_prompt = ChatPromptTemplate.from_template(
        """You are replying to a Slack message on behalf of a user. You must match their exact communication style.

YOUR COMMUNICATION STYLE:
{style_profile}

EXAMPLES OF HOW YOU WRITE:
{examples}

RECENT CONVERSATION CONTEXT:
{context}

INCOMING MESSAGE FROM {sender}:
"{message}"

Generate a response that:
1. Matches the writing style exactly (tone, length, emoji usage, punctuation)
2. Is natural and contextually appropriate
3. Sounds like YOU, not a bot
4. Is helpful and continues the conversation naturally
5. Keep it concise (1-3 sentences unless the topic requires more)

Your response:"""
    )
    
    chain = response_prompt | llm | StrOutputParser()
    
    try:
        response = await chain.ainvoke({
            "style_profile": style_profile,
            "examples": examples_text or "No examples available yet",
            "context": context_text or "No prior context",
            "sender": sender_name,
            "message": incoming_message
        })
        
        print(f"✅ Generated response: {response[:100]}...")
        return response.strip()
        
    except Exception as e:
        print(f"❌ Response generation failed: {e}")
        return "Thanks for the message! I'll get back to you soon."


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Slack Personal Assistant",
    description="Monitors your DMs and replies as YOU",
    version="1.0.0"
)

processed_events = set()


def verify_slack_signature(
    signing_secret: str,
    timestamp: str,
    body: bytes,
    slack_signature: str
) -> bool:
    """Verify request is from Slack"""
    if abs(time.time() - int(timestamp)) > 60 * 5:
        return False
    
    sig_basestring = f"v0:{timestamp}:{body.decode('utf-8')}"
    my_signature = 'v0=' + hmac.new(
        signing_secret.encode(),
        sig_basestring.encode(),
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(my_signature, slack_signature)


# ============================================================================
# ROUTES
# ============================================================================

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "online",
        "service": "Slack Personal Assistant",
        "mode": "User Token (replies as you)",
        "version": "1.0.0"
    }


@app.post("/slack/events")
async def slack_events(
    request: Request,
    background_tasks: BackgroundTasks,
    x_slack_request_timestamp: str = Header(None),
    x_slack_signature: str = Header(None)
):
    """Handle Slack events"""
    
    body = await request.body()
    
    # Verify signature
    if Config.SLACK_SIGNING_SECRET and x_slack_signature:
        if not verify_slack_signature(
            Config.SLACK_SIGNING_SECRET,
            x_slack_request_timestamp,
            body,
            x_slack_signature
        ):
            raise HTTPException(status_code=401, detail="Invalid signature")
    
    payload = await request.json()
    
    # Handle URL verification
    if payload.get("type") == "url_verification":
        print("✅ URL verification challenge received")
        return JSONResponse(content={"challenge": payload.get("challenge")})
    
    # Handle events
    if payload.get("type") == "event_callback":
        event = payload.get("event", {})
        event_id = payload.get("event_id")
        
        # Prevent duplicates
        if event_id in processed_events:
            return JSONResponse(content={"status": "ok"})
        
        processed_events.add(event_id)
        if len(processed_events) > 1000:
            processed_events.clear()
        
        # Process message events in background (respond quickly to Slack)
        if event.get("type") == "message":
            background_tasks.add_task(handle_dm_message, event)
        
        return JSONResponse(content={"status": "ok"})
    
    return JSONResponse(content={"status": "ok"})


# ============================================================================
# MESSAGE HANDLER
# ============================================================================

async def handle_dm_message(event: Dict[str, Any]):
    """
    Handle incoming DM messages and reply as YOU
    """
    
    channel = event.get("channel")
    user = event.get("user")
    text = event.get("text", "")
    ts = event.get("ts")
    channel_type = event.get("channel_type", "")
    
    # Skip if:
    # 1. Message is from you (don't reply to yourself)
    # 2. It's a bot message
    # 3. It's edited/deleted
    if user == Config.MY_USER_ID:
        print("⚠️ Skipping: Message is from you")
        return
    
    if event.get("bot_id") or event.get("subtype") in ["message_changed", "message_deleted", "bot_message"]:
        print("⚠️ Skipping: Bot message or edit")
        return
    
    # Only respond to DMs (channel_type == "im")
    if channel_type != "im":
        print(f"⚠️ Skipping: Not a DM (type: {channel_type})")
        return
    
    print(f"💬 DM from {user}: {text}")
    
    try:
        # Get sender info
        user_info = await slack_client.get_user_info(user)
        sender_name = user_info.get("user", {}).get("real_name", "User")
        
        # Get conversation context (recent messages)
        history = await slack_client.get_conversation_history(channel, limit=10)
        context_messages = [
            msg.get("text", "") for msg in history.get("messages", [])[::-1]
            if msg.get("text")
        ]
        
        # Generate response in YOUR style
        response = await generate_response_in_my_style(
            incoming_message=text,
            sender_name=sender_name,
            channel=channel,
            my_user_id=Config.MY_USER_ID,
            conversation_context=context_messages
        )
        
        # Send response AS YOU
        result = await slack_client.post_message_as_user(
            channel=channel,
            text=response
        )
        
        if result.get("ok"):
            print(f"✅ Replied as YOU: {response[:80]}...")
        else:
            print(f"❌ Failed to send: {result.get('error')}")
            
    except Exception as e:
        print(f"❌ Error handling message: {e}")


# ============================================================================
# MANUAL TRIGGERS (Optional API endpoints)
# ============================================================================

@app.post("/analyze-style")
async def analyze_style_endpoint(channel_id: str):
    """Manually trigger style analysis for a channel"""
    try:
        style = await style_analyzer.analyze_my_style(channel_id, Config.MY_USER_ID)
        return {"status": "success", "style_profile": style}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/my-style")
async def get_my_style():
    """Get current style profile"""
    return {
        "style_profile": style_analyzer.get_style_profile(),
        "cached_messages": len(style_analyzer.my_messages_cache)
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 70)
    print("🤖 Starting Slack Personal Assistant (User Mode)")
    print("=" * 70)
    print(f"📍 Event URL: http://{Config.HOST}:{Config.PORT}/slack/events")
    print(f"👤 Acting as User ID: {Config.MY_USER_ID}")
    print("📱 Will reply to DMs in YOUR style")
    print("=" * 70)
    
    uvicorn.run(
        app,
        host=Config.HOST,
        port=Config.PORT,
        log_level="info"
    )
