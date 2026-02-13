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
    
    async def get_thread_replies(
        self,
        channel: str,
        thread_ts: str,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Get all replies in a thread
        """
        url = f"{self.base_url}/conversations.replies"
        
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        
        params = {
            "channel": channel,
            "ts": thread_ts,
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

async def generate_response_with_thread_context(
    incoming_message: str,
    sender_name: str,
    channel: str,
    my_user_id: str,
    thread_context: str,
    all_messages: List[str]
) -> str:
    """
    Generate a response considering the ENTIRE thread conversation
    """
    
    style_profile = style_analyzer.get_style_profile()
    
    # Build example messages from your history
    examples_text = ""
    if style_analyzer.my_messages_cache:
        examples = list(style_analyzer.my_messages_cache)[:5]
        examples_text = "\n".join([f"Example {i+1}: {msg}" for i, msg in enumerate(examples)])
    
    # Enhanced prompt with full thread awareness
    response_prompt = ChatPromptTemplate.from_template(
        """You are replying to a Slack thread on behalf of a user. You have the FULL thread context.

YOUR COMMUNICATION STYLE:
{style_profile}

EXAMPLES OF HOW YOU WRITE:
{examples}

COMPLETE THREAD CONVERSATION (in chronological order):
{thread_context}

CURRENT MESSAGE FROM {sender} (that mentions you):
"{message}"

Generate a response that:
1. Matches your writing style exactly
2. Shows you've READ and UNDERSTOOD the entire thread context
3. Responds appropriately to the current message while considering all previous messages
4. References earlier points in the thread if relevant
5. Is natural and conversational
6. Keep it concise but contextually complete (2-4 sentences unless more needed)

Your response:"""
    )
    
    chain = response_prompt | llm | StrOutputParser()
    
    try:
        response = await chain.ainvoke({
            "style_profile": style_profile,
            "examples": examples_text or "No examples available yet",
            "thread_context": thread_context or "No prior context",
            "sender": sender_name,
            "message": incoming_message
        })
        
        print(f"✅ Generated context-aware response: {response[:100]}...")
        return response.strip()
        
    except Exception as e:
        print(f"❌ Response generation failed: {e}")
        return "Thanks for tagging me! I'll look into this."


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
async def slack_events(request: Request, background_tasks: BackgroundTasks):
    """Handle Slack events - DMs and mentions"""
    
    try:
        payload = await request.json()
        event_type = payload.get("type")
        
        # Handle URL verification
        if event_type == "url_verification":
            return {"challenge": payload.get("challenge")}
        
        # Handle message events
        if event_type == "event_callback":
            event = payload.get("event", {})
            event_id = payload.get("event_id")
            
            # Skip duplicates
            if event_id in processed_events:
                return {"status": "ok"}
            
            processed_events.add(event_id)
            if len(processed_events) > 1000:
                processed_events.clear()
            
            message_type = event.get("type")
            
            # Handle DMs
            if message_type == "message" and event.get("channel_type") == "im":
                background_tasks.add_task(handle_dm_message, event)
            
            # Handle mentions in channels (including threads)
            elif message_type == "message" and event.get("channel_type") in ["channel", "group"]:
                text = event.get("text", "")
                if f"<@{Config.MY_USER_ID}>" in text:
                    print(f"📣 Mention detected in {'thread' if event.get('thread_ts') else 'channel'}")
                    background_tasks.add_task(handle_mention, event)
            
            return {"status": "ok"}
        
        return {"status": "ok"}
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


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

async def handle_mention(event: Dict[str, Any]):
    """
    Handle when someone mentions you (@your_name) in a channel
    Reply in the thread with full thread context
    """
    
    channel = event.get("channel")
    user = event.get("user")
    text = event.get("text", "")
    ts = event.get("ts")
    thread_ts = event.get("thread_ts")  # If already in a thread
    
    # Skip if message is from you
    if user == Config.MY_USER_ID:
        print("⚠️ Skipping mention: From yourself")
        return
    
    # Skip bot messages
    if event.get("bot_id") or event.get("subtype") in ["message_changed", "message_deleted"]:
        print("⚠️ Skipping mention: Bot or edit")
        return
    
    # Check if you're actually mentioned
    if f"<@{Config.MY_USER_ID}>" not in text:
        print("⚠️ Not actually mentioned")
        return
    
    print(f"📣 Mentioned by {user} in channel {channel}: {text}")
    
    try:
        # Get sender info
        user_info = await slack_client.get_user_info(user)
        sender_name = user_info.get("user", {}).get("real_name", "User")
        
        # Remove the mention from text to get clean message
        clean_text = text.replace(f"<@{Config.MY_USER_ID}>", "").strip()
        
        # Get FULL thread context
        context_messages = []
        thread_summary = ""
        
        # Determine if this is a thread reply or a new thread
        parent_ts = thread_ts or ts  # Use thread_ts if in thread, else this message starts thread
        
        print(f"🔍 Getting full thread context (parent_ts: {parent_ts})...")
        
        # Get ALL thread replies using conversations.replies API
        thread_history = await slack_client.get_thread_replies(
            channel=channel,
            thread_ts=parent_ts
        )
        
        if thread_history.get("ok"):
            messages = thread_history.get("messages", [])
            print(f"✅ Found {len(messages)} messages in thread")
            
            # Build context from ALL messages in the thread
            for msg in messages:
                msg_user = msg.get("user", "Unknown")
                msg_text = msg.get("text", "")
                msg_ts = msg.get("ts", "")
                
                # Get user name for each message
                if msg_user and msg_user != "Unknown":
                    try:
                        msg_user_info = await slack_client.get_user_info(msg_user)
                        msg_user_name = msg_user_info.get("user", {}).get("real_name", msg_user)
                    except:
                        msg_user_name = msg_user
                else:
                    msg_user_name = "Unknown"
                
                # Mark if this is the current message (where you were mentioned)
                is_current = (msg_ts == ts)
                marker = " [← YOU WERE MENTIONED HERE]" if is_current else ""
                
                # Add to context
                if msg_text:
                    context_messages.append(f"{msg_user_name}: {msg_text}{marker}")
            
            # Create thread summary
            thread_summary = "\n".join(context_messages)
            print(f"📝 Thread context:\n{thread_summary[:500]}...")
            
        else:
            # Fallback: get recent channel messages if thread fetch fails
            print(f"⚠️ Could not get thread replies, using channel history")
            history = await slack_client.get_conversation_history(channel, limit=10)
            context_messages = [
                msg.get("text", "") for msg in history.get("messages", [])[::-1]
                if msg.get("text")
            ]
            thread_summary = "\n".join(context_messages)
        
        # Analyze your style if not done yet
        if not style_analyzer.style_profile:
            await style_analyzer.analyze_my_style(channel, Config.MY_USER_ID)
        
        # Generate response with FULL thread context
        response = await generate_response_with_thread_context(
            incoming_message=clean_text or "What's up?",
            sender_name=sender_name,
            channel=channel,
            my_user_id=Config.MY_USER_ID,
            thread_context=thread_summary,
            all_messages=context_messages
        )
        
        # Reply in thread (use thread_ts if exists, otherwise use message ts)
        result = await slack_client.post_message_as_user(
            channel=channel,
            text=response,
            thread_ts=parent_ts  # Always reply in the thread
        )
        
        if result.get("ok"):
            print(f"✅ Replied to mention in thread: {response[:80]}...")
        else:
            print(f"❌ Failed to send: {result.get('error')}")
            
    except Exception as e:
        print(f"❌ Error handling mention: {e}")
        import traceback
        traceback.print_exc()

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
