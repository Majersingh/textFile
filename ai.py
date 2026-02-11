import os  # Standard library for file/path operations
import tempfile  # For creating temporary files
from typing import TypedDict, List  # For type hints
import math  # For calculator tool

import streamlit as st  # Streamlit for UI

# LangChain + OpenAI components
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # OpenAI chat model + embeddings
from langchain_community.document_loaders import PyPDFLoader  # Loader to read PDF pages
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Split long texts into chunks
from langchain_community.vectorstores import FAISS  # FAISS vector store for similarity search
from langchain_core.documents import Document  # Document type used by LangChain
from langchain_classic.memory import ConversationBufferMemory  # Simple in-memory chat history

from langchain_core.prompts import ChatPromptTemplate  # For building chat prompts
from langchain_core.output_parsers import StrOutputParser  # Ensures we get string output from LLM

# Tools
from langchain_core.tools import tool  # For defining tools
from langchain_community.tools import DuckDuckGoSearchRun  # Web search tool

# LangGraph for agent graph orchestration
from langgraph.graph import StateGraph, START, END  # Graph building utilities
from langgraph.prebuilt import create_react_agent  # Prebuilt ReAct agent

print("✅ Imports completed")  # Debug print


# ---------------------------
# UI Setup
# ---------------------------
st.set_page_config(page_title="Advanced PDF Q&A", page_icon="📄")  # Basic page config
print("✅ Streamlit page config set")

st.title("📄 Advanced PDF Q&A with ReAct Tools")  # Page title
print("✅ Streamlit title rendered")


# ---------------------------
# LLM & Embeddings
# ---------------------------
# Initialize the chat model; temperature=0 for deterministic answers
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
print("✅ ChatOpenAI model initialized")

# Initialize embeddings model (used to embed PDF chunks)
embeddings = OpenAIEmbeddings()
print("✅ OpenAIEmbeddings initialized")


# ---------------------------
# Tools Definition
# ---------------------------
@tool
def calculator(expression: str) -> str:
    """
    Safely evaluate a basic math expression.
    Examples: "2 + 2", "10 * (5 - 3)", "sqrt(16)".
    """
    allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        print(f"🔹 Calculator tool called with: {expression} = {result}")
        return str(result)
    except Exception as e:
        print(f"⚠️ Calculator error: {e}")
        return f"Error evaluating expression: {e}"


# Web search tool
web_search = DuckDuckGoSearchRun()
print("✅ Tools (calculator, web_search) initialized")


# ---------------------------
# Memory
# ---------------------------
# Conversation buffer memory to store past Q&A (not deeply used but kept for future extension)
memory = ConversationBufferMemory(return_messages=True)
print("✅ ConversationBufferMemory initialized")


# ---------------------------
# Upload PDF
# ---------------------------
# File uploader widget; only accepts PDF
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
print(f"📄 Uploaded file object: {uploaded_file}")  # Debug: show whether a file is present

# If no file uploaded, show info and stop execution
if uploaded_file is None:
    print("ℹ️ No file uploaded yet. Stopping script.")
    st.info("Upload a PDF to start.")
    st.stop()


# Save temp file
# Create a temporary file on disk so PyPDFLoader can read it
with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
    tmp.write(uploaded_file.read())  # Write the uploaded file bytes to temp file
    pdf_path = tmp.name  # Save the temp file path
print(f"✅ PDF saved to temporary path: {pdf_path}")


# Load & split
# Use PyPDFLoader to load the PDF into LangChain Document objects
loader = PyPDFLoader(pdf_path)
print("✅ PyPDFLoader initialized")

docs = loader.load()  # Load all pages as Documents
print(f"✅ Loaded {len(docs)} document pages from PDF")

# Initialize text splitter to break documents into overlapping chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
print("✅ RecursiveCharacterTextSplitter created with chunk_size=1000, chunk_overlap=200")

# Split all documents into smaller chunks for better retrieval
chunks = splitter.split_documents(docs)
print(f"✅ Split into {len(chunks)} text chunks")


# Build vector store
# Create a FAISS vector store from the chunks using the embeddings model
vectorstore = FAISS.from_documents(chunks, embeddings)
print("✅ FAISS vectorstore built from chunks")

# Turn the vector store into a retriever with top-k = 4
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
print("✅ Retriever created from vectorstore with k=4")

st.success("PDF indexed successfully!")  # Notify user in UI
print("✅ PDF indexed successfully (UI message shown)")


# ---------------------------
# ReAct Agent with Tools
# ---------------------------
# Create a ReAct agent that can use calculator and web search
react_agent = create_react_agent(
    llm,
    tools=[calculator, web_search],
)
print("✅ ReAct tools agent initialized")


# ---------------------------
# LangGraph State
# ---------------------------
# Define the shared state structure used by LangGraph nodes
class GraphState(TypedDict):
    question: str  # User's question
    docs: List[Document]  # Retrieved documents (if any)
    answer: str  # Final answer to show to user
    decision: str  # "RETRIEVE" or "NO_RETRIEVE"


print("✅ GraphState TypedDict defined")


# ---------------------------
# Node 1: Decide if retrieval is needed
# ---------------------------
# Prompt that asks the LLM whether it needs to use the PDF context
decide_prompt = ChatPromptTemplate.from_template(
    """You are an AI assistant and User have uploaded a file pdf/txt or any text containing.
Question: {question}

Decide: Do we need to look into the document to answer this question?
Reply with only: "RETRIEVE" or "NO_RETRIEVE"."""
)
print("✅ decide_prompt template created")

# Build an LCEL chain: prompt -> llm -> string output
decide_chain = decide_prompt | llm | StrOutputParser()
print("✅ decide_chain (prompt | llm | StrOutputParser) created")


def decide_node(state: GraphState):
    """
    Node that decides if we should retrieve context from the PDF
    based on the question content. Uses streaming.
    """
    print("🔹 decide_node called with state:", state)
    q = state["question"]  # Extract question from state
    print("🔹 decide_node question:", q)

    # Stream the decision from LLM
    chunks = []
    for part in decide_chain.stream({"question": q}):
        chunks.append(part)
    decision = "".join(chunks).strip()
    print("🔹 decide_node decision from LLM:", decision)

    # Return a partial state update containing the decision
    return {"decision": decision}


# ---------------------------
# Node 2: Retrieve docs
# ---------------------------
def retrieve_node(state: GraphState):
    """
    Node that retrieves relevant documents from the vectorstore
    if retrieval is needed.
    """
    print("🔹 retrieve_node called with state:", state)
    q = state["question"]  # Extract question
    print("🔹 retrieve_node question:", q)

    # Use retriever.invoke (new API) to get relevant documents
    docs = retriever.invoke(q)
    print(f"🔹 retrieve_node retrieved {len(docs)} docs")

    # Return docs as state update
    return {"docs": docs}


# ---------------------------
# Node 3: Answer with context (streaming)
# ---------------------------
# Prompt that instructs LLM to answer using ONLY the provided context
answer_prompt = ChatPromptTemplate.from_template(
    """Answer the question using ONLY the context below assuming user is talking about only this context.
If the answer is not in the context, say "I don't know based on the document."

Context:
{context}

Question:
{question}

Answer:"""
)
print("✅ answer_prompt template created")

# Chain: prompt -> llm -> string
answer_chain = answer_prompt | llm | StrOutputParser()
print("✅ answer_chain created")


def answer_node(state: GraphState):
    """
    Node that uses retrieved docs to answer the question with streaming.
    """
    print("🔹 answer_node called with state:", state)
    q = state["question"]  # Extract question
    docs = state.get("docs", [])  # Get docs if present, else empty list
    print(f"🔹 answer_node question: {q}")
    print(f"🔹 answer_node number of docs: {len(docs)}")

    # Build context string by joining page_content from all retrieved docs
    context = "\n\n".join([d.page_content for d in docs])
    print("🔹 answer_node built context of length:", len(context))

    # Stream the answer from LLM
    chunks = []
    for token in answer_chain.stream({"question": q, "context": context}):
        chunks.append(token)
    answer = "".join(chunks)
    print("🔹 answer_node LLM answer:", answer)

    # Save Q&A into memory buffer (for potential future use)
    memory.save_context({"question": q}, {"answer": answer})
    print("🔹 answer_node saved to memory")

    # Return answer to update state
    return {"answer": answer}


# ---------------------------
# Node 4: Direct answer with ReAct tools (calculator + web search)
# ---------------------------
def direct_answer_node(state: GraphState):
    """
    Node that answers using a ReAct-style tools agent
    (calculator + web search), without using the PDF.
    """
    print("🔹 direct_answer_node called with state:", state)
    q = state["question"]  # Extract question
    print("🔹 direct_answer_node question:", q)

    # Use the ReAct agent with tools
    # Agent expects a messages-style input
    result = react_agent.invoke(
        {"messages": [{"role": "user", "content": q}]}
    )
    # Last message is the final assistant answer
    answer = result["messages"][-1].content
    print("🔹 direct_answer_node agent final answer:", answer)

    # Save Q&A into memory
    memory.save_context({"question": q}, {"answer": answer})
    print("🔹 direct_answer_node saved to memory")

    # Return answer to update state
    return {"answer": answer}


# ---------------------------
# Build LangGraph
# ---------------------------
# Initialize the LangGraph StateGraph with our GraphState schema
graph = StateGraph(GraphState)
print("✅ StateGraph initialized with GraphState schema")

# Register nodes in the graph by name
graph.add_node("decide", decide_node)
print("✅ Node 'decide' added")

graph.add_node("retrieve", retrieve_node)
print("✅ Node 'retrieve' added")

graph.add_node("answer", answer_node)
print("✅ Node 'answer' added")

graph.add_node("direct_answer", direct_answer_node)
print("✅ Node 'direct_answer' added")

# Set the entry point node when the graph starts
graph.set_entry_point("decide")
print("✅ Entry point set to 'decide'")


# Conditional routing function used by LangGraph
def route_decision(state: GraphState):
    """
    Decide which node to go to after 'decide' node based on LLM output.
    """
    print("🔹 route_decision called with state:", state)
    decision = state.get("decision", "")
    print("🔹 route_decision decision:", decision)

    # If the LLM said we need retrieval, go to 'retrieve' node
    if "RETRIEVE" == decision:
        print("🔹 route_decision routing to 'retrieve'")
        return "retrieve"
    # Otherwise, go directly to 'direct_answer'
    else:
        print("🔹 route_decision routing to 'direct_answer'")
        return "direct_answer"


# Add conditional edges from 'decide' node based on route_decision output
graph.add_conditional_edges(
    "decide",
    route_decision,
    {
        "retrieve": "retrieve",
        "direct_answer": "direct_answer",
    },
)
print("✅ Conditional edges from 'decide' added")

# Add fixed edges for downstream flow
graph.add_edge("retrieve", "answer")  # After retrieval, go to answer node
print("✅ Edge 'retrieve' -> 'answer' added")

graph.add_edge("answer", END)  # After answer, graph ends
print("✅ Edge 'answer' -> END added")

graph.add_edge("direct_answer", END)  # Direct answer also ends
print("✅ Edge 'direct_answer' -> END added")

# Compile the graph into a callable object
app_graph = graph.compile()
print("✅ Graph compiled into app_graph")


# ---------------------------
# Chat UI with Streaming
# ---------------------------
# Initialize session history the first time
if "history" not in st.session_state:
    st.session_state.history = []
    print("✅ Initialized st.session_state.history as empty list")

# Text input for the user's question
question = st.text_input("Ask a question about the PDF:")
print("🔹 Current question input value:", question)

# Button to trigger the graph execution with streaming
if st.button("Ask") and question:
    print("✅ 'Ask' button pressed and question is non-empty")
    
    # Create placeholder for streaming output
    with st.spinner("Thinking..."):
        print("🔹 Invoking app_graph with streaming")
        
        # Use stream mode to get updates as nodes complete
        final_answer = ""
        for event in app_graph.stream({"question": question}, stream_mode="updates"):
            print(f"🔹 Stream event: {event}")
            # Check if any node returned an answer
            for node_name, node_output in event.items():
                if "answer" in node_output:
                    final_answer = node_output["answer"]
        
        print("🔹 Final answer from stream:", final_answer)

    # Append Q&A to session history for UI display
    st.session_state.history.append((question, final_answer))
    print("✅ Appended Q&A to st.session_state.history")

# Show chat history in UI
st.subheader("💬 Chat History")
print("✅ Rendering chat history")
for q, a in st.session_state.history:
    print(f"🔹 History item - Q: {q}, A: {a}")
    st.markdown(f"**You:** {q}")
    st.markdown(f"**AI:** {a}")
    st.markdown("---")

# Show feature badges
st.sidebar.title("🚀 Features")
st.sidebar.markdown("""
- ✅ **PDF Q&A** with semantic search
- ✅ **ReAct Agent** with tools
- ✅ **Calculator** for math
- ✅ **Web Search** for current info
- ✅ **Streaming** responses
- ✅ **Conditional routing** (PDF vs tools)
""")

# Cleanup temp file at the end of script execution
try:
    os.remove(pdf_path)
    print(f"✅ Temporary PDF file removed: {pdf_path}")
except Exception as e:
    # In case file is already removed or inaccessible, log the error
    print(f"⚠️ Error while removing temp file {pdf_path}: {e}")
