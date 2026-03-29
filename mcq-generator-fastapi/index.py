"""
AI-Powered Online Test Platform — FastAPI Server
Author: Converted from Streamlit by Senior Developer
Tech Stack: FastAPI, LangChain, OpenAI, Pydantic

Endpoints:
  POST /generate-quiz/topic    → topic + difficulty + num_questions → Quiz
  POST /generate-quiz/pdf      → PDF file + difficulty + num_questions → Quiz
  POST /generate-quiz/youtube  → youtube_url + difficulty + num_questions → Quiz
  POST /submit-quiz            → Quiz + user_answers → QuizResult + suggestions
  GET  /                       → Health check
"""

import os
import re
import json
import tempfile
from typing import List, Optional, Dict
from enum import Enum

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    MIN_QUESTIONS = 10
    MAX_QUESTIONS = 50
    DEFAULT_QUESTIONS = 10
    MODEL_NAME = "gpt-4o-mini"
    TEMPERATURE = 0.7
    CHUNK_SIZE = 2000
    CHUNK_OVERLAP = 200
    MAX_CONTEXT = 8000


# ============================================================================
# ENUMS
# ============================================================================

class DifficultyLevel(str, Enum):
    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"


# ============================================================================
# SHARED DATA MODELS
# ============================================================================

class MCQOption(BaseModel):
    option_letter: str = Field(description="A, B, C, or D")
    option_text: str = Field(description="The option text")


class Question(BaseModel):
    question_id: int = Field(description="Unique question number starting from 1")
    question_text: str = Field(description="The question text")
    options: List[MCQOption] = Field(description="Exactly four MCQ options")
    correct_answer: str = Field(description="Correct option letter: A, B, C, or D")
    explanation: str = Field(description="Why this answer is correct")
    difficulty: str = Field(description="Easy, Medium, or Hard")


class Quiz(BaseModel):
    title: str = Field(description="Descriptive quiz title")
    questions: List[Question] = Field(description="List of questions")
    total_questions: int = Field(description="Total number of questions")


# ============================================================================
# REQUEST MODELS
# ============================================================================

class TopicQuizRequest(BaseModel):
    topic: str = Field(..., example="Python Programming Basics")
    difficulty: DifficultyLevel = Field(default=DifficultyLevel.MEDIUM)
    num_questions: int = Field(default=10, ge=10, le=25)


class YouTubeQuizRequest(BaseModel):
    youtube_url: str = Field(..., example="https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    difficulty: DifficultyLevel = Field(default=DifficultyLevel.MEDIUM)
    num_questions: int = Field(default=10, ge=10, le=25)


# ============================================================================
# RESPONSE MODELS
# ============================================================================

class UserAnswer(BaseModel):
    question_id: int
    selected_answer: str = Field(description="The letter the user selected: A/B/C/D or 'No Answer'")
    is_correct: bool
    correct_answer: str


class QuizResult(BaseModel):
    total_questions: int
    correct_answers: int
    score_percentage: float
    user_answers: List[UserAnswer]
    improvement_suggestions: List[str]


class SubmitQuizRequest(BaseModel):
    quiz: Quiz
    user_answers: Dict[int, str] = Field(
        ...,
        description="Map of question_id → selected answer letter (A/B/C/D)",
        example={1: "A", 2: "C", 3: "B"}
    )


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="AI Quiz Generator API",
    description=(
        "Generate MCQ quizzes from a topic, PDF document, or YouTube video. "
        "Submit answers to receive scored results and AI improvement suggestions."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# LLM HELPER
# ============================================================================

def get_llm() -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY environment variable is not set."
        )
    return ChatOpenAI(model=Config.MODEL_NAME, temperature=Config.TEMPERATURE)


# ============================================================================
# CORE QUIZ GENERATION LOGIC
# ============================================================================

QUIZ_PROMPT_TEMPLATE = """You are an expert quiz generator. Create a high-quality multiple-choice quiz.

{source_label}:
{source_content}

Difficulty Level: {difficulty}
Number of Questions: {num_questions}

Requirements:
1. Generate exactly {num_questions} questions.
2. Each question must have exactly 4 options (A, B, C, D).
3. Only ONE option should be correct.
4. Provide clear, educational explanations for correct answers.
5. Match the difficulty level:
   - Easy: Basic recall and understanding
   - Medium: Application and analysis
   - Hard: Synthesis and evaluation
6. Cover diverse aspects of the {source_type}; avoid repetition.
7. Avoid ambiguous or trick questions.

{format_instructions}

Generate the quiz now:"""


def run_quiz_chain(
    source_label: str,
    source_content: str,
    source_type: str,
    difficulty: DifficultyLevel,
    num_questions: int,
) -> Quiz:
    llm = get_llm()
    parser = PydanticOutputParser(pydantic_object=Quiz)
    prompt = ChatPromptTemplate.from_template(QUIZ_PROMPT_TEMPLATE)
    chain = prompt | llm | parser
    return chain.invoke({
        "source_label": source_label,
        "source_content": source_content[:Config.MAX_CONTEXT],
        "source_type": source_type,
        "difficulty": difficulty.value,
        "num_questions": num_questions,
        "format_instructions": parser.get_format_instructions(),
    })


# ============================================================================
# YOUTUBE HELPERS
# ============================================================================

def extract_video_id(url: str) -> Optional[str]:
    patterns = [
        r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def fetch_youtube_transcript(video_id: str) -> str:
    try:
        ytt = YouTubeTranscriptApi()
        transcript_obj = ytt.fetch(video_id)
        entries = transcript_obj.to_raw_data()
        return " ".join(e["text"] for e in entries)
    except Exception:
        try:
            entries = YouTubeTranscriptApi.get_transcript(video_id)
            return " ".join(e["text"] for e in entries)
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Could not fetch transcript for video '{video_id}'. "
                    "Make sure the video has captions/subtitles enabled. "
                    f"Error: {str(e)}"
                ),
            )


# ============================================================================
# ROUTES
# ============================================================================

@app.get("/", tags=["Health"])
def root():
    """Health check — confirms the API is running."""
    return {"status": "ok", "message": "AI Quiz Generator API v2 is running."}


# --------------------------------------------------------------------------- #
#  1. TOPIC-BASED QUIZ
# --------------------------------------------------------------------------- #

@app.post(
    "/generate-quiz/topic",
    response_model=Quiz,
    tags=["Generate Quiz"],
    summary="Generate a quiz from a topic",
)
def generate_topic_quiz(request: TopicQuizRequest):
    """
    Generate a multiple-choice quiz on any topic.

    **Body:**
    - `topic` (str) — e.g., "World War II", "Python Decorators"
    - `difficulty` (str) — `"Easy"` | `"Medium"` | `"Hard"`
    - `num_questions` (int) — 10–25
    """
    try:
        return run_quiz_chain(
            source_label="Topic",
            source_content=request.topic,
            source_type="topic",
            difficulty=request.difficulty,
            num_questions=request.num_questions,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quiz generation failed: {str(e)}")


# --------------------------------------------------------------------------- #
#  2. PDF-BASED QUIZ
# --------------------------------------------------------------------------- #

@app.post(
    "/generate-quiz/pdf",
    response_model=Quiz,
    tags=["Generate Quiz"],
    summary="Generate a quiz from an uploaded PDF",
)
async def generate_pdf_quiz(
    file: UploadFile = File(..., description="PDF document to generate quiz from"),
    difficulty: DifficultyLevel = Form(default=DifficultyLevel.MEDIUM),
    num_questions: int = Form(default=10, ge=10, le=25),
):
    """
    Upload a PDF and generate a quiz based on its content.

    **Form fields:**
    - `file` — PDF file (multipart/form-data)
    - `difficulty` — `"Easy"` | `"Medium"` | `"Hard"`
    - `num_questions` — 10–25
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            pdf_path = tmp.name

        # Load and split PDF
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
        )
        chunks = splitter.split_documents(docs)
        combined_text = "\n\n".join(chunk.page_content for chunk in chunks[:5])

        os.remove(pdf_path)

        return run_quiz_chain(
            source_label="Document Content",
            source_content=combined_text,
            source_type="document",
            difficulty=difficulty,
            num_questions=num_questions,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF quiz generation failed: {str(e)}")


# --------------------------------------------------------------------------- #
#  3. YOUTUBE-BASED QUIZ
# --------------------------------------------------------------------------- #

@app.post(
    "/generate-quiz/youtube",
    response_model=Quiz,
    tags=["Generate Quiz"],
    summary="Generate a quiz from a YouTube video",
)
def generate_youtube_quiz(request: YouTubeQuizRequest):
    """
    Generate a quiz based on a YouTube video's transcript/captions.

    **Body:**
    - `youtube_url` (str) — Full YouTube URL (video must have captions enabled)
    - `difficulty` (str) — `"Easy"` | `"Medium"` | `"Hard"`
    - `num_questions` (int) — 10–25
    """
    video_id = extract_video_id(request.youtube_url)
    if not video_id:
        raise HTTPException(
            status_code=400,
            detail="Invalid YouTube URL. Provide a valid youtube.com or youtu.be link."
        )

    transcript = fetch_youtube_transcript(video_id)

    try:
        return run_quiz_chain(
            source_label="Video Transcript",
            source_content=transcript,
            source_type="video",
            difficulty=request.difficulty,
            num_questions=request.num_questions,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"YouTube quiz generation failed: {str(e)}")


# --------------------------------------------------------------------------- #
#  4. SUBMIT QUIZ → SCORE + SUGGESTIONS
# --------------------------------------------------------------------------- #

@app.post(
    "/submit-quiz",
    response_model=QuizResult,
    tags=["Submit & Score"],
    summary="Submit answers and get score + improvement suggestions",
)
def submit_quiz(request: SubmitQuizRequest):
    """
    Submit user answers for a quiz and receive:
    - Score (correct answers + percentage)
    - Per-question correctness breakdown
    - AI-generated improvement suggestions

    **Body:**
    - `quiz` — The full Quiz object returned by any `/generate-quiz/*` endpoint
    - `user_answers` — Dict mapping `question_id` (int) → selected answer letter (`"A"/"B"/"C"/"D"`)

    **Example:**
    ```json
    {
      "quiz": { ... },
      "user_answers": { "1": "A", "2": "C", "3": "B" }
    }
    ```
    """
    quiz = request.quiz
    user_answers = request.user_answers

    answered: List[UserAnswer] = []
    correct_count = 0

    for question in quiz.questions:
        selected = user_answers.get(question.question_id, None)
        is_correct = (selected == question.correct_answer) if selected else False
        if is_correct:
            correct_count += 1
        answered.append(UserAnswer(
            question_id=question.question_id,
            selected_answer=selected or "No Answer",
            is_correct=is_correct,
            correct_answer=question.correct_answer,
        ))

    score_pct = (correct_count / quiz.total_questions) * 100

    # Generate improvement suggestions via LLM
    wrong_questions = [
        quiz.questions[ua.question_id - 1]
        for ua in answered
        if not ua.is_correct
    ]

    if not wrong_questions:
        suggestions = ["Excellent! You answered every question correctly. 🎉"]
    else:
        try:
            llm = get_llm()
            suggestion_prompt = ChatPromptTemplate.from_template(
                """You are a learning coach. A student answered the following questions incorrectly.
Provide 3-5 specific, actionable improvement suggestions to help them improve.

Questions Answered Incorrectly:
{wrong_questions}

Student Score: {score}%

Return ONLY a JSON array of strings. Each string is one suggestion.
Example format: ["Suggestion 1", "Suggestion 2", "Suggestion 3"]"""
            )
            chain = suggestion_prompt | llm
            wrong_text = "\n\n".join(
                f"Q{q.question_id}: {q.question_text}\n"
                f"Correct: {q.correct_answer} — {q.explanation}"
                for q in wrong_questions[:5]
            )
            response = chain.invoke({
                "wrong_questions": wrong_text,
                "score": f"{score_pct:.1f}",
            })
            suggestions = json.loads(response.content)
            if not isinstance(suggestions, list):
                raise ValueError("Not a list")
        except Exception:
            suggestions = [
                f"Review the topics covered in the questions you missed.",
                f"Your score was {score_pct:.0f}%. Aim for 80%+ on the next attempt.",
                f"Focus on {wrong_questions[0].difficulty.lower()}-level questions in this subject.",
            ]

    return QuizResult(
        total_questions=quiz.total_questions,
        correct_answers=correct_count,
        score_percentage=round(score_pct, 2),
        user_answers=answered,
        improvement_suggestions=suggestions,
    )
