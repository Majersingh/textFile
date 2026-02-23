"""
AI-Powered Online Test Platform
Author: Senior Developer (10+ years experience)
Tech Stack: Streamlit, LangChain, OpenAI, Pydantic
Features:
- Topic-based test generation
- PDF-based test generation
- Difficulty levels (Easy, Medium, Hard)
- Customizable question count (10-25)
- Score calculation with feedback
- Improvement suggestions
"""

import os
import tempfile
from typing import List, Literal, Optional
from enum import Enum
import json

import streamlit as st
from pydantic import BaseModel, Field

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi
import re

# ============================================================================
# DATA MODELS (Type-Safe Structures)
# ============================================================================

class DifficultyLevel(str, Enum):
    """Enumeration for difficulty levels"""
    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"


class QuestionType(str, Enum):
    """Enumeration for question types"""
    TOPIC_BASED = "Topic-Based Test"
    PDF_BASED = "PDF-Based Test"
    YOUTUBE_BASED = "YouTube Video Test" 


class MCQOption(BaseModel):
    """Single MCQ option"""
    option_letter: str = Field(description="A, B, C, or D")
    option_text: str = Field(description="The option text")


class Question(BaseModel):
    """Single question with metadata"""
    question_id: int = Field(description="Unique question number")
    question_text: str = Field(description="The question text")
    options: List[MCQOption] = Field(description="Four MCQ options")
    correct_answer: str = Field(description="Correct option letter (A/B/C/D)")
    explanation: str = Field(description="Why this answer is correct")
    difficulty: str = Field(description="Easy, Medium, or Hard")


class Quiz(BaseModel):
    """Complete quiz structure"""
    title: str = Field(description="Quiz title")
    questions: List[Question] = Field(description="List of questions")
    total_questions: int = Field(description="Total number of questions")


class UserAnswer(BaseModel):
    """User's answer to a question"""
    question_id: int
    selected_answer: str
    is_correct: bool
    correct_answer: str


class QuizResult(BaseModel):
    """Quiz result with analytics"""
    total_questions: int
    correct_answers: int
    score_percentage: float
    user_answers: List[UserAnswer]
    improvement_areas: List[str]


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Application configuration"""
    MIN_QUESTIONS = 10
    MAX_QUESTIONS = 25
    DEFAULT_QUESTIONS = 10
    MODEL_NAME = "gpt-4o-mini"
    TEMPERATURE = 0.7
    CHUNK_SIZE = 2000
    CHUNK_OVERLAP = 200


# ============================================================================
# LLM SERVICE LAYER (Business Logic)
# ============================================================================

class QuizGeneratorService:
    """Service class for quiz generation logic"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.MODEL_NAME,
            temperature=Config.TEMPERATURE
        )
        print("✅ QuizGeneratorService initialized")
    
    def generate_topic_based_quiz(
        self,
        topic: str,
        difficulty: DifficultyLevel,
        num_questions: int
    ) -> Quiz:
        """Generate quiz based on a topic"""
        print(f"🔹 Generating {num_questions} {difficulty.value} questions on: {topic}")
        
        # Define output parser
        parser = PydanticOutputParser(pydantic_object=Quiz)
        
        # Create prompt
        prompt = ChatPromptTemplate.from_template(
            """You are an expert quiz generator. Create a high-quality multiple-choice quiz.

Topic: {topic}
Difficulty Level: {difficulty}
Number of Questions: {num_questions}

Requirements:
1. Generate exactly {num_questions} questions
2. Each question must have 4 options (A, B, C, D)
3. Only ONE option should be correct
4. Provide clear explanations for correct answers
5. Match the difficulty level: {difficulty}
   - Easy: Basic recall and understanding
   - Medium: Application and analysis
   - Hard: Synthesis and evaluation
6. Questions should be diverse and cover different aspects of the topic
7. Avoid ambiguous or trick questions

{format_instructions}

Generate the quiz now:"""
        )
        
        # Create chain
        chain = prompt | self.llm | parser
        
        # Invoke
        result = chain.invoke({
            "topic": topic,
            "difficulty": difficulty.value,
            "num_questions": num_questions,
            "format_instructions": parser.get_format_instructions()
        })
        
        print(f"✅ Generated quiz with {len(result.questions)} questions")
        return result
    
    def generate_pdf_based_quiz(
        self,
        pdf_path: str,
        difficulty: DifficultyLevel,
        num_questions: int
    ) -> Quiz:
        """Generate quiz based on PDF content"""
        print(f"🔹 Loading PDF from: {pdf_path}")
        
        # Load PDF
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        print(f"✅ Loaded {len(docs)} pages from PDF")
        
        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        chunks = splitter.split_documents(docs)
        print(f"✅ Split into {len(chunks)} chunks")
        
        # Combine chunks (for smaller PDFs; for large PDFs use RAG)
        content = "\n\n".join([chunk.page_content for chunk in chunks[:5]])  # Use first 5 chunks
        
        # Define output parser
        parser = PydanticOutputParser(pydantic_object=Quiz)
        
        # Create prompt
        prompt = ChatPromptTemplate.from_template(
            """You are an expert quiz generator. Create a multiple-choice quiz based on the provided document content.

Document Content:
{content}

Difficulty Level: {difficulty}
Number of Questions: {num_questions}

Requirements:
1. Generate exactly {num_questions} questions based ONLY on the document content
2. Each question must have 4 options (A, B, C, D)
3. Only ONE option should be correct
4. Provide clear explanations referencing the document
5. Match the difficulty level: {difficulty}
6. Cover different sections of the document
7. Ensure questions test comprehension, not just recall

{format_instructions}

Generate the quiz now:"""
        )
        
        # Create chain
        chain = prompt | self.llm | parser
        
        # Invoke
        result = chain.invoke({
            "content": content[:8000],  # Limit context window
            "difficulty": difficulty.value,
            "num_questions": num_questions,
            "format_instructions": parser.get_format_instructions()
        })
        
        print(f"✅ Generated PDF-based quiz with {len(result.questions)} questions")
        return result
    
    def generate_improvement_suggestions(
        self,
        quiz: Quiz,
        result: QuizResult
    ) -> List[str]:
        """Generate personalized improvement suggestions"""
        print("🔹 Generating improvement suggestions")
        
        # Identify weak areas
        wrong_questions = [
            quiz.questions[ua.question_id - 1]
            for ua in result.user_answers
            if not ua.is_correct
        ]
        
        if not wrong_questions:
            return ["Excellent work! You got all answers correct! 🎉"]
        
        # Create prompt for suggestions
        prompt = ChatPromptTemplate.from_template(
            """You are a learning coach. Based on the questions the student answered incorrectly, provide 3-5 specific, actionable improvement suggestions.

Questions Answered Incorrectly:
{wrong_questions}

Student Score: {score}%

Provide:
1. Specific topics/concepts to review
2. Learning strategies
3. Resources or study techniques
4. Encouragement

Return as a JSON list of strings (each suggestion as one string).
Format: ["Suggestion 1", "Suggestion 2", ...]
"""
        )
        
        wrong_q_text = "\n\n".join([
            f"Q{q.question_id}: {q.question_text}\nCorrect: {q.correct_answer} - {q.explanation}"
            for q in wrong_questions[:5]  # Limit to 5
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({
            "wrong_questions": wrong_q_text,
            "score": result.score_percentage
        })
        
        # Parse response
        try:
            suggestions = json.loads(response.content)
            print(f"✅ Generated {len(suggestions)} suggestions")
            return suggestions
        except:
            # Fallback
            return [
                f"Review the concepts tested in questions you missed",
                f"Your accuracy was {result.score_percentage:.0f}%. Aim for 80%+ on the next attempt",
                f"Focus on {wrong_questions[0].difficulty} level questions"
            ]
    
    def extract_youtube_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from various YouTube URL formats"""
        patterns = [
            r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})',
            r'(?:https?:\/\/)?(?:www\.)?youtu\.be\/([a-zA-Z0-9_-]{11})',
            r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([a-zA-Z0-9_-]{11})',
            r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/v\/([a-zA-Z0-9_-]{11})',
        ]
    
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None

    def generate_youtube_based_quiz(
    self,
    youtube_url: str,
    difficulty: DifficultyLevel,
    num_questions: int
) -> Quiz:
        """Generate quiz based on YouTube video transcript"""
        print(f"🔹 Generating quiz from YouTube video: {youtube_url}")
    
        # Extract video ID
        video_id = self.extract_youtube_video_id(youtube_url)
        if not video_id:
            raise ValueError("Invalid YouTube URL. Please provide a valid YouTube link.")
        
        print(f"✅ Extracted video ID: {video_id}")
        
        try:
            # NEW API (v1.2.0+): Instantiate first, then call fetch()
            ytt_api = YouTubeTranscriptApi()
            transcript_obj = ytt_api.fetch(video_id)
            
            # Convert to raw data (list of dicts with 'text', 'start', 'duration')
            transcript_list = transcript_obj.to_raw_data()
            
            # Combine transcript into full text
            full_transcript = " ".join([entry['text'] for entry in transcript_list])
            print(f"✅ Retrieved transcript ({len(full_transcript)} characters)")
            
            # If transcript is too long, take first portion
            if len(full_transcript) > 8000:
                full_transcript = full_transcript[:8000]
                print(f"⚠️ Transcript truncated to 8000 characters")
            
        except Exception as e:
            # Fallback: Try old API for backward compatibility
            try:
                print("⚠️ Trying legacy API method...")
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                full_transcript = " ".join([entry['text'] for entry in transcript_list])
                print(f"✅ Retrieved transcript using legacy method ({len(full_transcript)} characters)")
            except:
                raise ValueError(f"Could not retrieve transcript. Error: {str(e)}. "
                                "Make sure the video has captions/subtitles enabled.")
        
        # Define output parser
        parser = PydanticOutputParser(pydantic_object=Quiz)
        
        # Create prompt
        prompt = ChatPromptTemplate.from_template(
            """You are an expert quiz generator. Create a multiple-choice quiz based on the YouTube video transcript provided.

    Video Transcript:
    {transcript}

    Difficulty Level: {difficulty}
    Number of Questions: {num_questions}

    Requirements:
    1. Generate exactly {num_questions} questions based ONLY on the video content
    2. Each question must have 4 options (A, B, C, D)
    3. Only ONE option should be correct
    4. Provide clear explanations referencing the video content
    5. Match the difficulty level: {difficulty}
    - Easy: Basic facts and definitions from the video
    - Medium: Understanding and application of concepts
    - Hard: Analysis and synthesis of video content
    6. Cover different parts of the video content
    7. Questions should test comprehension of key points

    {format_instructions}

    Generate the quiz now:"""
        )
        
        # Create chain
        chain = prompt | self.llm | parser
        
        # Invoke
        result = chain.invoke({
            "transcript": full_transcript,
            "difficulty": difficulty.value,
            "num_questions": num_questions,
            "format_instructions": parser.get_format_instructions()
        })
        
        print(f"✅ Generated YouTube-based quiz with {len(result.questions)} questions")
        return result

# ============================================================================
# UI COMPONENTS (Presentation Layer)
# ============================================================================

class QuizUI:
    """UI management class"""
    
    @staticmethod
    def render_header():
        """Render app header"""
        st.set_page_config(
            page_title="AI Quiz Platform",
            page_icon="📝",
            layout="wide"
        )
        
        st.title("🎓 AI-Powered Online Test Platform")
        st.markdown("""
        Create customized tests using AI. Choose between topic-based or PDF-based questions.
        """)
        st.markdown("---")
    
    @staticmethod
    def render_test_config() -> tuple:
        """Render test configuration sidebar"""
        with st.sidebar:
            st.header("⚙️ Test Configuration")
            
            # Test type selection
            test_type = st.selectbox(
                "Select Test Type",
                [
                    QuestionType.TOPIC_BASED.value,
                    QuestionType.PDF_BASED.value,
                    QuestionType.YOUTUBE_BASED.value  # NEW
                ],
                help="Choose test source: topic, PDF document, or YouTube video"
            )
            
            st.markdown("---")
            
            # Common settings
            difficulty = st.selectbox(
                "Difficulty Level",
                [d.value for d in DifficultyLevel],
                index=1,  # Default to Medium
                help="Select question difficulty"
            )
            
            num_questions = st.slider(
                "Number of Questions",
                min_value=Config.MIN_QUESTIONS,
                max_value=Config.MAX_QUESTIONS,
                value=Config.DEFAULT_QUESTIONS,
                help=f"Choose between {Config.MIN_QUESTIONS} and {Config.MAX_QUESTIONS} questions"
            )
            
            st.markdown("---")
            
            # Conditional inputs
            topic = None
            pdf_file = None
            youtube_url = None
            
            if test_type == QuestionType.TOPIC_BASED.value:
                topic = st.text_input(
                    "Enter Topic",
                    placeholder="e.g., General Knowledge, Python Programming, World History",
                    help="Enter the topic you want to be tested on"
                )
            elif test_type == QuestionType.PDF_BASED.value:
                pdf_file = st.file_uploader(
                    "Upload PDF Document",
                    type=["pdf"],
                    help="Upload a PDF to generate questions from"
                )
            else:  # YouTube-based
                youtube_url = st.text_input(
                    "Enter YouTube URL",
                    placeholder="https://www.youtube.com/watch?v=...",
                    help="Paste a YouTube video URL (video must have captions/subtitles)"
                )
                st.caption("⚠️ Note: Video must have captions/subtitles enabled")
            
            return test_type, difficulty, num_questions, topic, pdf_file, youtube_url

    @staticmethod
    def render_quiz_questions(quiz: Quiz):
        """Render quiz questions"""
        st.header(f"📝 {quiz.title}")
        st.markdown(f"**Total Questions:** {quiz.total_questions}")
        st.markdown("---")
        
        # Initialize answers in session state
        if 'user_answers_dict' not in st.session_state:
            st.session_state.user_answers_dict = {}
        
        # Render each question
        for idx, q in enumerate(quiz.questions):
            with st.container():
                st.markdown(f"### Question {q.question_id}")
                st.markdown(f"**{q.question_text}**")
                st.caption(f"Difficulty: {q.difficulty}")
                
                # Radio buttons for options
                options_text = [f"{opt.option_letter}. {opt.option_text}" for opt in q.options]
                
                selected = st.radio(
                    "Select your answer:",
                    options_text,
                    key=f"q_{q.question_id}",
                    index=None
                )
                
                # Store answer
                if selected:
                    st.session_state.user_answers_dict[q.question_id] = selected[0]  # Get letter (A/B/C/D)
                
                st.markdown("---")
    
    @staticmethod
    def render_results(quiz: Quiz, result: QuizResult, suggestions: List[str]):
        """Render quiz results"""
        st.header("📊 Test Results")
        
        # Score card
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Questions", result.total_questions)
        with col2:
            st.metric("Correct Answers", result.correct_answers)
        with col3:
            st.metric("Score", f"{result.score_percentage:.1f}%")
        
        st.markdown("---")
        
        # Performance indicator
        if result.score_percentage >= 80:
            st.success("🎉 Excellent Performance!")
        elif result.score_percentage >= 60:
            st.info("👍 Good Job! Room for improvement.")
        else:
            st.warning("📚 Keep Learning! Practice more.")
        
        st.markdown("---")
        
        # Detailed breakdown
        st.subheader("📋 Answer Breakdown")
        
        for ua in result.user_answers:
            question = quiz.questions[ua.question_id - 1]
            
            # Find the actual option text for user's answer and correct answer
            user_option_text = next(
                (opt.option_text for opt in question.options if opt.option_letter == ua.selected_answer),
                "No answer selected"
            )
            correct_option_text = next(
                (opt.option_text for opt in question.options if opt.option_letter == ua.correct_answer),
                ""
            )
            
            with st.expander(f"Question {ua.question_id}: {'✅ Correct' if ua.is_correct else '❌ Incorrect'}"):
                st.markdown(f"**{question.question_text}**")
                
                # Show user's answer
                if ua.is_correct:
                    st.success(f"✅ **Your Answer:** {ua.selected_answer}. {user_option_text}")
                else:
                    st.error(f"❌ **Your Answer:** {ua.selected_answer}. {user_option_text}")
                    st.success(f"✅ **Correct Answer:** {ua.correct_answer}. {correct_option_text}")
                
                st.markdown("---")
                st.markdown(f"**💡 Explanation:** {question.explanation}")

                
                st.markdown("---")
                
                # Improvement suggestions
                st.subheader("💡 Improvement Suggestions")
                for idx, suggestion in enumerate(suggestions, 1):
                    st.markdown(f"{idx}. {suggestion}")



# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================

def initialize_session_state():
    """Initialize session state variables"""
    if 'quiz' not in st.session_state:
        st.session_state.quiz = None
    if 'quiz_started' not in st.session_state:
        st.session_state.quiz_started = False
    if 'quiz_submitted' not in st.session_state:
        st.session_state.quiz_submitted = False
    if 'result' not in st.session_state:
        st.session_state.result = None
    if 'suggestions' not in st.session_state:
        st.session_state.suggestions = None


def reset_quiz():
    """Reset quiz state"""
    st.session_state.quiz = None
    st.session_state.quiz_started = False
    st.session_state.quiz_submitted = False
    st.session_state.result = None
    st.session_state.suggestions = None
    if 'user_answers_dict' in st.session_state:
        del st.session_state.user_answers_dict


# ============================================================================
# BUSINESS LOGIC
# ============================================================================

def calculate_result(quiz: Quiz, user_answers: dict) -> QuizResult:
    """Calculate quiz result"""
    user_answer_objects = []
    correct_count = 0
    
    for question in quiz.questions:
        user_answer = user_answers.get(question.question_id, None)
        is_correct = user_answer == question.correct_answer if user_answer else False
        
        if is_correct:
            correct_count += 1
        
        user_answer_objects.append(UserAnswer(
            question_id=question.question_id,
            selected_answer=user_answer or "No Answer",
            is_correct=is_correct,
            correct_answer=question.correct_answer
        ))
    
    score_percentage = (correct_count / quiz.total_questions) * 100
    
    return QuizResult(
        total_questions=quiz.total_questions,
        correct_answers=correct_count,
        score_percentage=score_percentage,
        user_answers=user_answer_objects,
        improvement_areas=[]  # Will be filled by AI
    )


# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    """Main application entry point"""
    
    # Initialize
    initialize_session_state()
    QuizUI.render_header()
    
    # Initialize service
    service = QuizGeneratorService()
    
    # Render config sidebar (now returns 6 values including youtube_url)
    test_type, difficulty, num_questions, topic, pdf_file, youtube_url = QuizUI.render_test_config()
    
    # Main content area
    if not st.session_state.quiz_started:
        # Quiz configuration screen
        st.info("👈 Configure your test in the sidebar and click 'Generate Test' to begin")
        
        # Show appropriate preview/info based on test type
        if test_type == QuestionType.YOUTUBE_BASED.value and youtube_url:
            st.info(f"🎥 **YouTube Video URL:** {youtube_url}")
            st.caption("AI will analyze the video transcript and generate questions")
        
        # Generate button
        can_generate = (
            (test_type == QuestionType.TOPIC_BASED.value and topic) or
            (test_type == QuestionType.PDF_BASED.value and pdf_file) or
            (test_type == QuestionType.YOUTUBE_BASED.value and youtube_url)
        )
        
        if st.button("🚀 Generate Test", disabled=not can_generate, type="primary"):
            with st.spinner("🤖 AI is generating your personalized test..."):
                try:
                    if test_type == QuestionType.TOPIC_BASED.value:
                        quiz = service.generate_topic_based_quiz(
                            topic=topic,
                            difficulty=DifficultyLevel(difficulty),
                            num_questions=num_questions
                        )
                    
                    elif test_type == QuestionType.PDF_BASED.value:
                        # Save PDF to temp file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(pdf_file.read())
                            pdf_path = tmp.name
                        
                        quiz = service.generate_pdf_based_quiz(
                            pdf_path=pdf_path,
                            difficulty=DifficultyLevel(difficulty),
                            num_questions=num_questions
                        )
                        
                        # Cleanup
                        os.remove(pdf_path)
                    
                    else:  # YouTube-based
                        quiz = service.generate_youtube_based_quiz(
                            youtube_url=youtube_url,
                            difficulty=DifficultyLevel(difficulty),
                            num_questions=num_questions
                        )
                    
                    st.session_state.quiz = quiz
                    st.session_state.quiz_started = True
                    st.rerun()
                    
                except ValueError as ve:
                    st.error(f"❌ {str(ve)}")
                    print(f"❌ Validation Error: {ve}")
                except Exception as e:
                    st.error(f"❌ Error generating quiz: {str(e)}")
                    print(f"❌ Error: {e}")
    
    elif not st.session_state.quiz_submitted:
        # Quiz taking screen
        QuizUI.render_quiz_questions(st.session_state.quiz)
        
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("📤 Submit Test", type="primary"):
                # Validate all questions answered
                if len(st.session_state.user_answers_dict) < 1:
                    st.warning("⚠️ Please answer at least 1 questions before submitting!")
                else:
                    with st.spinner("📊 Calculating results..."):
                        # Calculate result
                        result = calculate_result(
                            st.session_state.quiz,
                            st.session_state.user_answers_dict
                        )
                        st.session_state.result = result
                        
                        # Generate suggestions
                        suggestions = service.generate_improvement_suggestions(
                            st.session_state.quiz,
                            result
                        )
                        st.session_state.suggestions = suggestions
                        
                        st.session_state.quiz_submitted = True
                        st.rerun()
        
        with col2:
            if st.button("🔄 Reset Test"):
                reset_quiz()
                st.rerun()
    
    else:
        # Results screen
        QuizUI.render_results(
            st.session_state.quiz,
            st.session_state.result,
            st.session_state.suggestions
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🏠 Take Another Test", type="primary"):
                reset_quiz()
                st.rerun()
        with col2:
            if st.button("📥 Download Results"):
                # Create downloadable results
                results_text = f"""
=== Quiz Results ===
Score: {st.session_state.result.score_percentage:.1f}%
Correct: {st.session_state.result.correct_answers}/{st.session_state.result.total_questions}

=== Improvement Suggestions ===
"""
                for idx, suggestion in enumerate(st.session_state.suggestions, 1):
                    results_text += f"{idx}. {suggestion}\n"
                
                st.download_button(
                    label="📄 Download as TXT",
                    data=results_text,
                    file_name="quiz_results.txt",
                    mime="text/plain"
                )


if __name__ == "__main__":
    main()
