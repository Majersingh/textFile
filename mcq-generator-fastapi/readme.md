# AI Quiz Generator — FastAPI Server

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/generate-quiz/topic` | Quiz from topic (JSON body) |
| POST | `/generate-quiz/pdf` | Quiz from uploaded PDF (multipart/form-data) |
| POST | `/generate-quiz/youtube` | Quiz from YouTube video (JSON body) |
| POST | `/submit-quiz` | Submit answers → score + suggestions (JSON body) |

***

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your OpenAI API key
export OPENAI_API_KEY="sk-..."

# 3. Start the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Interactive docs available at:
- Swagger UI → http://localhost:8000/docs
- ReDoc       → http://localhost:8000/redoc

***

## Usage Examples

### 1. Topic-based quiz (JSON)
```bash
curl -X POST http://localhost:8000/generate-quiz/topic \
  -H "Content-Type: application/json" \
  -d '{"topic": "Python Decorators", "difficulty": "Medium", "num_questions": 10}'
```

### 2. PDF-based quiz (multipart/form-data)
```bash
curl -X POST http://localhost:8000/generate-quiz/pdf \
  -F "file=@/path/to/document.pdf" \
  -F "difficulty=Hard" \
  -F "num_questions=15"
```

### 3. YouTube-based quiz (JSON)
```bash
curl -X POST http://localhost:8000/generate-quiz/youtube \
  -H "Content-Type: application/json" \
  -d '{"youtube_url": "https://www.youtube.com/watch?v=VIDEO_ID", "difficulty": "Easy", "num_questions": 10}'
```

### 4. Submit quiz answers
```bash
curl -X POST http://localhost:8000/submit-quiz \
  -H "Content-Type: application/json" \
  -d '{
    "quiz": { ...quiz object from any generate endpoint... },
    "user_answers": {"1": "A", "2": "C", "3": "B", "4": "D"}
  }'
```

***

## Python client example

```python
import requests

BASE = "http://localhost:8000"

# Step 1: Generate quiz
quiz_resp = requests.post(f"{BASE}/generate-quiz/topic", json={
    "topic": "Machine Learning Basics",
    "difficulty": "Medium",
    "num_questions": 10
})
quiz = quiz_resp.json()
print(f"Quiz: {quiz['title']} — {quiz['total_questions']} questions")

# Step 2: (user answers go here in your app)
user_answers = {q["question_id"]: "A" for q in quiz["questions"]}  # dummy answers

# Step 3: Submit and get score
result_resp = requests.post(f"{BASE}/submit-quiz", json={
    "quiz": quiz,
    "user_answers": user_answers
})
result = result_resp.json()
print(f"Score: {result['score_percentage']}%")
print(f"Suggestions: {result['improvement_suggestions']}")
```

***

## Response Schemas

### Quiz
```json
{
  "title": "string",
  "total_questions": 10,
  "questions": [
    {
      "question_id": 1,
      "question_text": "string",
      "options": [
        {"option_letter": "A", "option_text": "string"},
        {"option_letter": "B", "option_text": "string"},
        {"option_letter": "C", "option_text": "string"},
        {"option_letter": "D", "option_text": "string"}
      ],
      "correct_answer": "B",
      "explanation": "string",
      "difficulty": "Medium"
    }
  ]
}
```

### QuizResult
```json
{
  "total_questions": 10,
  "correct_answers": 7,
  "score_percentage": 70.0,
  "user_answers": [
    {
      "question_id": 1,
      "selected_answer": "A",
      "is_correct": true,
      "correct_answer": "A"
    }
  ],
  "improvement_suggestions": [
    "Review backpropagation concepts in Chapter 3",
    "Practice gradient descent problems"
  ]
}
```
