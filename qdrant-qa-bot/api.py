from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import os

# Импортируем компоненты из pipeline_file
from pipeline_file import DocumentationQA

# Инициализация QA системы
qa_system = DocumentationQA()

app = FastAPI(title="Qdrant Documentation QA API", 
              description="API for answering questions about Qdrant using vector search",
              version="1.0.0")

class Question(BaseModel):
    text: str
    top_k: Optional[int] = 1

class Answer(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

class StatusResponse(BaseModel):
    is_initialized: bool
    embedding_model: str
    articles_count: int


@app.get("/")
async def root():
    """Корневой эндпоинт для проверки работы API."""
    return {"status": "API is running", "service": "Qdrant Documentation QA"}


@app.post("/answer", response_model=Answer)
async def get_answer(question: Question):
    """Эндпоинт для получения ответа на вопрос."""
    try:
        # Устанавливаем количество фрагментов, если оно указано
        if question.top_k:
            qa_system.set_answer_top_k(question.top_k)
        
        # Проверяем, инициализирована ли база данных
        if not qa_system.is_initialized:
            qa_system.initialize_database()
        
        # Получаем похожие параграфы
        similar_paragraphs = qa_system.search_similar_paragraphs(question.text)
        
        # Генерируем ответ
        answer_text = qa_system.answer_generator.generate_answer(question.text, similar_paragraphs)
        
        # Готовим источники для ответа
        sources = [
            {
                "text": p["text"],
                "name": p["name"],
                "score": p.get("rerank_score", p.get("score", 0))
            }
            for p in similar_paragraphs
        ]
        
        return Answer(answer=answer_text, sources=sources)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Эндпоинт для проверки статуса системы."""
    return StatusResponse(
        is_initialized=qa_system.is_initialized,
        embedding_model="BAAI/bge-large-en-v1.5",
        articles_count=len(qa_system.article_downloader.md_list)
    )


@app.post("/initialize")
async def initialize_database():
    """Эндпоинт для инициализации базы данных."""
    try:
        if not qa_system.is_initialized:
            qa_system.initialize_database()
            return {"status": "Database initialized successfully"}
        return {"status": "Database already initialized"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initializing database: {str(e)}")


if __name__ == "__main__":
    # Получаем порт из переменной окружения или используем порт по умолчанию
    port = int(os.environ.get("API_PORT", 8000))
    # Запускаем сервер
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)