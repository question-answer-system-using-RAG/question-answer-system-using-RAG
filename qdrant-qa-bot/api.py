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
    use_cache: Optional[bool] = True

class Answer(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    cached: bool = False

class StatusResponse(BaseModel):
    is_initialized: bool
    embedding_model: str
    articles_count: int
    using_docker_qdrant: bool = True
    semantic_cache_enabled: bool = True


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
        
        # Настраиваем использование кэша
        qa_system.use_semantic_cache = question.use_cache
        
        # Проверяем, инициализирована ли база данных
        if not qa_system.is_initialized:
            qa_system.initialize_database()
        
        # Получаем ответ (может быть из кэша или сгенерирован)
        response = qa_system.get_answer(question.text)
        
        return Answer(
            answer=response['answer'], 
            sources=response.get('sources', []),
            cached=response.get('cached', False)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Эндпоинт для проверки статуса системы."""
    return StatusResponse(
        is_initialized=qa_system.is_initialized,
        embedding_model="BAAI/bge-large-en-v1.5",
        articles_count=len(qa_system.article_downloader.md_list),
        using_docker_qdrant=True,
        semantic_cache_enabled=qa_system.use_semantic_cache
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


@app.post("/clear-cache")
async def clear_cache():
    """Эндпоинт для очистки семантического кэша."""
    try:
        if qa_system.is_initialized:
            # Удаляем и создаем заново коллекцию кэша
            qa_system.qdrant_manager.client.delete_collection(
                collection_name=qa_system.qdrant_manager.cache_collection_name
            )
            
            # Создаем новую коллекцию для кэша
            qa_system.qdrant_manager.client.create_collection(
                collection_name=qa_system.qdrant_manager.cache_collection_name,
                vectors_config=models.VectorParams(
                    size=qa_system.embedding_manager.get_vector_size(),
                    distance=models.Distance.COSINE
                )
            )
            
            return {"status": "Cache cleared successfully"}
        return {"status": "Database not initialized yet"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing cache: {str(e)}")


@app.post("/toggle-cache")
async def toggle_cache(enable: bool):
    """Эндпоинт для включения/отключения семантического кэша."""
    qa_system.use_semantic_cache = enable
    return {"status": f"Semantic cache is now {'enabled' if enable else 'disabled'}"}


if __name__ == "__main__":
    # Получаем порт из переменной окружения или используем порт по умолчанию
    port = int(os.environ.get("API_PORT", 8000))
    # Запускаем сервер
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)