from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer, CrossEncoder
import requests
import markdown
from bs4 import BeautifulSoup
import re
from typing import List, Dict, Tuple, Any, Optional
from abc import ABC, abstractmethod
from gpt4all import GPT4All


# Абстрактный класс для реранкеров
class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = None) -> List[Dict[str, Any]]:
        """Переранжирует документы на основе запроса."""
        pass
    
    @abstractmethod
    def get_reranker_name(self) -> str:
        """Возвращает имя реранкера."""
        pass


# Реализация без реранкера (проходной реранкер)
class NoReranker(Reranker):
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = None) -> List[Dict[str, Any]]:
        """Возвращает документы без изменения порядка."""
        if top_k is not None and top_k < len(documents):
            return documents[:top_k]
        return documents
    
    def get_reranker_name(self) -> str:
        return "No-Reranker"


# Реранкер на основе CrossEncoder
class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str):
        self.model_name = model_name
        
        # Настройки для токенизатора перед созданием модели
        self.max_length = 512  # Стандартный предел для большинства моделей
        
        # Создаем модель с указанием максимальной длины для токенизатора
        self.model = CrossEncoder(
            model_name,
            max_length=self.max_length,  # Устанавливаем максимальную длину при инициализации
            device=None  # Автоматический выбор устройства
        )
    
    def truncate_text(self, text: str, max_chars: int = 1500) -> str:
        """Обрезает текст до заданного количества символов."""
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "..."
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = None) -> List[Dict[str, Any]]:
        """Переранжирует документы с использованием CrossEncoder."""
        if not documents:
            return []
        
        # Подготовка пар (запрос, документ) для оценки, обрезая длинные тексты
        pairs = []
        for doc in documents:
            # Обрезаем длинные документы, чтобы избежать проблем с токенизацией
            truncated_text = self.truncate_text(doc['text'])
            pairs.append((query, truncated_text))
        
        # Получение оценок
        scores = self.model.predict(
            pairs,
            show_progress_bar=False
        )
        
        # Обновление оценок в документах
        for i, score in enumerate(scores):
            documents[i]['rerank_score'] = float(score)
        
        # Сортировка по новым оценкам
        reranked_docs = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
        
        # Ограничение количества результатов, если нужно
        if top_k is not None and top_k < len(reranked_docs):
            return reranked_docs[:top_k]
        return reranked_docs
    
    def get_reranker_name(self) -> str:
        return f"CrossEncoder-{self.model_name.split('/')[-1]}"


class ArticleDownloader:
    """Класс для скачивания статей из документации."""
    
    def __init__(self):
        self.md_list = [
            'https://raw.githubusercontent.com/qdrant/landing_page/master/qdrant-landing/content/documentation/concepts/collections.md',
            'https://raw.githubusercontent.com/qdrant/landing_page/master/qdrant-landing/content/documentation/concepts/explore.md',
            'https://raw.githubusercontent.com/qdrant/landing_page/master/qdrant-landing/content/documentation/concepts/filtering.md',
            'https://raw.githubusercontent.com/qdrant/landing_page/master/qdrant-landing/content/documentation/concepts/hybrid-queries.md',
            'https://raw.githubusercontent.com/qdrant/landing_page/master/qdrant-landing/content/documentation/concepts/indexing.md',
            'https://raw.githubusercontent.com/qdrant/landing_page/master/qdrant-landing/content/documentation/concepts/optimizer.md',
            'https://raw.githubusercontent.com/qdrant/landing_page/master/qdrant-landing/content/documentation/concepts/payload.md',
            'https://raw.githubusercontent.com/qdrant/landing_page/master/qdrant-landing/content/documentation/concepts/search.md',
            'https://raw.githubusercontent.com/qdrant/landing_page/master/qdrant-landing/content/documentation/concepts/points.md',
            'https://raw.githubusercontent.com/qdrant/landing_page/master/qdrant-landing/content/documentation/concepts/snapshots.md',
            'https://raw.githubusercontent.com/qdrant/landing_page/master/qdrant-landing/content/documentation/concepts/storage.md',
            'https://raw.githubusercontent.com/qdrant/landing_page/master/qdrant-landing/content/documentation/concepts/vectors.md'
        ]
    
    def download_article(self, url: str) -> str:
        """Скачивает статью по указанному URL."""
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    
    def get_all_articles(self) -> List[Dict[str, str]]:
        """Скачивает все статьи из списка и возвращает их с метаданными."""
        articles = []
        for url in self.md_list:
            content = self.download_article(url)
            name = url.split('concepts/')[1].split('.md')[0]
            articles.append({
                'name': name,
                'url': url,
                'content': content
            })
        return articles


class TextPreprocessor:
    """Класс для предобработки текстов статей."""
    
    def preprocess_markdown(self, markdown_text: str) -> str:
        """Конвертирует markdown в обычный текст."""
        html_content = markdown.markdown(markdown_text)
        soup = BeautifulSoup(html_content, features="html.parser")
        return soup.get_text()
    
    def remove_code_blocks(self, markdown_text: str) -> str:
        """Удаляет блоки кода из markdown-документа.
        
        Обрабатывает как блоки кода в формате ```language ... ``` или в формате с отступами.
        """
        # Удаление блоков кода в формате ```code```
        text_without_fenced_code = re.sub(
            r'```[\s\S]*?```', 
            ' ', 
            markdown_text
        )
        
        # Удаление однострочных блоков кода в формате `code`
        text_without_inline_code = re.sub(
            r'`.*?`', 
            ' ', 
            text_without_fenced_code
        )
        
        # Удаление блоков кода с отступами (строки, начинающиеся с 4+ пробелов или табуляции)
        lines = text_without_inline_code.split('\n')
        in_code_block = False
        result_lines = []
        
        for line in lines:
            if re.match(r'^( {4,}|\t)', line):
                # Это строка кода с отступом
                if not in_code_block:
                    in_code_block = True
            else:
                # Это не строка кода
                if in_code_block:
                    in_code_block = False
                result_lines.append(line)
        
        # Объединяем строки обратно в текст
        cleaned_text = '\n'.join(result_lines)
        
        # Удаляем возможные оставшиеся пустые участки после удаления кода
        cleaned_text = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_text)
        
        return cleaned_text
    
    def split_into_paragraphs(self, text: str, max_characters: int = 1500, 
                              new_after_n_chars: int = 1000, overlap: int = 0) -> List[str]:
        """Разделяет текст на параграфы с учетом заданных ограничений."""
        # Разделяем на смысловые элементы
        raw_paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        
        paragraphs = []
        current_chunk = ""
        
        for p in raw_paragraphs:
            # Нормализуем пробелы
            cleaned_p = re.sub(r'\s+', ' ', p).strip()
            
            # Пропускаем слишком короткие фрагменты
            if len(cleaned_p.split()) < 5:
                continue
                
            # Определяем, является ли текущий параграф заголовком
            is_title = len(cleaned_p.split()) < 10 and not cleaned_p.endswith(('.', '?', '!'))
            
            # Если новый параграф - заголовок или текущий чанк станет слишком большим
            if is_title or len(current_chunk) + len(cleaned_p) > new_after_n_chars:
                # Сохраняем предыдущий чанк, если он не пустой
                if current_chunk:
                    paragraphs.append(current_chunk)
                    current_chunk = ""
            
            # Если параграф слишком большой, разбиваем его на части
            if len(cleaned_p) > max_characters:
                # Разбиваем на предложения
                sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', cleaned_p)
                
                sentence_chunk = ""
                for sentence in sentences:
                    if len(sentence_chunk) + len(sentence) > max_characters:
                        paragraphs.append(sentence_chunk)
                        # Добавляем перекрытие, если задано
                        if overlap > 0:
                            words = sentence_chunk.split()
                            overlap_text = ' '.join(words[-min(len(words), overlap//5):])
                            sentence_chunk = overlap_text + " " + sentence
                        else:
                            sentence_chunk = sentence
                    else:
                        sentence_chunk = (sentence_chunk + " " + sentence).strip() if sentence_chunk else sentence
                
                if sentence_chunk:
                    paragraphs.append(sentence_chunk)
            else:
                # Добавляем параграф к текущему чанку
                current_chunk = (current_chunk + "\n\n" + cleaned_p).strip() if current_chunk else cleaned_p
                
                # Если чанк превысил максимальный размер, сохраняем его
                if len(current_chunk) > max_characters:
                    paragraphs.append(current_chunk)
                    current_chunk = ""
        
        # Добавляем последний чанк, если он не пустой
        if current_chunk:
            paragraphs.append(current_chunk)
        
        return paragraphs
    
    def extract_text_from_md(self, markdown_text: str, max_characters: int = 1500, 
                            new_after_n_chars: int = 1000, overlap: int = 0) -> List[str]:
        """Объединяет предобработку markdown и разделение на параграфы."""
        # Сначала удаляем блоки кода
        text_without_code = self.remove_code_blocks(markdown_text)
        
        # Затем конвертируем в обычный текст
        plain_text = self.preprocess_markdown(text_without_code)
        
        # И разделяем на параграфы
        return self.split_into_paragraphs(plain_text, max_characters, new_after_n_chars, overlap)
    
    def filter_paragraphs(self, paragraphs: List[str], name: str) -> List[str]:
        """Фильтрует параграфы, удаляя служебную информацию."""
        if name == 'collections':
            return [p for p in paragraphs if '/ Collections' not in p]
        else:
            return [p for p in paragraphs if f'/{name}' not in p]


class EmbeddingManager:
    """Класс для работы с эмбеддингами."""
    
    def __init__(self):
        self.model = SentenceTransformer('BAAI/bge-large-en-v1.5')
    
    def compute_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Вычисляет эмбеддинги для списка текстов."""
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return [embedding.tolist() for embedding in embeddings]
    
    def compute_single_embedding(self, text: str) -> List[float]:
        """Вычисляет эмбеддинг для одного текста."""
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
    
    def get_vector_size(self):
        """Возвращает размерность векторов для текущей модели."""
        return self.model.get_sentence_embedding_dimension()


class QdrantManager:
    """Класс для работы с базой Qdrant."""
    
    def __init__(self):
        self.client = QdrantClient(":memory:")
        self.collection_name = "documentation"
    
    def initialize_collection(self, vector_size: int = 1024):
        """Инициализирует коллекцию в Qdrant."""
        # Удаление существующей коллекции, если она есть
        try:
            self.client.delete_collection(collection_name=self.collection_name)
        except Exception:
            pass
        
        # Создание новой коллекции
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            )
        )
    
    def upsert_batch(self, id_offset: int, vectors: List[List[float]], 
                    payloads: List[Dict[str, Any]]) -> None:
        """Добавляет партию данных в Qdrant."""
        points = [
            models.PointStruct(
                id=idx + id_offset,
                vector=vector,
                payload=payload
            )
            for idx, (vector, payload) in enumerate(zip(vectors, payloads))
        ]
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
    
    def search(self, query_vector: List[float], limit: int = 3, 
              threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Выполняет поиск похожих документов в Qdrant."""
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=threshold
        )
        
        return [
            {
                'text': hit.payload['text'],
                'name': hit.payload['name'],
                'score': hit.score
            }
            for hit in search_result
        ]


class AnswerGenerator:
    """Класс для генерации ответов на основе контекста."""
    
    def __init__(self, top_k: int = 3):
        self.model = None
        self.top_k = top_k  # Количество фрагментов для использования
    
    def initialize_model(self):
        """Инициализирует модель для генерации ответов."""
        if self.model is None:
            self.model = GPT4All("orca-mini-3b-gguf2-q4_0.gguf")
    
    def set_top_k(self, top_k: int):
        """Устанавливает количество фрагментов для использования."""
        self.top_k = top_k
    
    def generate_answer(self, question: str, context_list: List[Dict[str, Any]]) -> str:
        """Генерирует ответ на основе вопроса и контекста."""
        self.initialize_model()
        
        # Ограничиваем число фрагментов для использования
        context_list = context_list[:self.top_k]
        
        # Формирование контекста для генерации ответа
        context_texts = "\n\n".join([
            f"Fragment {i+1} (from {doc.get('name', 'unknown')}, similarity: {doc.get('rerank_score', doc.get('score', 0)):.3f}):\n{doc['text']}"
            for i, doc in enumerate(context_list)
        ])
        
        prompt = f"""Based on the following context, please answer the question accurately and concisely.
                    Question: {question}
                    Context:
                    {context_texts}
                    Answer: """
        
        try:
            response = self.model.generate(
                prompt, 
                max_tokens=500,
                temp=0.7,
                top_k=40,
                top_p=0.4
            )
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"


class DocumentationQA:
    """Главный класс, объединяющий все компоненты системы."""
    
    def __init__(self, reranker: Optional[Reranker] = None):
        self.article_downloader = ArticleDownloader()
        self.text_preprocessor = TextPreprocessor()
        self.embedding_manager = EmbeddingManager()
        self.qdrant_manager = QdrantManager()
        # Если реранкер не передан, используем NoReranker
        self.reranker = reranker or NoReranker()
        self.answer_generator = AnswerGenerator(top_k=1)  # По умолчанию используем один фрагмент
        self.is_initialized = False
    
    def set_reranker(self, reranker: Reranker) -> None:
        """Устанавливает реранкер."""
        self.reranker = reranker
    
    def set_answer_top_k(self, top_k: int) -> None:
        """Устанавливает количество фрагментов для генерации ответа."""
        self.answer_generator.set_top_k(top_k)
    
    def initialize_database(self):
        """Инициализирует базу данных, скачивая статьи и добавляя их в Qdrant."""
        if self.is_initialized:
            return
        
        # Инициализация коллекции Qdrant
        self.qdrant_manager.initialize_collection(vector_size=self.embedding_manager.get_vector_size())
        
        # Скачивание и обработка статей
        articles = self.article_downloader.get_all_articles()
        
        # Обработка всех документов
        doc_paragraphs = []
        for article in articles:
            # Извлечение параграфов из markdown
            paragraphs = self.text_preprocessor.extract_text_from_md(article['content'])
            
            # Фильтрация специфичных заголовков
            filtered_paragraphs = self.text_preprocessor.filter_paragraphs(paragraphs, article['name'])
            
            # Добавление в общий список
            for paragraph in filtered_paragraphs:
                doc_paragraphs.append({
                    'name': article['name'],
                    'text': paragraph
                })
        
        # Добавление документов в Qdrant партиями
        batch_size = 100
        for i in range(0, len(doc_paragraphs), batch_size):
            batch = doc_paragraphs[i:i + batch_size]
            texts = [item['text'] for item in batch]
            
            # Вычисление эмбеддингов
            embeddings = self.embedding_manager.compute_embeddings(texts)
            
            # Подготовка payload
            payloads = [
                {
                    'text': item['text'],
                    'name': item['name']
                }
                for item in batch
            ]
            
            # Добавление в Qdrant
            self.qdrant_manager.upsert_batch(i, embeddings, payloads)
        
        self.is_initialized = True
    
    def search_similar_paragraphs(self, user_query: str, top_k: int = 3, first_stage_k: int = None) -> List[Dict[str, Any]]:
        """Ищет параграфы, похожие на запрос пользователя, с применением реранкинга."""
        # Определяем, сколько документов получить на первом этапе
        if first_stage_k is None:
            # По умолчанию берем больше документов для реранкинга, если используется реранкер
            first_stage_k = top_k * 3 if not isinstance(self.reranker, NoReranker) else top_k
        
        # Вычисление эмбеддинга для запроса
        query_vector = self.embedding_manager.compute_single_embedding(user_query)
        
        # Поиск через Qdrant (первый этап)
        first_stage_results = self.qdrant_manager.search(query_vector, limit=first_stage_k)
        
        # Применяем реранкинг (второй этап)
        reranked_results = self.reranker.rerank(user_query, first_stage_results, top_k=top_k)
        
        return reranked_results
    
    def get_answer(self, user_question: str) -> str:
        """Отвечает на вопрос пользователя."""
        # Инициализация базы данных, если еще не инициализирована
        if not self.is_initialized:
            self.initialize_database()
        
        # Поиск похожих параграфов с реранкингом
        similar_paragraphs = self.search_similar_paragraphs(user_question)
        
        # Генерация ответа
        answer = self.answer_generator.generate_answer(user_question, similar_paragraphs)
        return answer
    