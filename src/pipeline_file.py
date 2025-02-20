from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from gpt4all import GPT4All
import requests
import markdown
from bs4 import BeautifulSoup
import re

class DocumentationQA:
    def __init__(self):
        self.model = SentenceTransformer('intfloat/multilingual-e5-large')
        self.qdrant_client = QdrantClient(":memory:")
        self.gpt4all_model = None
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

    def extract_text_from_md(self, url):
        response = requests.get(url)
        response.raise_for_status()
        html_content = markdown.markdown(response.text)
        soup = BeautifulSoup(html_content, features="html.parser")
        text = soup.get_text()

        # Разбиваем текст на абзацы
        raw_paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

        paragraphs = []
        for p in raw_paragraphs:
            # Очищаем текст от лишних пробелов
            cleaned_p = re.sub(r'\s+', ' ', p).strip()

            # Пропускаем короткие фрагменты
            if len(cleaned_p.split()) < 5:
                continue

            # Разбиваем на предложения
            # Учитываем сокращения Mr., Dr., т.д., чтобы не разбивать их
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', cleaned_p)

            # Группируем предложения по 2
            for i in range(0, len(sentences), 2):
                if i + 1 < len(sentences):
                    # Берем два предложения
                    fragment = ' '.join(sentences[i:i+2]).strip()
                else:
                    # Если осталось одно предложение
                    fragment = sentences[i].strip()

                # Проверяем минимальную длину фрагмента
                if len(fragment.split()) >= 5:
                    paragraphs.append(fragment)

        return paragraphs

    def initialize_database(self):
        # Удаление существующей коллекции, если она есть
        try:
            self.qdrant_client.delete_collection(collection_name="documentation")
        except Exception:
            pass

        # Создание новой коллекции в Qdrant
        self.qdrant_client.create_collection(
            collection_name="documentation",
            vectors_config=models.VectorParams(
                size=1024,
                distance=models.Distance.COSINE
            )
        )

        # Обработка всех документов
        doc_paragraphs = []
        for url in self.md_list:
            paragraphs = self.extract_text_from_md(url)
            name = url.split('concepts/')[1].split('.md')[0]

            if name == 'collections':
                paragraphs = [p for p in paragraphs if '/ Collections' not in p]
            else:
                paragraphs = [p for p in paragraphs if f'/{name}' not in p]

            for paragraph in paragraphs:
                doc_paragraphs.append({
                    'name': name,
                    'text': paragraph
                })

        # Создание эмбеддингов и загрузка в Qdrant
        batch_size = 100
        for i in range(0, len(doc_paragraphs), batch_size):
            batch = doc_paragraphs[i:i + batch_size]
            texts = [item['text'] for item in batch]
            embeddings = self.model.encode(texts, normalize_embeddings=True)

            points = [
                models.PointStruct(
                    id=idx + i,
                    vector=embedding.tolist(),
                    payload={
                        'text': item['text'],
                        'name': item['name']
                    }
                )
                for idx, (item, embedding) in enumerate(zip(batch, embeddings))
            ]

            self.qdrant_client.upsert(
                collection_name="documentation",
                points=points
            )

    def get_answer(self, user_question):
        similar_paragraphs = self.search_similar_paragraphs(user_question)
        context_texts = "\n\n".join([
            f"Fragment {i+1} (from {doc_name}, similarity: {score:.3f}):\n{text}"
            for i, (text, doc_name, score) in enumerate(similar_paragraphs)
        ])
        
        if self.gpt4all_model is None:
            self.gpt4all_model = GPT4All("orca-mini-3b-gguf2-q4_0.gguf")
            
        answer = self.generate_answer_gpt4all(user_question, context_texts)
        return answer

    def search_similar_paragraphs(self, user_query, top_k=3):
        query_vector = self.model.encode(user_query, normalize_embeddings=True)
        search_result = self.qdrant_client.search(
            collection_name="documentation",
            query_vector=query_vector.tolist(),
            limit=top_k,
            score_threshold=0.5
        )
        return [(hit.payload['text'], hit.payload['name'], hit.score) for hit in search_result]

    def generate_answer_gpt4all(self, user_question, relevant_texts):
        prompt = f"""Based on the following context, please answer the question accurately and concisely.
                    Question: {user_question}
                    Context:
                    {relevant_texts}
                    Answer: """

        try:
            response = self.gpt4all_model.generate(prompt, 
                                            max_tokens=500,
                                            temp=0.7,
                                            top_k=40,
                                            top_p=0.4)
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"