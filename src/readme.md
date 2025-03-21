# Пайплайн

### 1. Инициализация класса
Метод __init__:

Загружается модель SentenceTransformer для создания embedding векторов. Эти вектора используются для поиска похожих фрагментов текста в базе данных.
Инициализируется клиент Qdrant с базой данных, работающей исключительно в оперативной памяти (:memory:).
Путь к markdown-документации (список ссылок в self.md_list указанных URL-адресов документов) заранее прописан.
Каждая ссылка в списке указывает на файл с описанием определенного аспекта работы Qdrant.

### 2. Метод extract_text_from_md — Извлечение текста из Markdown
Этот метод выполняет:

Загружает содержимое markdown-файла с указанного URL.
Преобразует содержимое из формата markdown в HTML с использованием библиотеки markdown.
Извлекает чистый текст из HTML-файла с помощью BeautifulSoup.
Преобразует текст в список параграфов:
Параграфы выделяются с помощью регулярных выражений (например, разделение блоков текста по "пустым строкам").
Текст очищается (убираются лишние пробелы).
Короткие фрагменты (меньше 5 слов) отбрасываются.
Параграфы дополнительно разделяются на предложения (например, по точкам, знакам вопросов и восклицания).
Группируются по два предложения для большей связности.
Метод возвращает список очищенных фрагментов документа.

### 3. Метод initialize_database — Инициализация базы данных
Этот метод выполняет несколько шагов для подготовки Qdrant к работе:

Удаляет существующую коллекцию (если коллекция "documentation" уже существует в базе данных, она удаляется, чтобы избавиться от возможных конфликтов или устаревших данных).
Создает новую коллекцию в Qdrant documentation с параметрами:
size=1024: Это размерность векторов (определяется моделью для SentenceTransformer).
distance=models.Distance.COSINE: Используется косинусное расстояние для поиска наиболее похожих векторов.
Извлекает и обрабатывает данные:
Загружает все markdown-файлы из self.md_list.
Использует метод extract_text_from_md, чтобы извлечь параграфы из текстов.
Исключает отдельные специфичные фрагменты из каждого документа (например, заголовки, содержащие ссылки на текущую статью).
Загружает обработанные данные в Qdrant:
Преобразует каждый текстовый фрагмент в эмбеддинг (embedding) с использованием модели SentenceTransformer.
Формирует структуру векторов и метаданных (id, вектор, текст, имя документа).
Загружает данные в коллекцию Qdrant партиями по 100 элементов.

### 4. Метод get_answer — Получение ответа на пользовательский вопрос
Для ответа на вопрос пользователя реализован следующий алгоритм:

Ищутся фрагменты текста, похожие на заданный вопрос:
Метод search_similar_paragraphs преобразует вопрос в его embedding (векторное представление).
Сравнивает embedding вопроса со всеми embedding текстов в Qdrant.
Возвращает 3 наиболее релевантных фрагмента из документации с их похожестью (similarity) и названием документа.
Если модель GPT4All еще не была загружена, она инициализируется.
Генерация финального ответа:
Формируется контекстный текст из найденных фрагментов (добавляются их оценки схожести и источник из документации).
Передается в модель GPT4All вместе с вопросом.
Генерируется ответ с использованием этого контекста.

### 5. Метод search_similar_paragraphs — Поиск похожих фрагментов документации
Этот метод работает с коллекцией Qdrant:

Преобразует текст пользовательского запроса в вектор (с помощью SentenceTransformer).
Выполняет поиск в Qdrant по косинусному расстоянию между векторами.
Возвращает top-k (по умолчанию 3) наиболее похожих текстов, их названия и оценки схожести.

### 6. Метод generate_answer_gpt4all — Генерация ответа на основе контекста
Этот метод генерирует текст ответа с использованием GPT4All:

Формирует конкретный промпт для модели:
Включает сам вопрос пользователя.
Добавляет найденные фрагменты документации в качестве контекста.
Генерирует текстовый ответ (с регулировкой параметров модели, таких как температура, длина ответа и др.).
Возвращает результат пользователю.
Если возникнут ошибки (например, неверный путь к GPT4All модели или ошибки генерации), возвращается сообщение об ошибке.

### 7. Основная архитектура
Класс DocumentationQA берет на себя весь процесс — от загрузки документации до поиска ответов.
Используются два ключевых инструмента:
Qdrant для хранения векторных представлений и быстрого поиска по текстовым данным.
GPT4All для генерации человеческоподобных ответов на основе похожих фрагментов.
Код можно адаптировать под любую текстовую документацию (не обязательно markdown), добавив URL-адреса в self.md_list.
