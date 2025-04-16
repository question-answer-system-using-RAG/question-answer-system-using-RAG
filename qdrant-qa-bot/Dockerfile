# Используем Python 3.10 как базовый образ
FROM python:3.10-slim

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Устанавливаем необходимые системные зависимости
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Копируем файл с зависимостями
COPY requirements.txt .

# Устанавливаем NumPy отдельно до других зависимостей
RUN pip install --no-cache-dir numpy

# Устанавливаем остальные зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Создаем директорию для моделей
RUN mkdir -p /app/models

# Копируем файлы приложения в контейнер
COPY pipeline_file.py bot.py ./

# Устанавливаем переменную окружения для хранения моделей gpt4all
ENV GPT4ALL_MODEL_PATH=/app/models

# Указываем порт, который будет использоваться для Telegram бота
EXPOSE 8443

# Запускаем бота
CMD ["python", "bot.py"]