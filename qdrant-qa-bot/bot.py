import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import os
import requests
import json

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# URL API сервиса
API_URL = os.environ.get("API_URL", "http://api:8000")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /start."""
    # Логируем информацию о пользователе
    user = update.effective_user
    logger.info(f"Пользователь {user.id} ({user.username}) запустил бота")
    
    await update.message.reply_text(
        'Hello! I am a bot that will help you with your Qdrant questions.'
        'Just ask your question!\n\n'
        'Documentation database will be initialized on first request.'
    )
async def generation_params_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда для отображения текущих параметров генерации."""
    try:
        response = requests.get(f"{API_URL}/generation-params")
        if response.status_code == 200:
            params = response.json()
            params_message = (
                f"Текущие параметры генерации ответов:\n"
                f"- Модель: {params['model_name']}\n"
                f"- Максимальное количество токенов: {params['max_tokens']}\n"
                f"- Температура: {params['temperature']}\n"
                f"- Top-K: {params['top_k']}\n"
                f"- Top-P: {params['top_p']}\n\n"
                f"Для изменения параметров используйте команду /set_optimal_gen"
            )
        else:
            params_message = f"Не удалось получить параметры генерации. Код ошибки: {response.status_code}"
    except Exception as e:
        params_message = f"Ошибка при запросе параметров генерации: {str(e)}"
        
    await update.message.reply_text(params_message)

async def set_optimal_generation_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда для установки оптимальных параметров генерации."""
    try:
        # Используем оптимальные параметры из экспериментов
        params = {
            "model_name": "deepseek-coder-1.3b-instruct.Q4_K_M.gguf",
            "max_tokens": 500,
            "temperature": 0.0,
            "top_k": 40,
            "top_p": 0.0
        }
        
        response = requests.post(f"{API_URL}/set-generation-params", json=params)
        
        if response.status_code == 200:
            await update.message.reply_text(
                "Установлены оптимальные параметры генерации ответов:\n"
                f"- Модель: {params['model_name']}\n"
                f"- Максимальное количество токенов: {params['max_tokens']}\n"
                f"- Температура: {params['temperature']}\n"
                f"- Top-K: {params['top_k']}\n"
                f"- Top-P: {params['top_p']}\n\n"
                "Эти значения были определены как оптимальные в результате экспериментов."
            )
        else:
            await update.message.reply_text(f"Ошибка при установке параметров: {response.text}")
    except Exception as e:
        await update.message.reply_text(f"Ошибка при установке параметров: {str(e)}")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /help."""
    await update.message.reply_text(
        'Ask me a question about Qdrant and I will try to answer it.\n\n'
        'I use the Qdrant vector database to find relevant '
        'documentation snippets and generate answers based on them.\n\n'
        'Additional commands:\n'
        '/status - Check system status\n'
        '/clearcache - Clear semantic cache\n'
        '/cache_on - Enable semantic cache\n'
        '/cache_off - Disable semantic cache\n'
        '/hybrid_on - Enable hybrid search (better results)\n'
        '/hybrid_off - Disable hybrid search\n'
        '/optimal_weights - Set optimal weights for hybrid search\n'
        '/gen_params - Show current generation parameters\n'
        '/set_optimal_gen - Set optimal generation parameters'
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик текстовых сообщений."""
    user = update.effective_user
    user_question = update.message.text
    
    logger.info(f"Получен вопрос от пользователя {user.id}: {user_question}")
    
    # Отправка сообщения "печатает..."
    await update.message.reply_chat_action("typing")
    
    try:
        # Проверяем статус API
        status_response = requests.get(f"{API_URL}/status")
        status_data = status_response.json()
        
        if not status_data.get("is_initialized"):
            await update.message.reply_text("Initializing the documentation database, this may take some time...")
            # Инициируем инициализацию базы данных
            requests.post(f"{API_URL}/initialize")
        
        # Формируем запрос к API
        payload = {
            "text": user_question,
            "top_k": 3,  # Можно настроить количество источников
            "use_cache": True,  # Включаем использование кэша
            "use_hybrid": True  # Включаем гибридный поиск
        }
        
        # Отправляем запрос к API
        response = requests.post(f"{API_URL}/answer", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            answer = data["answer"]
            cached = data.get("cached", False)
            
            logger.info(f"Отправлен ответ пользователю {user.id} (из кэша: {cached})")
            
            # Добавляем информацию о кэше
            answer_text = answer
            if cached:
                answer_text = f"{answer_text}\n\n[⚡ Response from semantic cache]"
                
            await update.message.reply_text(answer_text)
        else:
            logger.warning(f"API вернул ошибку: {response.status_code} - {response.text}")
            await update.message.reply_text(
                "Sorry, we couldn't generate an answer. Try rephrase your question."
            )
    except Exception as e:
        logger.error(f"Ошибка обработки сообщения от пользователя {user.id}: {e}")
        await update.message.reply_text(
            "There was an error processing your question. Please try again later."
        )

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда для проверки статуса системы."""
    try:
        response = requests.get(f"{API_URL}/status")
        if response.status_code == 200:
            data = response.json()
            status_message = (
                f"Статус системы:\n"
                f"База данных инициализирована: {data['is_initialized']}\n"
                f"Модель эмбеддингов: {data['embedding_model']}\n"
                f"Количество статей в базе: {data['articles_count']}\n"
                f"Используется Qdrant в Docker: {data.get('using_docker_qdrant', 'N/A')}\n"
                f"Семантический кэш включен: {data.get('semantic_cache_enabled', 'N/A')}\n"
                f"Гибридный поиск включен: {data.get('hybrid_search_enabled', 'N/A')}"
            )
        else:
            status_message = f"Не удалось получить статус системы. Код ошибки: {response.status_code}"
    except Exception as e:
        status_message = f"Ошибка при запросе статуса: {str(e)}"
        
    await update.message.reply_text(status_message)

async def clear_cache_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда для очистки семантического кэша."""
    try:
        response = requests.post(f"{API_URL}/clear-cache")
        if response.status_code == 200:
            await update.message.reply_text("Семантический кэш успешно очищен.")
        else:
            await update.message.reply_text(f"Ошибка при очистке кэша: {response.text}")
    except Exception as e:
        await update.message.reply_text(f"Ошибка при очистке кэша: {str(e)}")

async def enable_cache_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда для включения семантического кэша."""
    try:
        response = requests.post(f"{API_URL}/toggle-cache", json={"enable": True})
        if response.status_code == 200:
            await update.message.reply_text("Семантический кэш включен.")
        else:
            await update.message.reply_text(f"Ошибка при включении кэша: {response.text}")
    except Exception as e:
        await update.message.reply_text(f"Ошибка при включении кэша: {str(e)}")

async def disable_cache_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда для отключения семантического кэша."""
    try:
        response = requests.post(f"{API_URL}/toggle-cache", json={"enable": False})
        if response.status_code == 200:
            await update.message.reply_text("Семантический кэш отключен.")
        else:
            await update.message.reply_text(f"Ошибка при отключении кэша: {response.text}")
    except Exception as e:
        await update.message.reply_text(f"Ошибка при отключении кэша: {str(e)}")

# Новые функции для управления гибридным поиском
async def enable_hybrid_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда для включения гибридного поиска."""
    try:
        response = requests.post(f"{API_URL}/toggle-hybrid", json={"enable": True})
        if response.status_code == 200:
            await update.message.reply_text("Гибридный поиск включен. Это должно улучшить качество ответов.")
        else:
            await update.message.reply_text(f"Ошибка при включении гибридного поиска: {response.text}")
    except Exception as e:
        await update.message.reply_text(f"Ошибка при включении гибридного поиска: {str(e)}")

async def disable_hybrid_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда для отключения гибридного поиска."""
    try:
        response = requests.post(f"{API_URL}/toggle-hybrid", json={"enable": False})
        if response.status_code == 200:
            await update.message.reply_text("Гибридный поиск отключен. Будет использоваться только плотный векторный поиск.")
        else:
            await update.message.reply_text(f"Ошибка при отключении гибридного поиска: {response.text}")
    except Exception as e:
        await update.message.reply_text(f"Ошибка при отключении гибридного поиска: {str(e)}")

async def set_optimal_weights(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда для установки оптимальных весов гибридного поиска."""
    try:
        # Используем оптимальные веса из экспериментов
        response = requests.post(
            f"{API_URL}/set-hybrid-weights", 
            params={"dense_weight": 0.8, "sparse_weight": 0.2}
        )
        
        if response.status_code == 200:
            await update.message.reply_text(
                "Установлены оптимальные веса для гибридного поиска:\n"
                "- Вес для плотных векторов: 0.8\n"
                "- Вес для разреженных векторов: 0.2\n"
                "Эти значения были определены как оптимальные в результате экспериментов."
            )
        else:
            await update.message.reply_text(f"Ошибка при установке весов: {response.text}")
    except Exception as e:
        await update.message.reply_text(f"Ошибка при установке весов: {str(e)}")


def main() -> None:
    """Точка входа в программу."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN не установлен в переменных окружения!")
    
    # Создание приложения с использованием токена бота
    application = Application.builder().token(token).build()

    # Добавление обработчиков
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("clearcache", clear_cache_command))
    application.add_handler(CommandHandler("cache_on", enable_cache_command))
    application.add_handler(CommandHandler("cache_off", disable_cache_command))
    
    # Добавляем новые обработчики для гибридного поиска
    application.add_handler(CommandHandler("gen_params", generation_params_command))
    application.add_handler(CommandHandler("set_optimal_gen", set_optimal_generation_command))
    application.add_handler(CommandHandler("hybrid_on", enable_hybrid_command))
    application.add_handler(CommandHandler("hybrid_off", disable_hybrid_command))
    application.add_handler(CommandHandler("optimal_weights", set_optimal_weights))
    
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Логирование запуска бота
    logger.info("Бот запущен. Ожидание сообщений...")
    
    # Запуск бота
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()