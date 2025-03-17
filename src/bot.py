import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Импортируем необходимые классы из нашего pipeline_file.py
from pipeline_file import DocumentationQA, ArticleDownloader, TextPreprocessor, EmbeddingManager, QdrantManager, AnswerGenerator

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Инициализация QA системы с новой структурой
qa_system = DocumentationQA()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /start."""
    # Логируем информацию о пользователе
    user = update.effective_user
    logger.info(f"Пользователь {user.id} ({user.username}) запустил бота")
    
    await update.message.reply_text(
        'Привет! Я бот, который поможет ответить на вопросы о Qdrant. '
        'Просто задайте свой вопрос!\n\n'
        'База данных документации будет инициализирована при первом запросе.'
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /help."""
    await update.message.reply_text(
        'Задайте мне вопрос о Qdrant, и я постараюсь на него ответить.\n\n'
        'Я использую векторную базу данных Qdrant для поиска релевантных '
        'фрагментов документации и генерирую ответы на их основе.'
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик текстовых сообщений."""
    user = update.effective_user
    user_question = update.message.text
    
    logger.info(f"Получен вопрос от пользователя {user.id}: {user_question}")
    
    # Отправка сообщения "печатает..."
    await update.message.reply_chat_action("typing")
    
    try:
        # Проверяем, инициализирована ли база данных
        if not qa_system.is_initialized:
            await update.message.reply_text("Инициализирую базу данных документации, это может занять некоторое время...")
        
        # Получение ответа от QA системы
        # Теперь get_answer сам проверяет инициализацию и вызывает initialize_database при необходимости
        answer = qa_system.get_answer(user_question)
        
        if answer:
            logger.info(f"Отправлен ответ пользователю {user.id}")
            await update.message.reply_text(answer)
        else:
            logger.warning(f"Не удалось сгенерировать ответ для пользователя {user.id}")
            await update.message.reply_text(
                "Извините, не удалось сгенерировать ответ. Попробуйте переформулировать вопрос."
            )
    except Exception as e:
        logger.error(f"Ошибка обработки сообщения от пользователя {user.id}: {e}")
        await update.message.reply_text(
            "Произошла ошибка при обработке вашего вопроса. Пожалуйста, попробуйте позже."
        )

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда для проверки статуса системы."""
    status_message = (
        f"Статус системы:\n"
        f"База данных инициализирована: {qa_system.is_initialized}\n"
        f"Модель эмбеддингов: intfloat/multilingual-e5-large\n"
        f"Количество статей в базе: {len(qa_system.article_downloader.md_list)}"
    )
    await update.message.reply_text(status_message)

def main() -> None:
    """Точка входа в программу."""
    # Создание приложения с использованием токена бота
    application = Application.builder().token("7560295068:AAET7OEVGbUfVaSEGgr7VMWvEpqnr20ai84").build()

    # Добавление обработчиков
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Логирование запуска бота
    logger.info("Бот запущен. Ожидание сообщений...")
    
    # Запуск бота
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()