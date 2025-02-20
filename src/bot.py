import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from pipeline_file import DocumentationQA

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Инициализация QA системы
qa_system = DocumentationQA()
qa_system.initialize_database()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        'Привет! Я бот, который поможет ответить на вопросы о Qdrant. Просто задайте свой вопрос!'
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('Задайте мне вопрос о Qdrant, и я постараюсь на него ответить.')

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_question = update.message.text
    
    # Отправка сообщения "печатает..."
    await update.message.reply_chat_action("typing")
    
    try:
        # Получение ответа от QA системы
        answer = qa_system.get_answer(user_question)
        
        if answer:
            await update.message.reply_text(answer)
        else:
            await update.message.reply_text(
                "Извините, не удалось сгенерировать ответ. Попробуйте переформулировать вопрос."
            )
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        await update.message.reply_text(
            "Произошла ошибка при обработке вашего вопроса. Пожалуйста, попробуйте позже."
        )

def main() -> None:
    # Создание приложения с использованием токена бота
    application = Application.builder().token("7560295068:AAET7OEVGbUfVaSEGgr7VMWvEpqnr20ai84").build()

    # Добавление обработчиков
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Запуск бота
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()