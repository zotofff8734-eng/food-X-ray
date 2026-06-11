import logging
import os
import asyncio
from dotenv import load_dotenv
from pathlib import Path

from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
    ConversationHandler,
)

# Import utility functions
from yolo_inference import predict_image
from db_utils import get_dish_by_name # Later: add_dish
from stt import convert_oga_to_wav, transcribe_audio
from ocr import recognize_text_from_image

# --- Pre-run Checks & Setup ---
load_dotenv()
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN not found in .env file!")

# --- Global Setup ---
TEMP_DIR = Path("temp_uploads")
os.makedirs(TEMP_DIR, exist_ok=True)

# --- Conversation States ---
# Order defines the steps for adding a new dish
PHOTO, NAME, CALORIES, RECIPE, INGREDIENTS, CONFIRMATION = range(6)


# --- Helper Functions ---
async def show_main_menu(update: Update):
    keyboard = [
        [KeyboardButton("📸 Сфотографировать блюдо")],
        [KeyboardButton("➕ Добавить новое блюдо")],
        [KeyboardButton("📋 Мои блюда"), KeyboardButton("ℹ️ Помощь")],
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    await update.message.reply_text("Выберите действие:", reply_markup=reply_markup)


# --- Standard Command & Message Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(f"👋 Привет, {update.effective_user.first_name}!")
    await show_main_menu(update)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_text = (
        "<b>Как пользоваться ботом:</b>

"
        "📸 <b>Сфотографировать блюдо</b> - Отправьте фото еды для анализа.
"
        "➕ <b>Добавить новое блюдо</b> - Запустите процесс добавления вашего блюда в базу.
"
        "📋 <b>Мои блюда</b> - Посмотреть список добавленных вами блюд.

"
        "Чтобы отменить любое действие, введите команду /cancel."
    )
    await update.message.reply_html(help_text)


async def recognize_dish(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info(f"User {update.effective_user.name} sent a photo for recognition.")
    await update.message.reply_text("Анализирую фото... Это может занять некоторое время.")
    photo_file = await update.message.photo[-1].get_file()
    temp_file_path = TEMP_DIR / f"{photo_file.file_id}.jpg"
    try:
        await photo_file.download_to_drive(temp_file_path)
        predictions = predict_image(str(temp_file_path))
        if not predictions:
            await update.message.reply_text("К сожалению, не удалось распознать блюда на этом фото.")
            return
        response_parts = []
        for item in predictions:
            dish = await get_dish_by_name(item['name'])
            if dish:
                response_parts.append(
                    f"<b>{dish.name_ru}</b> ({item['confidence']:.0%})
"
                    f"🔥 Калории: {dish.calories or 'н/д'} ккал/100г
"
                    f"👍 Оценка: {dish.health_impact or 'н/д'}
"
                    f"📝 Рецепт: {dish.recipe or 'н/д'}"
                )
            else:
                response_parts.append(
                    f"✔️ Распознано: <b>{item['name']}</b> ({item['confidence']:.0%})
"
                    f"<i>(Информация еще не добавлена в базу.)</i>"
                )
        await update.message.reply_html(f"<b>Результаты анализа:</b>

" + "

".join(response_parts))
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


async def unhandled_button_or_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    button_text = update.message.text
    logger.warning(f"User {update.effective_user.name} sent unhandled text: {button_text}")
    if button_text == "📸 Сфотографировать блюдо":
        await update.message.reply_text("Пожалуйста, просто отправьте мне фотографию вашего блюда для анализа.")
    elif button_text == "📋 Мои блюда":
        await update.message.reply_text("Эта функция пока находится в разработке.")
    else:
        await update.message.reply_text("Я вас не понял. Пожалуйста, используйте кнопки меню или отправьте фото.")


# --- "Add Dish" Conversation Handlers ---
async def add_dish_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    logger.info(f"User {update.effective_user.name} starting 'add dish' conversation.")
    context.user_data['new_dish'] = {}
    await update.message.reply_text(
        "<b>Шаг 1/5:</b> Отправьте фото блюда.
Для отмены введите /cancel.",
        parse_mode='HTML'
    )
    return PHOTO

async def received_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    photo = update.message.photo[-1]
    context.user_data['new_dish']['photo_id'] = photo.file_id
    logger.info(f"Received photo for new dish from {update.effective_user.name}.")
    await update.message.reply_text(
        "<b>Шаг 2/5:</b> Теперь введите или надиктуйте название блюда.",
        parse_mode='HTML'
    )
    return NAME

async def received_name(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    dish_name = update.message.text
    context.user_data['new_dish']['name'] = dish_name
    logger.info(f"Received name '{dish_name}' for new dish.")
    await update.message.reply_text(
        f"Отлично! Название: '{dish_name}'.

"
        "<b>Шаг 3/5:</b> Укажите калорийность (число ккал на 100 г).",
        parse_mode='HTML'
    )
    return CALORIES

async def received_voice_for_name(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    logger.info(f"User {update.effective_user.name} sent a voice message for the dish name.")
    voice = update.message.voice
    oga_file_path = TEMP_DIR / f"{voice.file_id}.oga"
    wav_file_path = TEMP_DIR / f"{voice.file_id}.wav"
    
    try:
        voice_file = await voice.get_file()
        await voice_file.download_to_drive(oga_file_path)
        
        if not convert_oga_to_wav(str(oga_file_path), str(wav_file_path)):
            await update.message.reply_text("Не удалось обработать аудио. Пожалуйста, попробуйте еще раз или введите название текстом.")
            return NAME

        transcribed_text = transcribe_audio(str(wav_file_path))
        if not transcribed_text:
            await update.message.reply_text("Не удалось распознать речь. Пожалуйста, попробуйте еще раз или введите название текстом.")
            return NAME
        
        update.message.text = transcribed_text
        return await received_name(update, context)

    finally:
        if os.path.exists(oga_file_path):
            os.remove(oga_file_path)
        if os.path.exists(wav_file_path):
            os.remove(wav_file_path)


async def received_calories(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        calories = float(update.message.text)
        context.user_data['new_dish']['calories'] = calories
        logger.info(f"Received calories '{calories}'.")
        await update.message.reply_text(
            "<b>Шаг 4/5:</b> Отлично. Теперь отправьте рецепт.
"
            "Вы можете либо прислать фотографию с текстом, либо просто отправить текст.",
            parse_mode='HTML'
        )
        return RECIPE
    except ValueError:
        await update.message.reply_text("Нужно ввести число. Попробуйте еще раз.")
        return CALORIES

async def received_recipe_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data['new_dish']['recipe'] = update.message.text
    logger.info("Received recipe as text.")
    await update.message.reply_text(
        "Рецепт сохранен.

"
        "<b>Шаг 5/5:</b> Теперь введите список ингредиентов (одним сообщением).",
        parse_mode='HTML'
    )
    return INGREDIENTS

async def received_recipe_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    logger.info(f"User {update.effective_user.name} sent a photo for the recipe.")
    await update.message.reply_text("Распознаю текст с фото, это может занять время...")
    
    photo_file = await update.message.photo[-1].get_file()
    temp_file_path = TEMP_DIR / f"recipe_{photo_file.file_id}.jpg"

    try:
        await photo_file.download_to_drive(temp_file_path)
        recognized_text = recognize_text_from_image(str(temp_file_path))

        if recognized_text is None:
            await update.message.reply_text("Ошибка! Не удалось распознать текст. Попробуйте отправить фото лучшего качества или введите рецепт вручную.")
            return RECIPE
        if not recognized_text.strip():
            await update.message.reply_text("На фото не найден текст. Пожалуйста, введите рецепт вручную.")
            return RECIPE

        context.user_data['new_dish']['recipe'] = recognized_text
        
        preview = (recognized_text[:900] + '...') if len(recognized_text) > 900 else recognized_text
        await update.message.reply_text(
            f"<b>Я распознал следующий текст:</b>
<i>{preview}</i>

Рецепт сохранен.
<b>Шаг 5/5:</b> Теперь введите список ингредиентов (одним сообщением).",
            parse_mode='HTML'
        )
        return INGREDIENTS
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

async def received_ingredients(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data['new_dish']['ingredients'] = update.message.text
    logger.info("Received ingredients.")
    
    # Next step would be to show a confirmation and save to DB.
    # For now, we end the conversation.
    await update.message.reply_text(
        "Отлично! Все данные собраны.
"
        "Следующим шагом будет подтверждение и сохранение в базу данных.
"
        "Спасибо за участие! Диалог завершен.",
        parse_mode='HTML'
    )
    logger.info(f"Finished 'add dish' conversation for user {update.effective_user.name}. Data: {context.user_data['new_dish']}")
    context.user_data.clear()
    return ConversationHandler.END

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    logger.info(f"User {update.effective_user.name} canceled the conversation.")
    context.user_data.clear()
    await update.message.reply_text('Действие отменено.')
    await show_main_menu(update)
    return ConversationHandler.END


# --- Main Application ---
async def main() -> None:
    application = Application.builder().token(TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[MessageHandler(filters.Regex('^➕ Добавить новое блюдо$'), add_dish_start)],
        states={
            PHOTO: [MessageHandler(filters.PHOTO, received_photo)],
            NAME: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, received_name),
                MessageHandler(filters.VOICE, received_voice_for_name),
            ],
            CALORIES: [MessageHandler(filters.TEXT & ~filters.COMMAND, received_calories)],
            RECIPE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, received_recipe_text),
                MessageHandler(filters.PHOTO, received_recipe_photo),
            ],
            INGREDIENTS: [MessageHandler(filters.TEXT & ~filters.COMMAND, received_ingredients)],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
        per_user=True, per_chat=True,
    )

    application.add_handler(conv_handler)
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.Regex('^ℹ️ Помощь$'), help_command))
    application.add_handler(MessageHandler(filters.PHOTO & (~filters.UpdateType.EDITED_MESSAGE), recognize_dish))
    application.add_handler(MessageHandler(filters.Regex('^(📸 Сфотографировать блюдо|📋 Мои блюда)$'), unhandled_button_or_text))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, unhandled_button_or_text))

    logger.info("Starting bot polling...")
    await application.run_polling()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.fatal(f"Failed to start bot: {e}")
