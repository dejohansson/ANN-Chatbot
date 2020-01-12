from bot import Conversation

from telegram.ext import Updater, CommandHandler, Filters, MessageHandler

TOKEN = ""
conv = Conversation("models/bestModelBert.pt", "models/bestModelGPT2CONV.pt")

def start(update, context):
    update.message.reply_text("Hello")

def chat(update, context):
    answer, sentiment = conv.next_sentence(update.message.text[5:])
    context.bot.send_message(chat_id=update.message.chat_id, text=answer + " " + {sentiment > .9: ":)", sentiment < .1: ":("}.get(True, ""))

def reset(update, context):
    conv.reset()
    update.message.reply_text("Message history has been removed.")

def chat_direct(update, context):
    answer, sentiment = conv.next_sentence(update.message.text)
    context.bot.send_message(chat_id=update.message.chat_id, text=answer + " " + {sentiment > .9: ":)", sentiment < .1: ":("}.get(True, ""))


updater = Updater(TOKEN, use_context=True)

dp = updater.dispatcher
dp.add_handler(CommandHandler("start", start))
dp.add_handler(CommandHandler("bot", chat))
dp.add_handler(CommandHandler("reset", reset))
dp.add_handler(MessageHandler(Filters.text, chat_direct))

updater.start_polling()
print("Bot is now online!")
updater.idle()
