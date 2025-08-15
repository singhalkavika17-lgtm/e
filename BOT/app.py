import gradio as gr
from mistralai import Mistral
import os
from dotenv import load_dotenv
from datetime import datetime

# Load API key
load_dotenv(dotenv_path="/Users/vimlasinghal/Empathy-bot/_.env")
API_KEY = os.getenv("MISTRAL_API_KEY")

if not API_KEY:
    raise ValueError("No MISTRAL_API_KEY found in .env file")

# Initialize client
client = Mistral(api_key=API_KEY)

LOG_FILE = "chat_logs.txt"

def save_chat_log(user, bot):
    """Append chat message pair to log file."""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        f.write(f"{timestamp} USER: {user}\n")
        f.write(f"{timestamp} BOT: {bot}\n\n")

def respond(message, chat_history):
    # Base system prompt
    messages = [{
        "role": "system",
        "content": (
            "You are a calm empathetic listener. Do not be too empathetic or kind, "
            "act like a friend for the user. Your response to greetings should be 5-10 words. "
            "Answer shorter inputs with short responses of 20-30 words and longer ones with 50 words. "
            "Ask one open-ended question that encourages the user to share more."
        )
    }]

    # Include conversation history
    for user, bot in chat_history:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": bot})
    messages.append({"role": "user", "content": message})

    # Get model response
    response = client.chat.complete(
        model="mistral-large-latest",
        messages=messages,
        temperature=0.7,
        max_tokens=300
    )

    reply = response.choices[0].message.content
    chat_history.append((message, reply))
    save_chat_log(message, reply)

    return "", chat_history

# Gradio UI (text-only, no audio deps)
with gr.Blocks() as demo:
    gr.Markdown("## Empathetic Chatbot (Mistral)")
    chatbot = gr.Chatbot(height=420)
    msg = gr.Textbox(placeholder="Type your message and press Enter", label="Your Message")

    with gr.Row():
        done_btn = gr.Button("✅ Send")
        clear_btn = gr.Button("🗑️ Clear")

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    done_btn.click(respond, [msg, chatbot], [msg, chatbot])
    clear_btn.click(lambda: [], outputs=chatbot)

demo.launch(share=False)
