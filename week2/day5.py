import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
import sqlite3
import base64
from io import BytesIO
from PIL import Image

load_dotenv(override=True)


openai = OpenAI()

# Ollama local
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss")

ollama = OpenAI(base_url=OLLAMA_BASE_URL)

DB = "ticket_prices.db"


# 1. Lets setup a database to store ticket prices for city with some sample data
def seed_db():
    sample_data = [
        ("london", "$799"),
        ("paris", "$699"),
        ("new york", "$599"),
        ("tokyo", "$999"),
        ("sydney", "$899"),
        ("berlin", "$649"),
        ("dubai", "$749"),
        ("singapore", "$849"),
        ("mumbai", "$699"),
        ("delhi", "$659"),
        ("pune", "$899"),
        ("chennai", "$859"),
        ("bangalore", "$799"),
        ("kolkata", "$699"),
    ]

    conn = sqlite3.connect(DB)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ticket_prices (
            city TEXT PRIMARY KEY,
            price TEXT
        )
    """)

    cursor.executemany(
        "INSERT OR REPLACE INTO ticket_prices (city, price) VALUES (?, ?)", sample_data
    )
    conn.commit()
    conn.close()


seed_db()


def get_ticket_price(city):
    conn = sqlite3.connect(DB)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT price FROM ticket_prices WHERE city = ?", (city.lower().strip(),)
    )
    result = cursor.fetchone()
    conn.close()
    if result:
        return f"The ticket price for {city.title()} is {result[0]}."
    else:
        return f"Sorry, we do not have ticket price information for {city.title()}."


def artist(city):
    image_response = openai.images.generate(
        model="gpt-image-1-mini",
        prompt=f"An image representing a vacation in {city}, showing tourist spots and everything unique about {city}, in a vibrant pop-art style",
        size="1024x1024",
        n=1,
    )
    image_base64 = image_response.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(BytesIO(image_bytes))
    return image


def talker(message):
    response = openai.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="onyx",  # Also, try replacing onyx with alloy or coral
        input=message,
    )
    return response.content


system_message = """
You are a helpful assistant for an Airline called FlightAI.
Give short, courteous answers, no more than 1 sentence.
Always be accurate. If you don't know the answer, say so.
"""

ticket_price_function = {
    "name": "get_ticket_price",
    "description": "Get the ticket price for a given city.",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "The name of the city to get the ticket price for.",
            }
        },
        "required": ["city"],
    },
}
tools = [{"type": "function", "function": ticket_price_function}]


def handle_tool_call(tool_call) -> tuple[dict, str | None]:
    if tool_call.function.name == "get_ticket_price":
        arguments = json.loads(tool_call.function.arguments)
        city = arguments.get("city")
        price_response = get_ticket_price(city)
        response = {
            "role": "tool",
            "content": price_response,
            "tool_call_id": tool_call.id,
        }
        return response, city

    return {
        "role": "tool",
        "content": "Tool call not recognized.",
        "tool_call_id": tool_call.id,
    }, None


def chat(history):
    history = [{"role": h["role"], "content": h["content"]} for h in history]
    messages = [{"role": "system", "content": system_message}] + history

    response = ollama.chat.completions.create(
        model=OLLAMA_MODEL, messages=messages, tools=tools
    )

    cities = []
    image = None

    while response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        messages.append(message)
        for tool_call in message.tool_calls:
            response_dict, city = handle_tool_call(tool_call)
            if city:
                cities.append(city)
            messages.append(response_dict)
        response = ollama.chat.completions.create(
            model=OLLAMA_MODEL, messages=messages, tools=tools
        )

    reply = response.choices[0].message.content
    history += [{"role": "assistant", "content": reply}]

    voice = talker(reply)

    if cities:
        image = artist(cities[0])

    return history, voice, image


def generate_gradio_ui():
    def put_messages_in_chat_box(message, history):
        return "", history + [{"role": "user", "content": message}]

    with gr.Blocks() as ui:
        gr.Markdown(
            """
            # FlightAI - Your Personal Travel Assistant
            Ask me about flight ticket prices for different cities and get a pop-art style image of the city!
            """
        )
        with gr.Row():
            chatbot = gr.Chatbot(
                elem_id="chatbot", label="FlightAI Chatbot", height=500, type="messages"
            )
            image_output = gr.Image(
                elem_id="image_output",
                label="City Image",
                height=500,
                interactive=False,
            )
        with gr.Row():
            audio_output = gr.Audio(
                elem_id="audio_output",
                label="Assistant's Voice Response",
                autoplay=True,
            )
        with gr.Row():
            message = gr.Textbox(
                elem_id="message",
                label="Chat with FlightAI",
                placeholder="Type your message here...",
            )

        message.submit(
            fn=put_messages_in_chat_box,
            inputs=[message, chatbot],
            outputs=[message, chatbot],
        ).then(
            fn=chat,
            inputs=[chatbot],
            outputs=[chatbot, audio_output, image_output],
        )

    return ui


ui = generate_gradio_ui()
ui.launch()
