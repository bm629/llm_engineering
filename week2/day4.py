import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

# Load environment variables from .env file
load_dotenv(override=True)


# Ollama local
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss")

ollama = OpenAI(base_url=OLLAMA_BASE_URL)


# Define ticket prices for different destinations
ticket_prices = {
    "london": "$799",
    "paris": "$699",
    "new york": "$599",
    "tokyo": "$999",
    "sydney": "$1099",
}


def get_ticket_price(destination_city):
    print(f"Fetching ticket price for {destination_city}...")
    price = ticket_prices.get(
        destination_city.lower().strip(), "Price not available for this destination."
    )

    if price == "Price not available for this destination.":
        return "Sorry, we do not have ticket prices available for {}.".format(
            destination_city
        )

    return "The ticket price for {} is {}.".format(destination_city, price)


##################################################################################################

# system message for the model
system_message = """
You are a helpful assistant for an Airline called FlightAI.
Give short, courteous answers, no more than 1 sentence.
Always be accurate. If you don't know the answer, say so.
"""

price_function_tool = {
    "name": "get_ticket_price",
    "description": "Get the ticket price for a given destination city.",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {
                "type": "string",
                "description": "The name of the destination city to get the ticket price for.",
            }
        },
        "required": ["destination_city"],
        "additionalProperties": False,
    },
}
tools = [{"type": "function", "function": price_function_tool}]


def handle_tool_call(tool_call):
    if tool_call.function.name == "get_ticket_price":
        arguments = json.loads(tool_call.function.arguments)
        destination_city = arguments.get("destination_city")
        price_response = get_ticket_price(destination_city)
        response = {
            "role": "tool",
            "content": price_response,
            "tool_call_id": tool_call.id,
        }
        return response

    return {
        "role": "tool",
        "content": "Tool call not recognized.",
        "tool_call_id": tool_call.id,
    }


def chat_stream(message, history):
    history = [{"role": h["role"], "content": h["content"]} for h in history]
    messages = (
        [{"role": "system", "content": system_message}]
        + history
        + [{"role": "user", "content": message}]
    )

    while True:
        stream = ollama.chat.completions.create(
            model=OLLAMA_MODEL, messages=messages, stream=True, tools=tools
        )
        response = ""
        for chunk in stream:
            delta = chunk.choices[0].delta

            if delta.tool_calls:
                messages.append(delta)

                for tool_call in delta.tool_calls:
                    tool_response = handle_tool_call(tool_call)
                    messages.append(tool_response)
            elif delta.content:
                response += delta.content
                yield response

        # stream fully read. Same check as the notebook, on the last chunk.
        if chunk.choices[0].finish_reason != "tool_calls":
            break


# Create a Gradio ChatInterface in background
app = gr.ChatInterface(fn=chat_stream, title="FlightAI Chat", type="messages")
app.launch()
