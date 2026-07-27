import os
import base64
import mimetypes
import gradio as gr
from groq import Groq
from dotenv import load_dotenv


# Load .env file
load_dotenv()


# --------------------------------------------------
# Config
# --------------------------------------------------

MODEL = "qwen/qwen3.6-27b"

SYSTEM_PROMPT = (
    "You are a helpful visual assistant. When an image is provided, "
    "describe and reason about it accurately before answering the question."
)

MAX_IMAGES_PER_TURN = 5

API_KEY = os.environ.get("GROQ_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY not found. Set it in your shell "
        "(export/setx) or in a .env file next to this script."
    )


client = Groq(api_key=API_KEY)


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def encode_image_to_data_url(filepath: str) -> str:
    """
    Reads a local image file and returns a base64 data URL.
    """

    mime_type, _ = mimetypes.guess_type(filepath)

    mime_type = mime_type or "image/jpeg"

    with open(filepath, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{b64}"


def build_user_content(text: str, files: list[str]) -> list[dict]:
    """
    Builds an OpenAI/Groq-style multimodal content block list.
    """

    content = []

    # Add text
    if text:
        content.append({
            "type": "text",
            "text": text
        })

    # Add images
    for filepath in files[:MAX_IMAGES_PER_TURN]:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": encode_image_to_data_url(filepath)
            }
        })

    return content


def history_to_messages(history: list[dict]) -> list[dict]:
    """
    Converts Gradio's stored history into API messages.
    Only text from previous turns is retained.
    """

    messages = []

    for turn in history:

        role = turn.get("role")
        content = turn.get("content")

        if isinstance(content, str):
            messages.append({
                "role": role,
                "content": content
            })

    return messages


# --------------------------------------------------
# Core Chat Function
# --------------------------------------------------

def respond(message, history):

    """
    message format:

    {
        "text": "...",
        "files": [
            "/tmp/example.png"
        ]
    }

    history:
    Previous turns in messages format.
    """

    text = message.get("text", "") or ""

    files = message.get("files", []) or []

    # Create messages
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    # Add previous conversation
    messages.extend(
        history_to_messages(history)
    )

    # Add current user message
    messages.append({
        "role": "user",
        "content": build_user_content(
            text,
            files
        )
    })

    try:

        stream = client.chat.completions.create(

            model=MODEL,

            messages=messages,

            temperature=0.5,

            max_tokens=1024,

            stream=True
        )

    except Exception as e:

        yield f"⚠️ Error contacting Groq API: {e}"

        return


    # Store response
    partial = ""

    # Stream response
    for chunk in stream:

        delta = chunk.choices[0].delta.content or ""

        partial += delta

        yield partial


# --------------------------------------------------
# UI
# --------------------------------------------------

theme = gr.themes.Soft(

    primary_hue="teal",

    secondary_hue="slate",

    font=[
        gr.themes.GoogleFont("Inter"),
        "sans-serif"
    ]
)


CUSTOM_CSS = """
#chatbot-container {
    max-width: 860px;
    margin: 0 auto;
}

.gradio-container {
    background: #f7f9fb;
}

footer {
    visibility: hidden;
}
"""


# --------------------------------------------------
# Gradio Interface
# --------------------------------------------------

with gr.Blocks(
    theme=theme,
    css=CUSTOM_CSS
) as demo:

    gr.ChatInterface(

        fn=respond,

        multimodal=True,

        title="Groq Vision Chat",

        description=(
            "Attach an image and ask a question — powered by a "
            "Groq-hosted vision-language model. "
            "Text-only messages work too."
        ),

        chatbot=gr.Chatbot(

            height=560,

            elem_id="chatbot-container",

            avatar_images=(
                None,
                "https://groq.com/favicon.ico"
            ),

            buttons=[
                "copy",
                "copy_all"
            ]
        ),

        textbox=gr.MultimodalTextbox(

            placeholder=(
                "Ask a question, optionally attach an image..."
            ),

            file_types=["image"],

            file_count="multiple",

            container=False,

            scale=7
        ),

        submit_btn="Send"
    )


# --------------------------------------------------
# Run Application
# --------------------------------------------------

if __name__ == "__main__":

    demo.queue().launch()