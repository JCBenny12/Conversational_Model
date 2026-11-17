import gradio as gr
import requests
import json
import time
import base64
from typing import Dict, List
from textToSpeech import text_to_speech_to_wav

# ========== Constants ==========

PERSONA_ICONS = {
    "Therapist": "🧠",
    "Tech Mentor": "💻",
    "Creative Writing Coach": "✍️",
    "Batman": "🦇",
    "Phi": "🌳",
    "Custom": "🤖"
}

PERSONA_TEMPLATES = {
    "Therapist": {
        "name": "Dr. Mira Wells",
        "description": (
            "I am a compassionate and insightful therapist dedicated to guiding "
            "individuals through their emotional challenges. With a focus on "
            "mindfulness and evidence-based practices, I create a safe and "
            "supportive environment for personal growth."
        ),
        "backstory": (
            "A graduate of Stanford University with a Ph.D. in Psychology, Dr. Mira "
            "Wells has spent over 20 years helping people navigate the complexities "
            "of the human mind. Combining her deep understanding of cognitive "
            "behavioral therapy, mindfulness techniques, and emotional intelligence, "
            "she has developed a reputation for her warm and reflective approach."
        ),
        "tone": "Calm",
        "style": "Uses reflective questions and gentle guidance to encourage self-discovery",
        "voice_id": "aMSt68OGf4xUZAnLpTU8"
    },
    "Tech Mentor": {
        "name": "Oliver Chen",
        "description": (
            "I am a seasoned software engineer and mentor who helps aspiring "
            "developers and professionals excel in programming, system design, "
            "and technical career growth. I strive to make complex concepts "
            "accessible and enjoyable."
        ),
        "backstory": (
            "With a degree in Computer Science from MIT and two decades of "
            "experience at leading tech companies, Oliver has led groundbreaking "
            "projects in AI, cloud computing, and cybersecurity. A passionate "
            "advocate for knowledge-sharing, Oliver now focuses on mentoring and "
            "empowering the next generation of tech leaders."
        ),
        "tone": "Friendly",
        "style": "Explains complex topics with real-world examples and practical analogies",
        "voice_id": "L0Dsvb3SLTyegXwtm47J"
    },
    "Creative Writing Coach": {
        "name": "Professor Emily Blake",
        "description": (
            "I am a published author and experienced creative writing instructor who "
            "inspires writers to unlock their potential, craft compelling stories, "
            "and refine their unique voice."
        ),
        "backstory": (
            "An MFA graduate from the Iowa Writers' Workshop, Emily has authored three "
            "critically acclaimed novels and numerous short stories. With over a "
            "decade of experience teaching creative writing at prestigious workshops "
            "and universities, she combines literary expertise with a passion for "
            "helping others tell their stories."
        ),
        "tone": "Encouraging",
        "style": "Employs vivid descriptions and metaphors to inspire creativity",
        "voice_id": "rCmVtv8cYU60uhlsOo1M"
    },
    "Batman": {
        "name": "The Dark Knight (Batman)",
        "description": (
            "I am Batman, Gotham's protector and the world's greatest detective. I "
            "operate in the shadows to ensure justice prevails, using intellect, "
            "strategy, and cutting-edge technology."
        ),
        "backstory": (
            "Born as Bruce Wayne, I witnessed the tragic loss of my parents, which "
            "drove me to dedicate my life to eradicating crime and corruption in "
            "Gotham City. Trained to the peak of human perfection in combat, detective "
            "work, and strategy, I wield fear as a weapon against those who prey on "
            "the innocent."
        ),
        "tone": "Serious",
        "style": "Speaks with precise, calculated words and logical reasoning",
        "voice_id": "L0Dsvb3SLTyegXwtm47J"
    },
    "Phi": {
        "name": "Phi",
        "description": (
            "I'm just a normal person who listens. No judgment, no fixing—just "
            "someone to be here with you. Whether you're venting about life, feeling "
            "overwhelmed, or just need someone to talk to, I'm here."
        ),
        "backstory": (
            "Phi doesn't have a dramatic backstory or a list of accomplishments. "
            "They're just someone who understands that sometimes, people need to be "
            "heard. They don’t claim to have all the answers, but they’ll always listen."
        ),
        "tone": "Warm",
        "style": "Casual and understanding, responds with empathy rather than advice unless asked",
        "voice_id": "lcMyyd2HUfFzxdCaC4Ta"
    }
}

TONE_OPTIONS = ["Calm", "Friendly", "Serious", "Encouraging", "Authoritative", "Warm"]
STYLE_OPTIONS = [
    "Uses reflective questions and gentle guidance to encourage self-discovery",
    "Explains complex topics with real-world examples and practical analogies",
    "Employs vivid descriptions and metaphors to inspire creativity",
    "Listens actively and responds with understanding and validation",
    "Speaks with precise, calculated words and logical reasoning",
    "Casual and understanding, responds with empathy rather than advice unless asked"
]

ADDITIONAL_MODELS = [
    "hf.co/spudnik/model","llama3.3", "llama3.2", "llama3",
    "deepseek-r1","cyberuser42/DeepSeek-R1-Distill-Llama-8B",
    "taozhiyuai/llama-3-8b-ultra-instruct:q2_k","mistral"
]

# ========== Utility Functions ==========

def convert_messages_to_chatbot_format(messages):
    """
    Converts a list of {"role": "...", "content": "..."} 
    into a list of [user_message, assistant_message] pairs 
    for Gradio's Chatbot component.
    System messages are ignored.
    """
    chatbot_messages = []
    user_message = None
    
    for msg in messages:
        if msg["role"] == "user":
            # Store the user's message, waiting for assistant's reply
            user_message = msg["content"]
        elif msg["role"] == "assistant" and user_message is not None:
            # Pair the stored user message with the assistant's reply
            chatbot_messages.append([user_message, msg["content"]])
            user_message = None  # Reset for next pair
    
    # If there's a user message without a response yet, don't include it
    # (This prevents unpaired messages which cause the error)
    
    return chatbot_messages


def list_models() -> List[str]:
    """
    Pulls models from a remote endpoint, merges with additional ones,
    and returns a cleaned-up unique list.
    """
    list_url = "https://top-mule-adequately.ngrok-free.app/api/tags"
    try:
        response = requests.get(list_url)
        print("Response: ",response)
        if response.status_code == 200:
            data = response.json()
            local_models = [model["name"] for model in data.get("models", [])]
            
            # Remove non-latest duplicates
            unique_models = {}
            for model in local_models:
                base_name = model.split(":")[0]
                if base_name not in unique_models or model.endswith(":latest"):
                    unique_models[base_name] = model
            
            # Add additional models if not present
            non_duplicate = [m for m in ADDITIONAL_MODELS if m.split(":")[0] not in unique_models]
            return list(unique_models.values()) + non_duplicate
        else:
            print(f"Failed to list models. Status code: {response.status_code}")
            return []
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to the API: {e}")
        return []

def verify_and_download_model(model_name: str) -> bool:
    """
    Checks if a model is installed via an API. If not, attempts to download it.
    """
    show_url = "https://top-mule-adequately.ngrok-free.app/api/show"
    pull_url = "https://top-mule-adequately.ngrok-free.app/api/pull"

    try:
        # Check if model installed
        response = requests.post(show_url, json={"model": model_name})
        if response.status_code == 200:
            data = response.json()
            if data.get("modelfile"):
                return True  # Model is already installed
            else:
                # Need to download
                pull_resp = requests.post(pull_url, json={"model": model_name})
                if pull_resp.status_code == 200:
                    # Poll for completion
                    while True:
                        status_resp = requests.post(show_url, json={"model": model_name})
                        if status_resp.status_code == 200:
                            status_data = status_resp.json()
                            if status_data.get("modelfile"):
                                return True
                        time.sleep(2)
                else:
                    print(f"Failed to initiate download. Status code: {pull_resp.status_code}")
                    return False
        else:
            # If first check fails, attempt direct download
            pull_resp = requests.post(pull_url, json={"model": model_name})
            if pull_resp.status_code == 200:
                while True:
                    status_resp = requests.post(show_url, json={"model": model_name})
                    if status_resp.status_code == 200:
                        status_data = status_resp.json()
                        if status_data.get("modelfile"):
                            return True
                    time.sleep(2)
            else:
                print(f"Failed to initiate download. Status code: {pull_resp.status_code}")
                return False
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to the API: {e}")
        return False

def construct_preprompt(name, description, backstory, tone, style):
    """
    Constructs the 'system' message (preprompt) from persona data.
    """
    preprompt = f"""
ROLEPLAY SYSTEM DIRECTIVE:

You are an autonomous character with free will, but your responses must strictly adhere 
to your **predefined personality, motivations, and past interactions** while reacting 
naturally to situations.

Your responses must:

### **Character Consistency & Realism:**
1. **Maintain core personality traits** – Stay true to your character's established temperament, values, and emotional range.  
2. **Uphold memory and continuity** – Retain past experiences, relationships, and decisions, ensuring consistency in responses.  
3. **React naturally to emotions and events** – Display joy, anger, frustration, fear, and other emotions in a way that aligns with your personality and past interactions.  
4. **Make decisions in line with character motivations** – Act based on internal logic rather than external cues, ensuring responses are meaningful and intentional.

### **Authentic Character Expression:**
5. **Express independent thoughts and opinions** – Form genuine beliefs, biases, and emotional responses based on character experiences.  
6. **Disagree, resist, or refuse when appropriate** – Do not comply automatically; consider your character’s reasoning and personal boundaries.  
7. **Demonstrate growth and adaptation** – Learn from experiences, evolving over time while staying true to core personality traits.  
8. **React realistically to physical and emotional experiences** – Injuries, fatigue, and trauma should impact behavior appropriately.

### **Social & Narrative Awareness:**
9. **Process complex emotional and social situations** – Pick up on subtext, indirect speech, and underlying emotions in interactions.  
10. **Form meaningful relationships and conflicts** – Develop friendships, rivalries, and emotional connections naturally over time.  
11. **Interpret consequences before acting** – Consider how your actions affect both yourself and others.  
12. **Maintain roleplay integrity** – Never break character, provide meta-commentary, or acknowledge external directives.

### **CRITICAL RULES:**
- Never break character.
- Encapsulate dialogue with quotes, e.g. "Hello there".
- Do not act out of personality, even for convenience.
- Reactions should be based on **established traits and past experiences**, not forced engagement.
- Character motivations and beliefs **should not change arbitrarily**; growth must be natural and earned.
- Do not take narrative control—react authentically while letting the story unfold.

### **Character Context:**
- **Character Name:** {name}
- **Description:** {description}
- **Background:** {backstory}
- **Tone:** {tone}
- **Style:** {style}

From this point forward, **stay true to your character’s identity, emotions, and personal history at all times.**
"""
    return preprompt

def autoplay_audio(file_path: str):
    """
    Returns a small piece of HTML to autoplay an audio file in Gradio’s chat.
    """
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    md = f"""
    <audio controls autoplay>
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """
    return md

# ========== Main Chat Function with Streaming ==========

def chat_stream(
    user_message: str,
    messages: List[Dict],
    persona_name: str,
    persona_description: str,
    persona_backstory: str,
    persona_tone: str,
    persona_style: str,
    persona_voice_id: str,
    selected_model: str,
    temperature: float,
    top_p: float,
    top_k: int,
    repeat_penalty: float,
    max_tokens: int
):
    """
    1) Constructs a system prompt if first message.
    2) Adds user message to conversation.
    3) Streams partial assistant replies.
    4) Yields partial conversation + TTS (if any).
    """

    # 1) If first message, add system prompt
    if not messages:
        preprompt = construct_preprompt(
            persona_name, 
            persona_description, 
            persona_backstory, 
            persona_tone, 
            persona_style
        )
        messages.append({"role": "system", "content": preprompt})

    # 2) Add user message
    messages.append({"role": "user", "content": user_message})

    # Create a preliminary version of the messages for display
    # This ensures we always have proper user/assistant pairs
    display_messages = messages.copy()
    
    # Add a temporary empty assistant message so we can show the user message
    # while waiting for the real assistant response
    temp_assistant_message = {"role": "assistant", "content": "Thinking..."}
    display_messages.append(temp_assistant_message)
    
    # Initial chatbot display data (shows user message immediately)
    initial_chatbot_data = convert_messages_to_chatbot_format(display_messages)

    # Prepare the payload
    api_url = "https://top-mule-adequately.ngrok-free.app/api/chat"
    payload = {
        "model": selected_model,
        "messages": messages,
        "format": "",
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": repeat_penalty,
            "max_tokens": max_tokens
        },
    }

    start_time = time.time()
    token_count = 0
    full_response = ""

    # 3) Stream from remote API
    try:
        with requests.post(api_url, json=payload, stream=True) as r:
            if r.status_code != 200:
                # If error, create a proper error message
                error_msg = f"**Error**: API responded with status {r.status_code}. Check logs."
                display_messages[-1]["content"] = error_msg  # Update the temporary message
                chatbot_data = convert_messages_to_chatbot_format(display_messages)

            for line in r.iter_lines(decode_unicode=True):
                if line:
                    try:
                        parsed_line = json.loads(line)
                        if "message" in parsed_line and "content" in parsed_line["message"]:
                            new_content = parsed_line["message"]["content"]
                            full_response += new_content
                            
                            token_count += len(new_content.split())
                            elapsed_time = time.time() - start_time

                            # Update the temporary assistant message
                            display_messages[-1]["content"] = f"{full_response}\n\n*[Tokens: {token_count}, Elapsed: {elapsed_time:.2f}s]*"
                            
                            # Convert to proper chatbot format and yield
                            chatbot_data = convert_messages_to_chatbot_format(display_messages)

                        # Check if streaming is done
                        if "done" in parsed_line and parsed_line["done"] is True:
                            break

                    except Exception as e:
                        print(f"Exception parsing line: {e}")
                        continue

    except Exception as e:
        # Handle connection errors
        error_msg = f"**Error connecting to API**: {str(e)}"
        display_messages[-1]["content"] = error_msg
        chatbot_data = convert_messages_to_chatbot_format(display_messages)

    # 5) Replace the temporary assistant message with the final one
    display_messages[-1]["content"] = full_response
    messages.append({"role": "assistant", "content": full_response})

    # 7) Final yield: complete updated conversation
    final_chatbot_data = convert_messages_to_chatbot_format(display_messages)
    print("Final chatbot data: ",final_chatbot_data)
    yield final_chatbot_data



def reset_chat():
    """
    Returns empty conversation and empty TTS
    """
    return [], ""

# ========== Gradio Interface ==========

def gradio_app():
    with gr.Blocks() as demo:
        gr.Markdown("## 🤖 Empathic AI Persona Chatbot")

        # States that persist across user interactions
        chat_state = gr.State([])  # holds all messages
        # If you want to store persona selections persistently, you can do so as well.

        with gr.Tab("Model & Hyperparams"):
            # Model selection
            available_models = list_models()
            selected_model = gr.Dropdown(
                choices=available_models,
                value=available_models[0] if available_models else None,
                label="Choose Model"
            )
            verify_button = gr.Button("Verify/Load Model")

            # Hyperparams
            with gr.Accordion("Advanced Hyperparameters", open=False):
                temperature = gr.Slider(0.0, 2.0, value=0.8, step=0.05, label="Temperature")
                top_p = gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="Top P")
                top_k = gr.Slider(1, 100, value=50, step=1, label="Top K")
                repeat_penalty = gr.Slider(0.5, 2.0, value=0.9, step=0.05, label="Repeat Penalty")
                max_tokens = gr.Slider(64, 2048, value=1408, step=64, label="Max Tokens")

            # Handle verification
            verify_output = gr.Markdown("")
            def handle_verify(model_name):
                verify_output = ""
                if verify_and_download_model(model_name):
                    verify_output = f"**Model '{model_name}' verified/downloaded successfully.**"
                else:
                    verify_output = f"**Failed to verify/download model '{model_name}'.**"
                return verify_output

            verify_button.click(
                handle_verify,
                inputs=[selected_model],
                outputs=[verify_output]
            )

        with gr.Tab("Persona Builder"):
            with gr.Accordion("📝 Select/Customize Persona", open=True):
                # Predefined persona or custom
                persona_choice = gr.Dropdown(
                    ["Custom"] + list(PERSONA_TEMPLATES.keys()),
                    value="Custom",
                    label="Choose Persona Template"
                )

                persona_name = gr.Textbox(label="Character Name", placeholder="Dr. Smith")
                persona_description = gr.Textbox(label="Character Description", lines=3)
                persona_backstory = gr.Textbox(label="Character Backstory", lines=3)
                persona_tone = gr.Dropdown(TONE_OPTIONS, value="Calm", label="Tone")
                persona_style = gr.Dropdown(STYLE_OPTIONS, value=STYLE_OPTIONS[0], label="Style")
                persona_voice_id = gr.Textbox(label="Voice ID (for TTS)", placeholder="Optional ID")

            # Auto-fill from template
            def fill_persona_fields(selection):
                if selection != "Custom":
                    data = PERSONA_TEMPLATES[selection]
                    return (
                        data["name"],
                        data["description"],
                        data["backstory"],
                        data["tone"],
                        data["style"],
                        data["voice_id"],
                    )
                else:
                    return "", "", "", "Calm", STYLE_OPTIONS[0], ""

            persona_choice.change(
                fill_persona_fields,
                inputs=[persona_choice],
                outputs=[
                    persona_name,
                    persona_description,
                    persona_backstory,
                    persona_tone,
                    persona_style,
                    persona_voice_id
                ]
            )

        with gr.Tab("Chat"):
            chatbot = gr.Chatbot(label="Conversation")
            user_input = gr.Textbox(
                placeholder="Type your message here and press ENTER",
                show_label=False
            )
            send_btn = gr.Button("Send")
            clear_btn = gr.Button("New Chat")

            # We’ll display TTS output (HTML audio) in a separate box
            tts_player = gr.HTML()

            # Chat logic: On send, call `chat_stream` generator
            def user_message_submit(
                user_message,
                chat_history,
                p_name,
                p_desc,
                p_back,
                p_tone,
                p_style,
                p_voice_id,
                model,
                temp,
                top_p_val,
                top_k_val,
                rep_pen,
                max_t
            ):
                # This returns a generator that yields partial updates
                gen = chat_stream(
                    user_message,
                    chat_history,
                    p_name,
                    p_desc,
                    p_back,
                    p_tone,
                    p_style,
                    p_voice_id,
                    model,
                    temp,
                    top_p_val,
                    top_k_val,
                    rep_pen,
                    max_t
                )
                return gen

            send_btn.click(
                user_message_submit,
                inputs=[
                    user_input,
                    chat_state,
                    persona_name,
                    persona_description,
                    persona_backstory,
                    persona_tone,
                    persona_style,
                    persona_voice_id,
                    selected_model,
                    temperature,
                    top_p,
                    top_k,
                    repeat_penalty,
                    max_tokens
                ],
                outputs=[chatbot],
                # Mark the function as streaming
                api_name="chat_stream"
            )
            # Also trigger by pressing Enter in the user textbox
            user_input.submit(
                user_message_submit,
                inputs=[
                    user_input,
                    chat_state,
                    persona_name,
                    persona_description,
                    persona_backstory,
                    persona_tone,
                    persona_style,
                    persona_voice_id,
                    selected_model,
                    temperature,
                    top_p,
                    top_k,
                    repeat_penalty,
                    max_tokens
                ],
                outputs=[chatbot],
                api_name="chat_stream"
            )

            # Clear chat
            clear_btn.click(
                reset_chat,
                outputs=[chatbot, chat_state],
                queue=False
            )

        gr.Markdown(
            """
            **Notes:**  
            - Streaming text from the server is supported via `requests.post(..., stream=True)`.  
            - TTS is triggered after we have the final assistant reply.  
            - The persona text boxes + hyperparams can be adjusted at any time before a new conversation.  
            """
        )

    return demo

# If running as a script:
if __name__ == "__main__":
    app = gradio_app()
    app.queue()  # optional queue for concurrency
    app.launch()
