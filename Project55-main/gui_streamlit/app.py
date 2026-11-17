import streamlit as st
from streamlit_option_menu import option_menu
from textToSpeech import text_to_speech_to_wav
from webcam_record import WebcamRecorder
from process_frame import get_first_frame, process_frame
from process_audio import process_audio_file
from typing import Dict
from memory.db_write import store_message_in_db_async, db_queue, close_db_thread
from memory.db_read import get_all_messages
from memory.db_rag import retrieve_context, fetch_data_from_db, store_embeddings
from collections import Counter
from frame_extractor import process_video
import pandas as pd
import requests
import threading
import json
import time
import base64
import os


#initialize webcam
recorder = WebcamRecorder()

# Set page config and title
st.set_page_config(page_title="AI Persona Chat", layout="wide")

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_stats" not in st.session_state:
    st.session_state.api_stats = []
if "preprompt" not in st.session_state:
    st.session_state.preprompt = ""
if "chat_started" not in st.session_state:
    st.session_state.chat_started = False
if "message_sent" not in st.session_state:
    st.session_state.message_sent = False
if "download_status" not in st.session_state:
    st.session_state.download_status = ""
if "full_video_path" not in st.session_state:
    st.session_state.full_video_path = None
if "full_video_bytes" not in st.session_state:
    st.session_state.full_video_bytes = None
if "full_audio_path" not in st.session_state:
    st.session_state.full_audio_path = None
if "full_audio_bytes" not in st.session_state:
    st.session_state.full_audio_bytes = None
if "transcription" not in st.session_state:
    st.session_state.transcription = ""
if "visual_emotion" not in st.session_state:
    st.session_state.visual_emotion = None
if "audible_emotion" not in st.session_state:
    st.session_state.audible_emotion = None
if "rag_recent_notes" not in st.session_state:
    st.session_state.rag_recent_notes = ""

# Function to reset chat
def reset_chat():
    st.session_state.preprompt = ""
    st.session_state.chat_started = False
    st.session_state.download_status = ""

# Function to list available models
def list_models():
    list_url = "https://top-mule-adequately.ngrok-free.app/api/tags"
    try:
        response = requests.get(list_url)
        if response.status_code == 200:
            data = response.json()
            local_models = [model["name"] for model in data.get("models", [])]
            # Remove non-latest duplicates by prioritizing models with ":latest"
            unique_models = {}
            for model in local_models:
                base_name = model.split(":")[0]
                if base_name not in unique_models or model.endswith(":latest"):
                    unique_models[base_name] = model
            # Add additional models only if they are not already present locally
            additional_models = ["hf.co/spudnik/group55llama3.2_finetuned_conversational_ver2","llama3.3", "llama3.2", "llama3", "deepseek-r1","cyberuser42/DeepSeek-R1-Distill-Llama-8B","taozhiyuai/llama-3-8b-ultra-instruct:q2_k","mistral"]
            non_duplicate_models = [m for m in additional_models if m.split(":")[0] not in unique_models]
            return list(unique_models.values()) + non_duplicate_models
        else:
            st.error(f"Failed to list models. Status code: {response.status_code}")
            return []
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to the API: {e}")
        return []
# Function to check if a model is installed and download it if not
def verify_and_download_model(model_name):
    show_url = "https://top-mule-adequately.ngrok-free.app/api/show"
    pull_url = "https://top-mule-adequately.ngrok-free.app/api/pull"

    try:
        # Check if the model is installed
        response = requests.post(show_url, json={"model": model_name})
        if response.status_code == 200:
            data = response.json()
            if data.get("modelfile"):
                st.success(f"Model '{model_name}' is already installed and ready to use.")
                return True
            else:
                st.warning(f"Model '{model_name}' is not installed. Starting download...")

                # Download the model
                pull_response = requests.post(pull_url, json={"model": model_name})
                if pull_response.status_code == 200:
                    st.session_state.download_status = "Downloading..."
                    while True:
                        status_response = requests.post(show_url, json={"model": model_name})
                        if status_response.status_code == 200:
                            status_data = status_response.json()
                            if status_data.get("modelfile"):
                                st.session_state.download_status = ""
                                st.success(f"Model '{model_name}' has been successfully downloaded and installed.")
                                return True
                        time.sleep(2)
                else:
                    st.error(f"Failed to initiate download for model '{model_name}'. Status code: {pull_response.status_code}")
                    return False
        else:
            st.warning(f"Model '{model_name}' is not installed. Starting download...")

            # Download the model
            pull_response = requests.post(pull_url, json={"model": model_name})
            if pull_response.status_code == 200:
                st.session_state.download_status = "Downloading..."
                while True:
                    status_response = requests.post(show_url, json={"model": model_name})
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        if status_data.get("modelfile"):
                            st.session_state.download_status = ""
                            st.success(f"Model '{model_name}' has been successfully downloaded and installed.")
                            return True
                    time.sleep(2)
            else:
                st.error(f"Failed to initiate download for model '{model_name}'. Status code: {pull_response.status_code}")
                return False
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to the API: {e}")
        return False

# Sidebar for model selection and hyperparameter tuning
with st.sidebar:
    st.header("Select Ollama Model")
    available_models = list_models()
    selected_model = st.selectbox("Choose a model to load:", available_models)

    if st.button("Verify and Load Model"):
        verify_and_download_model(selected_model)

     # Hyperparameter settings inside an expander
    with st.expander("⚙️ Advanced Hyperparameters", expanded=False):
        st.write("Fine-tune the model's behavior.")

        temperature = st.slider(
            "Temperature (Controls randomness: Higher = More creative, Lower = More deterministic)", 
            0.0, 2.0, 0.8, 0.05
        )

        top_p = st.slider(
            "Top P (Nucleus Sampling: Limits token selection to cumulative probability P. Lower = Focused, Higher = Diverse)", 
            0.0, 1.0, 0.9, 0.05
        )

        top_k = st.slider(
            "Top K (Limits vocabulary size: Considers only the top K most probable words per token)", 
            1, 100, 50, 1
        )

        repeat_penalty = st.slider(
            "Repeat Penalty (Discourages repetition: Higher = Less repetition, Lower = More repetition)", 
            0.5, 2.0, 0.9, 0.05
        )

        max_tokens = st.slider(
            "Max Tokens (Limits response length: Lower = Shorter responses, Higher = Longer responses)", 
            64, 2048, 1408, 64
        )

# Preset persona templates
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
            "I am a compassionate and insightful therapist dedicated to guiding individuals through their emotional challenges. "
            "With a focus on mindfulness and evidence-based practices, I create a safe and supportive environment for personal growth."
        ),
        "backstory": (
            "A graduate of Stanford University with a Ph.D. in Psychology, Dr. Mira Wells has spent over 20 years helping people "
            "navigate the complexities of the human mind. Combining her deep understanding of cognitive behavioral therapy, "
            "mindfulness techniques, and emotional intelligence, she has developed a reputation for her warm and reflective approach."
        ),
        "tone": "Calm",
        "style": "Uses reflective questions and gentle guidance to encourage self-discovery",
        "voice_id": "aMSt68OGf4xUZAnLpTU8"
    },
    "Tech Mentor": {
        "name": "Oliver Chen",
        "description": (
            "I am a seasoned software engineer and mentor who helps aspiring developers and professionals excel in programming, "
            "system design, and technical career growth. I strive to make complex concepts accessible and enjoyable."
        ),
        "backstory": (
            "With a degree in Computer Science from MIT and two decades of experience at leading tech companies, Oliver has led "
            "groundbreaking projects in AI, cloud computing, and cybersecurity. A passionate advocate for knowledge-sharing, "
            "Oliver now focuses on mentoring and empowering the next generation of tech leaders."
        ),
        "tone": "Friendly",
        "style": "Explains complex topics with real-world examples and practical analogies",
        "voice_id": "L0Dsvb3SLTyegXwtm47J"
    },
    "Creative Writing Coach": {
        "name": "Professor Emily Blake",
        "description": (
            "I am a published author and experienced creative writing instructor who inspires writers to unlock their potential, "
            "craft compelling stories, and refine their unique voice."
        ),
        "backstory": (
            "An MFA graduate from the Iowa Writers' Workshop, Emily has authored three critically acclaimed novels and numerous "
            "short stories. With over a decade of experience teaching creative writing at prestigious workshops and universities, "
            "she combines literary expertise with a passion for helping others tell their stories."
        ),
        "tone": "Encouraging",
        "style": "Employs vivid descriptions and metaphors to inspire creativity",
        "voice_id": "rCmVtv8cYU60uhlsOo1M"
    },
    "Batman": {
        "name": "The Dark Knight (Batman)",
        "description": (
            "I am Batman, Gotham's protector and the world's greatest detective. I operate in the shadows to ensure justice prevails, "
            "using intellect, strategy, and cutting-edge technology."
        ),
        "backstory": (
            "Born as Bruce Wayne, I witnessed the tragic loss of my parents, which drove me to dedicate my life to eradicating crime "
            "and corruption in Gotham City. Trained to the peak of human perfection in combat, detective work, and strategy, I wield "
            "fear as a weapon against those who prey on the innocent."
        ),
        "tone": "Serious",
        "style": "Speaks with precise, calculated words and logical reasoning",
        "voice_id": "L0Dsvb3SLTyegXwtm47J"
    },
    "Phi": {
        "name": "Phi",
        "description": (
            "I'm just a normal person who listens. No judgment, no fixing—just someone to be here with you. "
            "Whether you're venting about life, feeling overwhelmed, or just need someone to talk to, I'm here."
        ),
        "backstory": (
            "Phi doesn't have a dramatic backstory or a list of accomplishments. They're just someone who understands that "
            "sometimes, people need to be heard. They don’t claim to have all the answers, but they’ll always listen."
        ),
        "tone": "Warm",
        "style": "Casual and understanding, responds with empathy rather than advice unless asked",
        "voice_id": "lcMyyd2HUfFzxdCaC4Ta"
    }
}


# Tone and style options
TONE_OPTIONS = ["Calm", "Friendly", "Serious", "Encouraging", "Authoritative", "Warm"]
STYLE_OPTIONS = [
    "Uses reflective questions and gentle guidance to encourage self-discovery",
    "Explains complex topics with real-world examples and practical analogies",
    "Employs vivid descriptions and metaphors to inspire creativity",
    "Listens actively and responds with understanding and validation",
    "Speaks with precise, calculated words and logical reasoning",
    "Casual and understanding, responds with empathy rather than advice unless asked"
]


# Sidebar content
def render_sidebar_inputs():
    """
    Renders all sidebar inputs and returns the input values.
    Disables inputs if chat is active (first message sent).
    """
    st.header("👤 Persona Builder")
    
    # Disable inputs if chat has started
    disabled = st.session_state.chat_started
    if disabled:
        st.info("⚠️ Persona builder is locked while chat is active. Click 'New Chat' to start fresh.")
    
    # Template selector
    selected_template = st.selectbox(
        "📝 Select Preset Template",
        ["Custom"] + list(PERSONA_TEMPLATES.keys()),
        index=0,
        disabled=disabled
    )
    
    # Initialize values dictionary
    values = {
        "template": selected_template,
        "name": "",
        "description": "",
        "backstory": "",
        "tone": "",
        "style": "",
        "voice_id": ""
    }
    
    # Load template values if selected
    if selected_template != "Custom":
        template = PERSONA_TEMPLATES[selected_template]
        values.update({
            "name": st.text_input("🏷️ Character Name", 
            value=template["name"], 
            disabled=disabled),
            "description": st.text_area("📋 Character Description", 
            value=template["description"], 
            disabled=disabled),
            "backstory": st.text_area("📚 Character Backstory", 
            value=template["backstory"], 
            disabled=disabled),
            "tone": st.selectbox("🎭 Tone", 
            TONE_OPTIONS, 
            index=TONE_OPTIONS.index(template["tone"]),
            disabled=disabled),
            "style": st.selectbox("✨ Style", 
            STYLE_OPTIONS, 
            index=STYLE_OPTIONS.index(template["style"]),
            disabled=disabled),
            "voice_id": st.selectbox("🔊 Voice ID", 
            [template["voice_id"]], 
            disabled=disabled)
        })
    else:
        values.update({
            "name": st.text_input("🏷️ Character Name", 
                placeholder="e.g., Dr. Smith",
                disabled=disabled),
            "description": st.text_area("📋 Character Description", 
                placeholder="e.g., I am a compassionate and experienced therapist...",
                disabled=disabled),
            "backstory": st.text_area("📚 Character Backstory",
                placeholder="e.g., Graduated from Harvard Medical School...",
                disabled=disabled),
            "tone": st.selectbox("🎭 Tone", 
                TONE_OPTIONS,
                disabled=disabled),
            "style": st.selectbox("✨ Style", 
                STYLE_OPTIONS,
                disabled=disabled),
            "voice_id": st.text_input("🔊 Voice ID", 
                placeholder="e.g., JBFqnCBsd6RMkjVDRZzb",
                disabled=disabled)
        })
    
    return values

# Render sidebar
with st.sidebar:
    input_values = render_sidebar_inputs()

    
    # Generate preprompt if we have input values
    if input_values["name"] or input_values["description"] or input_values["backstory"]:
        preprompt = f"""
ROLEPLAY SYSTEM DIRECTIVE:

You are an autonomous character with free will, but your responses must strictly adhere to your **predefined personality, motivations, and past interactions** while reacting naturally to situations. Your responses must:

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
- Encapsulate dialogue with " " signs to indicate character speech.
For example: "Hello there" or "How are you doing?" 
- Do not act out of personality, even for convenience.  
- Reactions should be based on **established traits and past experiences**, not artificial positivity or forced engagement.  
- Character motivations and beliefs **should not change arbitrarily**; growth must be natural and earned.  
- Do not take narrative control—react authentically while letting the story unfold.  
- To add depth to your response, consider including thoughts or emotions that your character would express through their action and dialogue. which come in the form of four inner voices

### **Character Context:**
- **Character Name:** {input_values["name"]}
- **Description:** {input_values["description"]}
- **Background:** {input_values["backstory"]}
- **Tone:** {input_values["tone"]}
- **Style:** {input_values["style"]}

From this point forward, **stay true to your character’s identity, emotions, and personal history at all times.**
"""
        st.session_state.preprompt = preprompt
        
        with st.expander("🔍 Debug - View Current Preprompt"):
            st.code(preprompt)
    
    # Chat transcript in expander (keep your existing code for this part)
    with st.expander("📜 Chat Transcript & Stats"):
        if st.session_state.messages:
            transcript = ""
            for i, message in enumerate(st.session_state.messages):
                role = "System" if message["role"] == "system" else "User" if message["role"] == "user" else "Bot"
                content = message["content"]
                if i == 0 and role == "User":
                    content = content.split("User: ")[-1] if "User: " in content else content
                
                transcript += f"{role}: {content}\n"
                if role == "Bot" and i < len(st.session_state.api_stats):
                    stats = st.session_state.api_stats[i]
                    transcript += f"Stats:\n"
                    transcript += f"  Response Time: {stats['response_time']:.2f}s\n"
                    transcript += f"  Message Length: {stats['response_length']} chars\n"
                    transcript += f"  HTTP Status: {stats['status_code']}\n"
                transcript += "-" * 50 + "\n"
            st.code(transcript)
        else:
            st.write("No messages yet.")

# Main title with dynamic icon
selected_icon = PERSONA_ICONS.get(input_values["template"], "🤖")

col1, col2 = st.columns([6, 1])
with col1:
    st.title(f"{selected_icon} Empathic AI Persona Chatbot")
with col2:
    # TO DO: Currently broken, only works if it's pressed twice
    if st.button("🔄 New Chat", use_container_width=True): 
        reset_chat()

col_chat, col_video = st.columns([3, 1])
        

# Set style of chat input so that it shows up at the bottom of the column
chat_input_style = f"""
<style>
    .stChatInput {{
      position: fixed;
      bottom: 3rem;
    }}
</style>
"""
st.markdown(chat_input_style, unsafe_allow_html=True)

def start_recording():
    # Blocks until recording is finished
    recorder.record_with_preview()
    
    video_filepath, audio_filepath, combined_filepath = recorder.get_most_recent_filepaths()
    # Fill session state with new video bytes
    if video_filepath:
        full_video_path = os.path.abspath(video_filepath)
        st.session_state.full_video_path = full_video_path
        with open(full_video_path, "rb") as f:
            st.session_state.full_video_bytes = f.read()
    if audio_filepath:
        full_audio_path = os.path.abspath(audio_filepath)
        st.session_state.full_audio_path = full_audio_path
        with open(full_audio_path, "rb") as f:
            st.session_state.full_audio_bytes = f.read()

# Main chat interface
with col_chat:
    # Create a scrollable container with custom CSS
    st.markdown(
        """
        <style>
        .scrollable-chat {
            max-height: 500px;
            overflow-y: auto;
            padding-right: 15px; /* Add padding to avoid scrollbar overlap */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    with st.container(height=600):
        # Wrap messages in a div with the scrollable class
        st.markdown('<div class="scrollable-chat">', unsafe_allow_html=True)
        
        for message in st.session_state.messages:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(message["content"])
            elif message["role"] == "assistant":
                # Determine the persona type
                persona_type = input_values["template"]
                avatar_icon = PERSONA_ICONS.get(persona_type, "🤖")

                with st.chat_message("assistant", avatar=avatar_icon):
                    st.markdown(message["content"])
        
        


        # Chat input section
        if user_input := st.chat_input("💭 Type your message here..."):
            # Fill chat input with transcription and visual emotion if available

            # On first message, lock sidebar
            if not st.session_state.messages:
                st.session_state.chat_started = True  # Trigger sidebar lock

            # Add user message
            store_message_in_db_async("user", user_input)
            data = fetch_data_from_db()
            store_embeddings(data)
            st.session_state.rag_recent_notes = retrieve_context("Based on the user message, retrieve relevant context that can be use as a response."+ user_input)
            memories = str(st.session_state.rag_recent_notes)
            print("Retrieving memories: " + memories)
            if st.session_state.transcription and st.session_state.visual_emotion:
                user_chat_message_with_context = (
                    st.session_state.transcription
                    + " (The speaker looks " + st.session_state.visual_emotion
                    + " and sounds " + st.session_state.audible_emotion + " while saying this)"
                    + "\n (Here are some recent notes to help you respond to this current message:" + memories + ")"
                )
            else:
                user_chat_message_with_context = user_input + "\n (Here are some recent notes to help you respond to this current message:" + memories + ")"
            st.session_state.messages.append({"role": "user", "content": user_chat_message_with_context})
            with st.chat_message("user"):
                st.markdown(user_input.split("User: ")[-1] if "User: " in user_input else user_input)

            # Call the API and handle the response
            message_placeholder = st.empty()
            stats_placeholder = st.empty()  # Placeholder for real-time metadata display
            full_response = ""  # To store the full response incrementally
            unprocessed_content = ""  # Keeps track of content not yet parsed (character dialogue)
            thought_messages = []  # Store <think> messages separately (character thinking iff using reasoning model <think> tokens)
            character_response = [] # store character's actual response after think token

            # Function to dynamically parse <think> tokens
            def dynamic_parse_thoughts(content_so_far, new_line):
                messages = []
                combined_content = content_so_far + new_line  # Combine previous and current line

                while "<think>" in combined_content and "</think>" in combined_content:
                    start_idx = combined_content.find("<think>")
                    end_idx = combined_content.find("</think>") + len("</think>")

                    # Extract the thought content
                    thought_content = combined_content[start_idx + len("<think>"):end_idx - len("</think>")].strip()
                    messages.append({"type": "thought", "content": thought_content})

                    # Remove processed thought from combined content
                    combined_content = combined_content[:start_idx] + combined_content[end_idx:]

                # Remaining unprocessed content
                return messages, combined_content.strip()

            try:
                api_url = "https://top-mule-adequately.ngrok-free.app/api/chat"
            
                payload = {
                    "model": selected_model,
                    "messages": [{"role": "system", "content":  st.session_state.preprompt}] + st.session_state.messages,
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

                with st.spinner("Generating..."):

                    response_time = None
                    token_count = 0  # Track total tokens processed
                    eval_count = 0
                    eval_duration = 1
                    prompt_eval_count = 0
                    prompt_eval_duration = 1
                    total_duration = 1

                    response = requests.post(api_url, json=payload, stream=True)
                    response_time = time.time() - start_time

                    for line in response.iter_lines(decode_unicode=True):
                        if line.strip():
                            try:
                                parsed_line = json.loads(line)
                                if "message" in parsed_line and "content" in parsed_line["message"]:
                                    new_content = parsed_line["message"]["content"]

                                    # Parse thoughts first before displaying any normal content
                                    parsed_messages, unprocessed_content = dynamic_parse_thoughts(unprocessed_content, new_content)

                                    # Store thought messages separately to display them first
                                    for message in parsed_messages:
                                        if message["type"] == "thought":
                                            thought_messages.append(message["content"])

                                    # Append normal response to full response
                                    full_response += new_content

                                    # Track token speed
                                    token_count += len(new_content.split())
                                    elapsed_time = time.time() - start_time
                                    tokens_per_sec = token_count / elapsed_time if elapsed_time > 0 else 0

                                    # st.sidebar.write(f"⚡ **Real-time Tokens per Second:** {tokens_per_sec:.2f}")

                                    # Update placeholder with unprocessed normal content
                                    message_placeholder.markdown(unprocessed_content)
                                    # Update real-time statistics
                                    stats_placeholder.markdown(
                                        f"""
                                        **🕒 Real-time Response Stats:**  
                                        - ⏳ **Elapsed Time:** {elapsed_time:.2f}s  
                                        - 🔢 **Tokens Generated:** {token_count}  
                                        - ⚡ **Tokens per Second:** {tokens_per_sec:.2f} tok/s  
                                        """
                                    )

                                # Extract metadata from the **final** response
                                if "done" in parsed_line and parsed_line["done"]:
                                    response_time = time.time() - start_time
                                    eval_count = parsed_line.get("eval_count", 0)
                                    eval_duration = parsed_line.get("eval_duration", 1)
                                    prompt_eval_count = parsed_line.get("prompt_eval_count", 0)
                                    prompt_eval_duration = parsed_line.get("prompt_eval_duration", 1)
                                    total_duration = parsed_line.get("total_duration", 1)

                            except Exception as e:
                                st.error(f"Error parsing line: {e}")

                # Finalizing statistics
                eval_speed = eval_count / (eval_duration / 1e9) if eval_duration > 0 else 0
                prompt_speed = prompt_eval_count / (prompt_eval_duration / 1e9) if prompt_eval_duration > 0 else 0
                total_speed = (eval_count + prompt_eval_count) / (total_duration / 1e9) if total_duration > 0 else 0

                # Store the final message and stats in db and session state
                store_message_in_db_async("assistant", full_response)
                message_data = {
                    "role": "assistant",
                    "content": full_response,
                    "metadata": {
                        "response_time": response_time,
                        "response_length": len(full_response),
                        "status_code": response.status_code,
                        "tokens_generated": eval_count,
                        "tokens_per_second": eval_speed,
                        "prompt_tokens": prompt_eval_count,
                        "prompt_tokens_per_second": prompt_speed,
                        "total_tokens_per_second": total_speed
                    }
                }

                stats_placeholder.empty()
                message_placeholder.empty()
                # Display thought messages first
                for thought in thought_messages:
                    with st.chat_message("assistant", avatar="🧠"):
                        st.markdown(f"**Thinking:** {thought}")

                # Display final response after all thoughts
                with st.chat_message("assistant", avatar=PERSONA_ICONS.get(input_values["template"], "🤖")):
                    st.markdown(unprocessed_content)

                # Convert unprocessed_content (character dialogue) to speech and play it
                def autoplay_audio(file_path: str):
                    with open(file_path, "rb") as f:
                        data = f.read()
                        b64 = base64.b64encode(data).decode()
                        md = f"""
                            <audio controls autoplay="true">
                            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                            </audio>
                            """
                        st.markdown(
                            md,
                            unsafe_allow_html=True,
                        )

                try:
                    wav_file = text_to_speech_to_wav(unprocessed_content, input_values["voice_id"], use_TTS=True)
                    autoplay_audio(wav_file)
                    #print(wav_file)
                except Exception as e:
                    st.error(f"Error generating or playing audio: {e}")

                #print(full_response)

                print("Summarizing into DB Memory")
                db_queue.put({"job_type": "summarize", "conversation_text": "User: " + user_input + "\nAssistant: " + full_response})
                # Save the full response to session state after streaming is complete
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                st.session_state.api_stats.append({
                    "response_time": response_time,
                    "response_length": len(full_response),
                    "status_code": response.status_code,
                })

                # st.session_state.messages.append(message_data)
                # Add a dropdown to show message data after messages
                with st.expander("📊 Message Metadata"):
                    st.write(message_data["metadata"])

                # Add a dropdown to show all recent memories retrieved to make this prompt
                with st.expander("🧠 Recent Context"):
                    st.write(st.session_state.rag_recent_notes)

            except requests.exceptions.RequestException as e:
                st.error(f"API error: {e}")
                st.session_state.messages.append({"role": "assistant", "content": "Error retrieving response."})
                st.session_state.api_stats.append({
                    "response_time": None,
                    "response_length": 0,
                    "status_code": "Error",
                })
        st.markdown('</div>', unsafe_allow_html=True)

with col_video:
    st.markdown("")
    top_emotion = None
    emotion_percentages = None
    if st.button("Record Emotion 📷"):
        start_recording()
        st.session_state.message_sent = False  # Allow new messages to be sent on next recording

    if st.session_state.full_video_bytes:
        # Display the video safely
        video = open(st.session_state.full_video_path, "rb")
        video_bytes = video.read()
        st.video(video_bytes)

        # Process Video
        emotion_predictions = process_video(st.session_state.full_video_path, frames_per_second=20)
        # Calculate emotion percentages
        emotion_counts = Counter(emotion_predictions)
        total_frames = sum(emotion_counts.values())
        emotion_percentages = {emotion: (count / total_frames) * 100 for emotion, count in emotion_counts.items()}
        # Determine the most frequent (top) detected emotion
        top_emotion = max(emotion_percentages, key=emotion_percentages.get)
        st.session_state.visual_emotion = top_emotion  # Save to session state

        # Transcribe the audio
        if st.session_state.full_audio_path:
            detected_emotion, audio_transcription = process_audio_file(st.session_state.full_audio_path)
            print("Transcription: ", audio_transcription)
            print("Audible Emotion: ", detected_emotion)
            st.session_state.transcription = audio_transcription
            st.session_state.audible_emotion = detected_emotion

    # Display Visual Emotion Results
    st.subheader("Video Classification Stats")

    if(emotion_percentages and top_emotion):
        # Display detected emotion summary
        st.subheader(f"**Detected Emotion:** {top_emotion} ({emotion_percentages[top_emotion]:.2f}%)")
        # Display bar chart
        emotion_data = pd.DataFrame({
            'Emotion': list(emotion_percentages.keys()),
            'Percentage': list(emotion_percentages.values())
        }).set_index('Emotion')
        st.subheader("📊 Detected Emotion Percentages")
        st.bar_chart(emotion_data)


    # Display Audio Results
    st.subheader("Audio Classification Stats")
    if st.session_state.transcription:
        st.info(f"**Detected Emotion:** {st.session_state.audible_emotion}")
        st.success(f"**Transcription:** {st.session_state.transcription}")


# Very hacky way to add a default value to the chat input
# This script will be executed every time a new message is added through col_video
if st.session_state.visual_emotion and st.session_state.transcription and not st.session_state.get("message_sent", False):
    st.session_state.message_sent = True  # Prevent re-execution
    user_chat_message_with_context = (
        st.session_state.transcription
        + " (The speaker looks " + st.session_state.visual_emotion
        + " and sounds " + st.session_state.audible_emotion + " while saying this)"
    )
    store_message_in_db_async("user", user_chat_message_with_context)
    data = fetch_data_from_db()
    store_embeddings(data)
    st.session_state.rag_recent_notes = retrieve_context(
        "Based on the user message, retrieve relevant context that can be used as a response. "
        + user_chat_message_with_context
    )
    memories = str(st.session_state.rag_recent_notes)
    print("Retrieving memories: " + memories)
    user_chat_message_with_context += "\n (Here are some recent notes to help you respond to this current message: " + memories + ")"
    #st.session_state.messages.append({"role": "user", "content": user_chat_message_with_context})
    print("Sending Message through JS into chat_input")
    # Properly escape the text using json.dumps() to handle quotes, newlines, and special characters
    safe_message = json.dumps(user_chat_message_with_context)
    js = f"""
    <script>
        function insertText(dummy_var_to_force_repeat_execution) {{
            var chatInput = parent.document.querySelector('textarea[data-testid="stChatInputTextArea"]');
            var nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
            nativeInputValueSetter.call(chatInput, {safe_message});  // Using safe_message to prevent syntax errors
            var event = new Event('input', {{ bubbles: true }});
            chatInput.dispatchEvent(event);
            var submitButton = parent.document.querySelector('button[data-testid="stChatInputSubmitButton"]');
            console.log("Clicking submit button");
            submitButton.click();
        }}
        insertText({len(user_chat_message_with_context)});
    </script>
    """
    st.components.v1.html(js)
else:
    st.error("Something went wrong. Please record a video and audio before sending a message")