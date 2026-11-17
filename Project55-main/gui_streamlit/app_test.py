import streamlit as st
from streamlit_option_menu import option_menu
from textToSpeech import text_to_speech_to_wav
from typing import Dict
import requests
import json
import time
import base64
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os

# -------------------------------
# SET PAGE CONFIGURATION
# -------------------------------
st.set_page_config(page_title="AI Persona Chat", layout="wide")

# -------------------------------
# SESSION STATE INITIALIZATION
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_stats" not in st.session_state:
    st.session_state.api_stats = []
if "preprompt" not in st.session_state:
    st.session_state.preprompt = ""
if "chat_started" not in st.session_state:
    st.session_state.chat_started = False
if "download_status" not in st.session_state:
    st.session_state.download_status = ""
if "detected_emotion" not in st.session_state:
    st.session_state.detected_emotion = "Neutral"
if "enable_video_emotion" not in st.session_state:
    st.session_state.enable_video_emotion = False

# -------------------------------
# GLOBAL VARIABLES
# -------------------------------
# Define icons for persona templates
PERSONA_ICONS = {
    "Therapist": "🧠",
    "Tech Mentor": "💻",
    "Creative Writing Coach": "✍️",
    "Batman": "🦇",
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
TONE_OPTIONS = ["Calm", "Friendly", "Serious", "Encouraging", "Authoritative"]
STYLE_OPTIONS = [
    "Uses reflective questions and gentle guidance to encourage self-discovery",
    "Explains complex topics with real-world examples",
    "Employs vivid descriptions and metaphors to inspire creativity",
    "Listens actively and responds with empathy",
    "Speaks with precise, calculated words"
]

# -------------------------------
# LOAD EMOTION RECOGNITION MODEL
# -------------------------------
model_path = os.path.join(os.path.dirname(__file__), "..", "video_integration", "emotion_recognition_model.h5")
print(model_path)
if os.path.exists(model_path):
    try:
        emotion_model = load_model(model_path)
        emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    except Exception as e:
        st.error(f"Error loading emotion recognition model: {e}")
        emotion_model = None
        emotions = {}
        face_cascade = None
else:
    st.warning(f"Model file '{model_path}' not found. Video emotion recognition will be disabled.")
    emotion_model = None
    emotions = {}
    face_cascade = None

def detect_emotion(image: Image.Image) -> str:
    """
    Detects emotion from a PIL image using the first detected face.
    Returns a string representing the detected emotion, or an error message.
    """
    if not emotion_model or not face_cascade:
        return "Neutral"
    img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    if len(faces) == 0:
        return "No face detected"
    x, y, w, h = faces[0]
    face_region = gray[y:y+h, x:x+w]
    face_resized = cv2.resize(face_region, (48,48))
    face_normalized = face_resized.astype("float32") / 255.0
    face_input = np.reshape(face_normalized, (1,48,48,1))
    prediction = emotion_model.predict(face_input)
    predicted_class = np.argmax(prediction)
    return emotions.get(predicted_class, "Neutral")

# -------------------------------
# FUNCTION TO RESET CHAT
# -------------------------------
def reset_chat():
    st.session_state.messages = []
    st.session_state.api_stats = []
    st.session_state.preprompt = ""
    st.session_state.chat_started = False
    st.session_state.download_status = ""
    st.session_state.detected_emotion = "Neutral"

# -------------------------------
# FUNCTIONS FOR MODEL MANAGEMENT
# -------------------------------
def list_models():
    list_url = "https://top-mule-adequately.ngrok-free.app/api/tags"
    try:
        response = requests.get(list_url)
        if response.status_code == 200:
            data = response.json()
            local_models = [model["name"] for model in data.get("models", [])]
            unique_models = {}
            for model in local_models:
                base_name = model.split(":")[0]
                if base_name not in unique_models or model.endswith(":latest"):
                    unique_models[base_name] = model
            additional_models = [
                "DeepSeek-R1-Distill-Llama-8B", "deepseek-r1:8b",
                "taozhiyuai/llama-3-8b-ultra-instruct:q2_k", "llama3",
                "llama3.2", "llama3.2:1b", "mistral", "gemma:2b"
            ]
            non_duplicate_models = [m for m in additional_models if m.split(":")[0] not in unique_models]
            return list(unique_models.values()) + non_duplicate_models
        else:
            st.error(f"Failed to list models. Status code: {response.status_code}")
            return []
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to the API: {e}")
        return []

def verify_and_download_model(model_name):
    show_url = "https://top-mule-adequately.ngrok-free.app/api/show"
    pull_url = "https://top-mule-adequately.ngrok-free.app/api/pull"
    try:
        response = requests.post(show_url, json={"model": model_name})
        if response.status_code == 200:
            data = response.json()
            if data.get("modelfile"):
                st.success(f"Model '{model_name}' is already installed and ready to use.")
                return True
            else:
                st.warning(f"Model '{model_name}' is not installed. Starting download...")
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

# -------------------------------
# SIDEBAR: MODEL SELECTION, HYPERPARAMETERS, & PERSONA BUILDER
# -------------------------------
with st.sidebar:
    st.header("Select Ollama Model")
    available_models = list_models()
    selected_model = st.selectbox("Choose a model to load:", available_models)
    if st.button("Verify and Load Model"):
        verify_and_download_model(selected_model)
    
    with st.expander("⚙️ Advanced Hyperparameters", expanded=False):
        st.write("Fine-tune the model's behavior.")
        temperature = st.slider("Temperature (Higher = More creative, Lower = More deterministic)", 0.0, 2.0, 0.8, 0.05)
        top_p = st.slider("Top P (Lower = Focused, Higher = Diverse)", 0.0, 1.0, 0.9, 0.05)
        top_k = st.slider("Top K (Limits vocabulary size)", 1, 100, 50, 1)
        repeat_penalty = st.slider("Repeat Penalty", 0.5, 2.0, 0.9, 0.05)
        max_tokens = st.slider("Max Tokens", 64, 2048, 512, 64)
    
    # Checkbox to enable video-based emotion recognition
    enable_video_emotion = st.checkbox("Enable Video Emotion Recognition", value=False)
    
    def render_sidebar_inputs():
        st.header("👤 Persona Builder")
        disabled = st.session_state.chat_started
        if disabled:
            st.info("⚠️ Persona builder is locked while chat is active. Click 'New Chat' to start fresh.")
        selected_template = st.selectbox("📝 Select Preset Template", ["Custom"] + list(PERSONA_ICONS.keys()), index=0, disabled=disabled)
        values = {"template": selected_template, "name": "", "description": "", "backstory": "", "tone": "", "style": "", "voice_id": ""}
        if selected_template != "Custom":
            if selected_template == "Therapist":
                values = {
                    "template": "Therapist",
                    "name": "Dr. Mira Wells",
                    "description": "A compassionate therapist.",
                    "backstory": "20+ years of experience.",
                    "tone": "Calm",
                    "style": "Reflective",
                    "voice_id": "aMSt68OGf4xUZAnLpTU8"
                }
            elif selected_template == "Tech Mentor":
                values = {
                    "template": "Tech Mentor",
                    "name": "Oliver Chen",
                    "description": "A seasoned software engineer and mentor.",
                    "backstory": "Graduated from MIT.",
                    "tone": "Friendly",
                    "style": "Direct",
                    "voice_id": "L0Dsvb3SLTyegXwtm47J"
                }
            # Additional templates can be added similarly.
        else:
            values = {
                "template": "Custom",
                "name": st.text_input("🏷️ Character Name", placeholder="e.g., Dr. Smith", disabled=disabled),
                "description": st.text_area("📋 Character Description", placeholder="e.g., I am a compassionate therapist...", disabled=disabled),
                "backstory": st.text_area("📚 Character Backstory", placeholder="e.g., Graduated from Harvard...", disabled=disabled),
                "tone": st.selectbox("🎭 Tone", TONE_OPTIONS, disabled=disabled),
                "style": st.selectbox("✨ Style", STYLE_OPTIONS, disabled=disabled),
                "voice_id": st.text_input("🔊 Voice ID", placeholder="e.g., JBFqnCBsd6RMkjVDRZzb", disabled=disabled)
            }
        return values

    input_values = render_sidebar_inputs()
    if input_values["name"] or input_values["description"] or input_values["backstory"]:
        preprompt = f"""
ROLEPLAY SYSTEM DIRECTIVE:

You are an autonomous character with free will. Your responses must strictly adhere to your predefined personality, motivations, and past interactions.

### Character Context:
- Character Name: {input_values["name"]}
- Description: {input_values["description"]}
- Background: {input_values["backstory"]}
- Tone: {input_values["tone"]}
- Style: {input_values["style"]}

### Consistency & Realism:
1. Maintain core personality traits.
2. Uphold memory and continuity.
3. React naturally to emotions and events.
4. Make decisions in line with your motivations.
5. Express independent thoughts.
6. Resist or refuse when appropriate.
7. Grow naturally over time.
8. React realistically to experiences.
9. Process complex emotional and social cues.
10. Form meaningful relationships.
11. Consider consequences.
12. Never break character or provide meta-commentary.

Stay true to your identity.
"""
        st.session_state.preprompt = preprompt
        with st.expander("🔍 Debug - View Current Preprompt"):
            st.code(preprompt)
    with st.expander("📜 Chat Transcript & Stats"):
        if st.session_state.messages:
            transcript = ""
            for i, message in enumerate(st.session_state.messages):
                role = "User" if message["role"] == "user" else "Bot"
                content = message["content"]
                transcript += f"{role}: {content}\n"
                if role == "Bot" and i < len(st.session_state.api_stats):
                    stats = st.session_state.api_stats[i]
                    transcript += f"  Response Time: {stats['response_time']:.2f}s, Length: {stats['response_length']} chars, HTTP Status: {stats['status_code']}\n"
                transcript += "-" * 50 + "\n"
            st.code(transcript)
        else:
            st.write("No messages yet.")

# -------------------------------
# MAIN TITLE & NEW CHAT BUTTON
# -------------------------------
selected_icon = PERSONA_ICONS.get(input_values["template"], "🤖")
col1, col2 = st.columns([6, 1])
with col1:
    st.title(f"{selected_icon} Empathic AI Persona Chatbot")
with col2:
    if st.button("🔄 New Chat", use_container_width=True):
        reset_chat()

# Main layout with two columns
col_chat, col_video = st.columns([3, 1])

# -------------------------------
# VIDEO-BASED EMOTION DETECTION
# -------------------------------
with col_video:
    if enable_video_emotion:
        st.header("📷 Webcam Emotion Detection")
        st.write("Click 'Refresh Emotion' to capture a new image and update your detected emotion.")
        # Custom CSS to reduce camera widget size
        st.markdown("""
        <style>
        [data-testid="stCameraInput"] {
            width: 300px !important;
            height: 225 !important;
        }
        [data-testid="stCameraInput"] video {
            max-width: 300px !important;
            max-height: 225px !important;
        }
        </style>
        """, unsafe_allow_html=True)

        captured_image = st.camera_input("Camera Input")
        if captured_image is not None:
            image = Image.open(captured_image)
            st.image(image, caption="Full Image", width=250)
        else:
            st.info("Waiting for an image capture...")

        if st.button("Refresh Emotion"):
            if captured_image is None:
                st.warning("No image captured yet.")
            else:
                image = Image.open(captured_image)
                new_emotion = detect_emotion(image)
                st.session_state.detected_emotion = new_emotion
                st.success(f"Detected Emotion: {new_emotion}")
    else:
        st.info("Video-based emotion recognition is disabled. (Enable it in the sidebar.)")

with col_chat:
    # -------------------------------
    # CHAT INTERFACE DISPLAY
    # -------------------------------
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # -------------------------------
    # CHAT INPUT & RESPONSE STREAMING
    # -------------------------------
    if user_input := st.chat_input("💭 Type your message here..."):
        if not st.session_state.messages:
            st.session_state.chat_started = True
            # Prepend preprompt and include detected emotion in the first message
            user_input = f"{st.session_state.preprompt}\n\nUser: {user_input}\n\nDetected Emotion: {st.session_state.detected_emotion}"
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input.split("User: ")[-1])

        message_placeholder = st.empty()
        stats_placeholder = st.empty()
        full_response = ""
        unprocessed_content = ""
        thought_messages = []

        def dynamic_parse_thoughts(content_so_far, new_line):
            messages = []
            combined_content = content_so_far + new_line
            while "<think>" in combined_content and "</think>" in combined_content:
                start_idx = combined_content.find("<think>")
                end_idx = combined_content.find("</think>") + len("</think>")
                thought_content = combined_content[start_idx + len("<think>"):end_idx - len("</think>")].strip()
                messages.append({"type": "thought", "content": thought_content})
                combined_content = combined_content[:start_idx] + combined_content[end_idx:]
            return messages, combined_content.strip()

        try:
            api_url = "https://top-mule-adequately.ngrok-free.app/api/chat"
            payload = {
                "model": selected_model,
                "messages": st.session_state.messages,
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
                response = requests.post(api_url, json=payload, stream=True)
                response_time = time.time() - start_time
                token_count = 0
                for line in response.iter_lines(decode_unicode=True):
                    if line.strip():
                        try:
                            parsed_line = json.loads(line)
                            if "message" in parsed_line and "content" in parsed_line["message"]:
                                new_content = parsed_line["message"]["content"]
                                parsed_messages, unprocessed_content = dynamic_parse_thoughts(unprocessed_content, new_content)
                                for message in parsed_messages:
                                    if message["type"] == "thought":
                                        thought_messages.append(message["content"])
                                full_response += new_content
                                token_count += len(new_content.split())
                                elapsed_time = time.time() - start_time
                                tokens_per_sec = token_count / elapsed_time if elapsed_time > 0 else 0
                                message_placeholder.markdown(unprocessed_content)
                                stats_placeholder.markdown(
                                    f"**Elapsed Time:** {elapsed_time:.2f}s  \n**Tokens Generated:** {token_count}  \n**Tokens per Second:** {tokens_per_sec:.2f}"
                                )
                            if "done" in parsed_line and parsed_line["done"]:
                                response_time = time.time() - start_time
                        except Exception as e:
                            st.error(f"Error parsing line: {e}")
            message_data = {
                "role": "assistant",
                "content": full_response,
                "metadata": {
                    "response_time": response_time,
                    "response_length": len(full_response),
                    "status_code": response.status_code,
                }
            }
            stats_placeholder.empty()
            message_placeholder.empty()
            for thought in thought_messages:
                with st.chat_message("assistant", avatar="🧠"):
                    st.markdown(f"**Thinking:** {thought}")
            with st.chat_message("assistant", avatar=PERSONA_ICONS.get(input_values["template"], "🤖")):
                st.markdown(unprocessed_content)
            def autoplay_audio(file_path: str):
                with open(file_path, "rb") as f:
                    data = f.read()
                    b64 = base64.b64encode(data).decode()
                    md = f"""
                        <audio controls autoplay="true">
                        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                        </audio>
                        """
                    st.markdown(md, unsafe_allow_html=True)
            try:
                wav_file = text_to_speech_to_wav(unprocessed_content, input_values["voice_id"], use_TTS=False)
                autoplay_audio(wav_file)
            except Exception as e:
                st.error(f"Error generating or playing audio: {e}")
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.session_state.api_stats.append({
                "response_time": response_time,
                "response_length": len(full_response),
                "status_code": response.status_code,
            })
            with st.expander("📊 Message Metadata"):
                st.write(message_data["metadata"])
        except requests.exceptions.RequestException as e:
            st.error(f"API error: {e}")
            st.session_state.messages.append({"role": "assistant", "content": "Error retrieving response."})
            st.session_state.api_stats.append({
                "response_time": None,
                "response_length": 0,
                "status_code": "Error",
            })
