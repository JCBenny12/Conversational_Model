from dotenv import load_dotenv
import os
from elevenlabs.client import ElevenLabs
import re

def text_to_speech_to_wav(text, voice_id="JBFqnCBsd6RMkjVDRZzb", output_file="output.wav", use_TTS=True):
    if use_TTS:
        # Extract text between quotes
        match = re.search(r'\*\*(.*?)\*\*', text)
        if match:
            prefix = match.group(0)
            text = text.replace(prefix, '').strip()
        else:
            prefix = ''

        if '"' in text:
            text = ' '.join(re.findall(r'"(.*?)"', text))
        else:
            text = f'"{text}"'

        text = f'{prefix} {text}'.strip()
        load_dotenv()

        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError("API key not found. Please set the ELEVENLABS_API_KEY environment variable.")

        client = ElevenLabs(api_key=api_key)

        
        print("Generating text")
        audio_generator = client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id="eleven_flash_v2_5",
            output_format="mp3_44100_128",
        )
        # Save the audio to a file
        with open(output_file, "wb") as file:
            for chunk in audio_generator:
                file.write(chunk)

    return output_file

# Example usage
#if __name__ == "__main__":
#    wav_file = text_to_speech_to_wav("One day I am gonna grow wings. A chemical reaction. Hysterical and useless. Hysterical and.")
#    print(f"Audio saved to {wav_file}")