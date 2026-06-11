import os
import subprocess
import whisper

# --- Model Cache ---
_model_cache = {}

def load_whisper_model(model_name: str = "base"):
    """Loads and caches a Whisper speech recognition model."""
    if model_name not in _model_cache:
        print(f"Loading Whisper model: '{model_name}'")
        try:
            # The model will be downloaded automatically on first run
            # and stored in a cache directory (~/.cache/whisper).
            model = whisper.load_model(model_name)
            _model_cache[model_name] = model
            print(f"Whisper '{model_name}' model loaded successfully.")
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            return None
    return _model_cache[model_name]

def convert_oga_to_wav(oga_path: str, wav_path: str) -> bool:
    """
    Converts an OGG/Opus audio file to a WAV file compatible with Whisper.
    Requires FFmpeg to be installed on the system.
    
    :param oga_path: Path to the input .oga file.
    :param wav_path: Path for the output .wav file.
    :return: True if conversion was successful, False otherwise.
    """
    try:
        # Command to convert OGA to WAV (16kHz, 16-bit PCM, mono)
        command = [
            'ffmpeg',
            '-i', oga_path,
            '-ar', '16000',          # Sample rate 16kHz
            '-ac', '1',              # Mono audio channel
            '-acodec', 'pcm_s16le',  # 16-bit PCM codec
            '-y',                    # Overwrite output file if it exists
            wav_path
        ]
        print(f"Running FFmpeg conversion: {' '.join(command)}")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("FFmpeg conversion successful.")
        return True
    except FileNotFoundError:
        print("ERROR: `ffmpeg` command not found. Please ensure FFmpeg is installed and in your system's PATH.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"ERROR: FFmpeg failed with error:
{e.stderr}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during conversion: {e}")
        return False

def transcribe_audio(audio_path: str) -> str | None:
    """
    Transcribes an audio file to text using the Whisper model.
    
    :param audio_path: Path to the audio file.
    :return: The transcribed text, or None if failed.
    """
    # Using the 'base' model as it's a good starting point.
    # Other options: 'tiny', 'small', 'medium', 'large'
    model = load_whisper_model("base")
    if not model:
        return None

    try:
        print(f"Transcribing audio file: {audio_path}")
        # Perform transcription
        result = model.transcribe(audio_path, fp16=False) # fp16=False for CPU
        
        transcribed_text = result['text']
        print(f"Transcription result: '{transcribed_text}'")
        return transcribed_text
    except Exception as e:
        print(f"An error occurred during transcription: {e}")
        return None

if __name__ == '__main__':
    print("--- Running Speech-to-Text Test (Whisper) ---")
    
    # This test checks if the model can be loaded.
    # The first time you run this, it will download the 'base' model.
    print("Attempting to load Whisper 'base' model...")
    load_whisper_model("base")
    
    print("
To test full functionality, you need FFmpeg installed.")
    print("The bot will call these functions when it receives a voice message.")
