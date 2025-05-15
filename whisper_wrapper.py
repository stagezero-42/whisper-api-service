import whisper
import os
import torch
import io  # For creating in-memory text streams

# Determine device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Whisper wrapper: Using device: {DEVICE}")

# Cache for different models
model_cache = {}


def load_whisper_model(model_name="base"):
    global model_cache
    if model_name in model_cache:
        print(f"Using cached Whisper model: {model_name} on {DEVICE}.")
        return model_cache[model_name]

    print(f"Loading Whisper model: {model_name}...")
    current_model = None
    try:
        # Check if GPU is available for "turbo" or other large models if that's a convention
        # For now, just load as requested.
        effective_device = DEVICE
        if model_name == "turbo" and DEVICE != "cuda":
            print("Warning: 'turbo' model selected but no CUDA GPU found. Attempting to load on CPU.")
            # effective_device = "cpu" # or let it fail if turbo is strictly GPU

        current_model = whisper.load_model(model_name, device=effective_device)
        model_cache[model_name] = current_model
        print(f"Model {model_name} loaded successfully on {effective_device}.")
    except Exception as e:
        print(f"Error loading model {model_name} on {DEVICE}: {e}")
        # Try CPU if primary device fails (especially if it was CUDA)
        if DEVICE == "cuda" and model_name != "turbo":  # Avoid retrying turbo on CPU if already warned
            try:
                print(f"Attempting to load model {model_name} on CPU...")
                current_model = whisper.load_model(model_name, device="cpu")
                model_cache[model_name] = current_model
                print(f"Model {model_name} loaded successfully on CPU.")
            except Exception as e_cpu:
                print(f"Error loading model {model_name} on CPU: {e_cpu}")
        if current_model is None:
            print(f"Failed to load model {model_name} on any device.")
    return current_model


def format_timestamp(seconds: float, always_include_hours: bool = False, decimal_marker: str = '.'):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)
    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000
    minutes = milliseconds // 60_000
    milliseconds %= 60_000
    seconds_val = milliseconds // 1_000
    milliseconds %= 1_000
    if always_include_hours or hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds_val:02d}{decimal_marker}{milliseconds:03d}"
    else:
        return f"{minutes:02d}:{seconds_val:02d}{decimal_marker}{milliseconds:03d}"


def to_srt(result_segments, as_dict=False):
    if not as_dict:
        srt_io = io.StringIO()
        for i, segment in enumerate(result_segments):
            srt_io.write(f"{i + 1}\n")
            srt_io.write(f"{format_timestamp(segment['start'], always_include_hours=True, decimal_marker=',')} --> ")
            srt_io.write(f"{format_timestamp(segment['end'], always_include_hours=True, decimal_marker=',')}\n")
            srt_io.write(f"{segment['text'].strip()}\n\n")
        return srt_io.getvalue()
    else:
        srt_dict = {}
        for i, segment in enumerate(result_segments):
            timestamp_line = (
                f"{format_timestamp(segment['start'], always_include_hours=True, decimal_marker=',')} --> "
                f"{format_timestamp(segment['end'], always_include_hours=True, decimal_marker=',')}")
            srt_dict[str(i + 1)] = [timestamp_line, segment['text'].strip()]
        return srt_dict


def to_vtt(result_segments, as_dict=False):
    if not as_dict:
        vtt_io = io.StringIO()
        vtt_io.write("WEBVTT\n\n")
        for segment in result_segments:
            vtt_io.write(
                f"{format_timestamp(segment['start'], decimal_marker='.')} --> {format_timestamp(segment['end'], decimal_marker='.')}\n")
            vtt_io.write(f"{segment['text'].strip()}\n\n")
        return vtt_io.getvalue()
    else:
        vtt_dict = {}
        # VTT doesn't typically have sequence numbers in the content body like SRT dict keys,
        # but for consistency with the requested SRT dict structure, we'll use segment index as key.
        # The "WEBVTT" header is for the string format, not the dict.
        for i, segment in enumerate(result_segments):
            timestamp_line = f"{format_timestamp(segment['start'], decimal_marker='.')} --> {format_timestamp(segment['end'], decimal_marker='.')}"
            vtt_dict[str(i + 1)] = [timestamp_line, segment['text'].strip()]
        return vtt_dict


def to_tsv(result_segments):  # TSV is always a string
    tsv_content = io.StringIO()
    tsv_content.write("start\tend\ttext\n")  # Header
    for segment in result_segments:
        start_ms = int(segment['start'] * 1000)
        end_ms = int(segment['end'] * 1000)
        tsv_content.write(f"{start_ms}\t{end_ms}\t{segment['text'].strip()}\n")
    return tsv_content.getvalue()


def transcribe_audio(audio_path, model_name="base", task="transcribe", language=None,
                     initial_prompt=None, temperature=0.0, best_of=5,
                     word_timestamps=False, verbose=None):
    loaded_model = load_whisper_model(model_name)
    if not loaded_model:
        return {"error": f"Whisper model '{model_name}' could not be loaded."}

    if not os.path.exists(audio_path):
        return {"error": "Audio file not found."}

    transcribe_options = {
        "task": task,
        "verbose": verbose,
        "fp16": True if DEVICE == "cuda" else False
    }
    if language:
        transcribe_options["language"] = language
    if initial_prompt:
        transcribe_options["initial_prompt"] = initial_prompt

    # Ensure word_timestamps is explicitly True or False
    transcribe_options["word_timestamps"] = bool(word_timestamps)

    if temperature is not None:
        transcribe_options["temperature"] = temperature
    if best_of is not None:
        transcribe_options["best_of"] = best_of

    for opt in ["temperature"]:
        if opt in transcribe_options and transcribe_options[opt] is not None:
            try:
                transcribe_options[opt] = float(transcribe_options[opt])
            except ValueError:
                return {"error": f"Invalid value for {opt}: must be a float."}
    for opt in ["best_of"]:
        if opt in transcribe_options and transcribe_options[opt] is not None:
            try:
                transcribe_options[opt] = int(transcribe_options[opt])
            except ValueError:
                return {"error": f"Invalid value for {opt}: must be an integer."}
    try:
        print(f"Transcribing {audio_path} with options: {transcribe_options}")
        result = loaded_model.transcribe(audio_path, **transcribe_options)
        print("Transcription successful.")
        return result
    except Exception as e:
        import traceback
        print(f"Error during transcription: {e}\n{traceback.format_exc()}")
        return {"error": str(e)}

# (Keep the if __name__ == '__main__': block for direct testing if desired)