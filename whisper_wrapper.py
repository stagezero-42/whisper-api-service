import whisper
import os
import torch
import io  # For creating in-memory text streams

# Determine default device at module level if not passed explicitly
DEFAULT_DEVICE_WHISPER = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Whisper wrapper: Default device set to: {DEFAULT_DEVICE_WHISPER}")

# Cache for different models
model_cache = {}


def load_whisper_model(model_name="base", device=None):
    global model_cache

    # Determine effective device for this load attempt
    # If a device is passed, use it; otherwise, use the module's default.
    effective_device = device if device is not None else DEFAULT_DEVICE_WHISPER

    cache_key = f"{model_name}_{effective_device}"  # Device-specific cache key

    if cache_key in model_cache:
        print(f"Using cached Whisper model: {model_name} on {effective_device} (from RAM cache).")
        return model_cache[cache_key]

    print(f"Attempting to load Whisper model: {model_name} for device: {effective_device}...")
    current_model = None
    try:
        current_model = whisper.load_model(model_name, device=effective_device)
        model_cache[cache_key] = current_model
        print(f"Model '{model_name}' loaded successfully on {effective_device} and cached in RAM.")
    except Exception as e:
        print(f"Error loading model '{model_name}' on primary device '{effective_device}': {e}")
        # Fallback logic if primary device fails (e.g., if CUDA was attempted and failed)
        if effective_device == "cuda":
            print(f"Attempting to load model '{model_name}' on CPU as fallback...")
            try:
                cpu_cache_key = f"{model_name}_cpu"
                if cpu_cache_key in model_cache:  # Check if already CPU cached
                    print(f"Using cached Whisper model: {model_name} on cpu (from RAM cache after CUDA fail).")
                    return model_cache[cpu_cache_key]

                current_model = whisper.load_model(model_name, device="cpu")
                model_cache[cpu_cache_key] = current_model  # Cache it under CPU key
                print(f"Model '{model_name}' loaded successfully on CPU (fallback) and cached in RAM.")
            except Exception as e_cpu:
                print(f"Error loading model '{model_name}' on CPU (fallback): {e_cpu}")
                # If all attempts fail, current_model remains None
        # If initial attempt was CPU and failed, current_model is already None

    if current_model is None:
        print(f"Failed to load model '{model_name}' on any attempted device.")

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
    # Model will be loaded for CUDA if available, else CPU, by load_whisper_model's default behavior
    loaded_model = load_whisper_model(model_name=model_name)

    if not loaded_model:
        # load_whisper_model now prints detailed errors.
        return {"error": f"Whisper model '{model_name}' could not be loaded (check logs for details)."}

    # Determine the actual device the model was loaded on for fp16 setting
    # This requires the model object to store its device, or infer it.
    # Whisper model objects have a `device` attribute.
    actual_model_device_type = loaded_model.device.type
    print(f"Model '{model_name}' will be used on device: {actual_model_device_type}")

    if not os.path.exists(audio_path):
        return {"error": "Audio file not found."}

    transcribe_options = {
        "task": task,
        "verbose": verbose,
        # Set fp16 based on the device the model *actually* loaded on
        "fp16": True if actual_model_device_type == "cuda" else False
    }
    # ... (rest of your transcribe_audio options setup) ...
    if language: transcribe_options["language"] = language
    if initial_prompt: transcribe_options["initial_prompt"] = initial_prompt
    transcribe_options["word_timestamps"] = bool(word_timestamps)
    if temperature is not None: transcribe_options["temperature"] = temperature
    if best_of is not None: transcribe_options["best_of"] = best_of
    # ... (type conversion for temp and best_of) ...

    try:
        print(f"Transcribing {audio_path} with options: {transcribe_options} using model on {actual_model_device_type}")
        result = loaded_model.transcribe(audio_path, **transcribe_options)
        print("Transcription successful.")
        return result
    except Exception as e:
        import traceback
        print(f"Error during transcription: {e}\n{traceback.format_exc()}")
        return {"error": str(e)}

# (Keep the if __name__ == '__main__': block for direct testing if desired)