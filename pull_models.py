# download_prompt_models.py
import whisper
import torch
import os


def get_user_confirmation(prompt_message):
    """Gets a yes/no confirmation from the user."""
    while True:
        response = input(f"{prompt_message} (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("Invalid input. Please enter 'y' or 'n'.")


def download_model_with_confirmation(model_name, device_to_try):
    """Downloads a specific model after user confirmation."""
    print(f"\n--- Model: {model_name} ---")
    if not get_user_confirmation(f"Do you want to attempt to download/verify model '{model_name}'?"):
        print(f"Skipping model '{model_name}'.")
        return False

    print(f"Attempting to download/load model: {model_name} on {device_to_try}...")
    try:
        # This will download the model if not already cached, and then load it.
        model = whisper.load_model(model_name, device=device_to_try)
        print(f"Model '{model_name}' is available and loaded successfully on {device_to_try}.")
        # We don't need to keep the model in memory here, so we can delete it
        # to free up resources if downloading many large models sequentially.
        del model
        if device_to_try == "cuda":
            torch.cuda.empty_cache()  # Clear CUDA cache if models were loaded to GPU
        return True
    except Exception as e:
        print(f"Error with model '{model_name}' on {device_to_try}: {e}")
        # If the primary device attempt failed (especially if it was CUDA), try CPU as a fallback.
        if device_to_try == "cuda":
            print(f"Attempting to download/load '{model_name}' using CPU as fallback...")
            if not get_user_confirmation(
                    f"Initial attempt for '{model_name}' on CUDA failed. Try downloading/loading '{model_name}' using CPU?"):
                print(f"Skipping CPU fallback for '{model_name}'.")
                return False
            try:
                model = whisper.load_model(model_name, device="cpu")
                print(f"Model '{model_name}' is available and loaded successfully on CPU.")
                del model
                return True
            except Exception as e_cpu:
                print(f"Error with model '{model_name}' on CPU: {e_cpu}")
                return False
        return False  # Return False if the initial attempt (on CPU or non-CUDA) failed


def main():
    # Determine preferred device for loading checks
    # The downloaded model file itself is device-agnostic.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"OpenAI Whisper Model Downloader")
    print(f"---------------------------------")
    # The exact cache path can be determined by whisper.utils.get_cache_dir() or similar internal functions,
    # but it's commonly ~/.cache/whisper.
    default_cache_path = os.path.join(os.path.expanduser("~"), ".cache", "whisper")
    print(f"Models will be downloaded to (typically): {default_cache_path}")
    print(f"Using preferred device for loading checks: {device}")
    print(f"---------------------------------\n")

    # List of models to consider, now including "turbo"
    # "large" usually points to the latest large version (e.g., large-v3).
    models_to_consider = [
        "tiny.en", "tiny",
        "base.en", "base",
        "small.en", "small",
        "medium.en", "medium",
        "large",  # Should get the latest large (e.g., large-v3 if that's current)
        "turbo"  # As per user's information, this is now a recognized model
        # "large-v1",   # Uncomment if you specifically need older versions
        # "large-v2",   # Uncomment if you specifically need older versions
        # "large-v3",   # Usually covered by "large", but can be explicit if needed
    ]

    downloaded_count = 0
    attempted_count = 0

    for model_name in models_to_consider:
        attempted_count += 1
        if download_model_with_confirmation(model_name, device):
            downloaded_count += 1

    print("\n--- Download Summary ---")
    print(f"Attempted to download/verify {attempted_count} models.")
    print(f"Successfully downloaded/verified {downloaded_count} models.")
    print(f"Models are cached in (typically): {default_cache_path}")
    print("------------------------")


if __name__ == "__main__":
    main()
