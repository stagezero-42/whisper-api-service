# celery_worker_app.py
import multiprocessing
import sys # Import sys to check command line arguments

# --- Attempt to set multiprocessing start method for CUDA ---
# This needs to execute when the main Celery process is starting.
# We will make this block run unconditionally when the script is executed,
# and then check if 'celery worker' is being run to print specific logs.
_is_celery_worker_invocation = 'celery' in sys.argv[0].lower() and 'worker' in sys.argv

try:
    import torch

    if torch.cuda.is_available():
        current_context = multiprocessing.get_start_method(allow_none=True)
        if _is_celery_worker_invocation:
            print(f"Celery Worker (Attempting Setup): Current multiprocessing start method: {current_context}")

        # Try to set to 'spawn' if it's not already set, or if it's 'fork'
        if current_context is None or current_context == 'fork':
            multiprocessing.set_start_method('spawn', force=True)
            if _is_celery_worker_invocation:
                print(
                    f"Celery Worker (Attempting Setup): Set multiprocessing start method to 'spawn'. New context: {multiprocessing.get_start_method(allow_none=True)}")
        elif _is_celery_worker_invocation:  # If already spawn or forkserver
            print(
                f"Celery Worker (Attempting Setup): Multiprocessing start method already '{current_context}'. Not changing.")
    else:
        if _is_celery_worker_invocation:
            print("Celery Worker (Attempting Setup): CUDA not available, not changing multiprocessing start method.")
except ImportError:
    if _is_celery_worker_invocation:
        print("Celery Worker (Attempting Setup): PyTorch not found, cannot set start method for CUDA.")
except RuntimeError as e:
    if _is_celery_worker_invocation:
        print(f"Celery Worker (Attempting Setup): ERROR setting multiprocessing start method: {e}")
        print(f"Celery Worker (Attempting Setup): Current method: {multiprocessing.get_start_method(allow_none=True)}")
# --- End of multiprocessing start method setting ---

from celery import Celery
import os
# import torch
from whisper_wrapper import transcribe_audio as actual_transcribe_function
from whisper_wrapper import load_whisper_model # For preloading
from celery.signals import worker_process_init

# Define default broker and backend URLs, allowing override via environment variables
CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

celery = Celery(
    'whisper_tasks', # Namespace for your tasks
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=['celery_worker_app'] # Module(s) where tasks are defined
)

# Optional Celery configuration
celery.conf.update(
    task_serializer='json',
    accept_content=['json'],  # Ignore other content
    result_serializer='json',
    timezone='UTC', # Or your preferred timezone
    enable_utc=True,
    worker_prefetch_multiplier=1, # Important for long-running tasks, especially with concurrency 1
    task_acks_late=True, # Acknowledge task only after it's completed (or failed)
)

@celery.task(name='transcribe_audio_task', bind=True) # bind=True gives access to self (the task instance)
def transcribe_audio_task(self, audio_path, model_name, task_type, language, initial_prompt, temperature, best_of, word_timestamps, verbose):
    """
    Celery task to transcribe audio.
    """
    print(f"Celery Task [{self.request.id}]: Starting transcription for {audio_path} with model {model_name}")
    try:
        result = actual_transcribe_function(
            audio_path=audio_path,
            model_name=model_name,
            task=task_type,
            language=language,
            initial_prompt=initial_prompt,
            temperature=temperature,
            best_of=best_of,
            word_timestamps=word_timestamps,
            verbose=verbose
        )

        # The task is responsible for cleaning up the temp file after processing.
        # We return the path so the status check can trigger deletion if needed,
        # or the task can delete it directly. For now, let's assume the task cleans it up.
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                print(f"Celery Task [{self.request.id}]: Temporary file {audio_path} deleted successfully.")
            except Exception as e_del:
                print(f"Celery Task [{self.request.id}]: Error deleting temporary file {audio_path}: {e_del}")

        if "error" in result:
            print(f"Celery Task [{self.request.id}]: Error during transcription: {result['error']}")
            # Optionally raise an exception to mark the task as FAILED more explicitly
            # raise ValueError(result['error'])
        else:
            print(f"Celery Task [{self.request.id}]: Transcription successful for {audio_path}")
        return result
    except Exception as e:
        print(f"Celery Task [{self.request.id}]: Exception during transcription for {audio_path}: {e}")
        # If audio_path still exists on unhandled exception, try to clean it up
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                print(f"Celery Task [{self.request.id}]: Temporary file {audio_path} deleted due to task exception.")
            except Exception as e_del:
                print(f"Celery Task [{self.request.id}]: Error deleting temporary file {audio_path} during task exception: {e_del}")
        raise # Re-raising the exception will mark the task as FAILED in Celery

@worker_process_init.connect
def preload_models(**kwargs):
    print("Celery Worker Process: Pre-loading default 'base' Whisper model...")
    try:
        # Let load_whisper_model decide the device (tries CUDA first if available)
        loaded_model = load_whisper_model(model_name="base")
        if loaded_model:
            print(f"Celery Worker Process: Model 'base' pre-loaded successfully on {loaded_model.device.type}.")
        else:
            print("Celery Worker Process: Failed to preload 'base' model.")
    except Exception as e:
        print(f"Celery Worker Process: Error pre-loading 'base' model: {e}")