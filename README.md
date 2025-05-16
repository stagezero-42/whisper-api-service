# Whisper API Service

This repository provides a Flask-based API service for OpenAI's Whisper model, allowing for easy audio transcription and translation. It uses Celery for asynchronous task processing to handle potentially long transcriptions efficiently and provides a user-friendly interface for testing and detailed documentation for API usage.

## Overview

This service wraps the OpenAI Whisper functionality into a web API. You can send audio files to the API, which then queues the transcription/translation task using Celery. The client receives a task ID and can poll for the result. This asynchronous architecture ensures the API remains responsive. The setup involves installing dependencies (including Celery and Redis), configuring the environment, and running the Flask application (via Gunicorn) and Celery workers as services.

## Features

* **Asynchronous Transcription/Translation:** Utilizes Celery and Redis for background processing of audio files.
* **Multiple Model Support:** Allows selection from various Whisper model sizes (tiny, base, small, medium, large, turbo).
* **GPU Acceleration:** Supports CUDA for faster transcription if a compatible NVIDIA GPU is available and correctly configured.
* **Variety of Output Formats:** Provides results in JSON, TXT, SRT, VTT, and TSV.
* **User Interface:** A simple web page (`index.html`) for easy testing of the API.
* **API Documentation:** A `docs.html` page explaining API parameters.
* **Service Management:** Instructions for running Gunicorn and Celery as systemd services on Ubuntu.

## Prerequisites

* Python 3.8+
* FFmpeg: Whisper depends on it for audio processing.
    ```bash
    sudo apt update && sudo apt install ffmpeg
    ```
* Redis: Used as the Celery message broker and result backend.
    ```bash
    sudo apt update && sudo apt install redis-server
    sudo systemctl start redis-server
    sudo systemctl enable redis-server
    ```
* Access to an NVIDIA GPU with CUDA toolkit installed is required for GPU-accelerated transcription. CPU-based transcription is also supported.

## Setting up in an IDE (e.g., PyCharm for Development)

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd whisper-api-service
    ```

2.  **Create and Activate a Virtual Environment:**
    It's highly recommended to use a virtual environment.
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
    (PyCharm can also manage this for you).

3.  **Install Dependencies:**
    The required packages are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Ensure Redis is Running:** For local development, ensure your Redis server is running.

5.  **Pre-download Whisper Models (Optional but Recommended):**
    The Whisper API service relies on OpenAI's Whisper models. These are downloaded automatically on first use and cached (typically in `~/.cache/whisper`). To avoid runtime downloads, you can pre-download them using the included utility script:
    ```bash
    python download_prompt_models.py
    ```
    Follow the on-screen prompts. This script needs to be run by the user who will execute the Celery workers.

6.  **Run the Application Stack (for Development):**
    You'll need to run three components in separate terminals:
    * **Terminal 1: Redis Server** (if not already running as a service)
    * **Terminal 2: Celery Worker**
        ```bash
        source venv/bin/activate
        # If using CUDA, the -P solo flag and spawn method are critical
        # The following command assumes celry_worker_app.py sets spawn method for CUDA
        celery -A celery_worker_app.celery worker -l INFO -P solo 
        ```
        *(Note: The `-P solo` flag is used here for CUDA compatibility by running tasks sequentially in the main worker process. The `celery_worker_app.py` file contains logic to set the Python multiprocessing start method to 'spawn' when CUDA is detected, which is essential.)*
    * **Terminal 3: Flask Development Server**
        ```bash
        source venv/bin/activate
        flask run --host=0.0.0.0 --port=5050 
        ```
        (Or run `app.py` directly from PyCharm).

7.  **Access the Application:**
    Open your browser to `http://127.0.0.1:5050` (or `http://0.0.0.0:5050`).

## Deploying to a Cloud Ubuntu Server (Production)

These instructions assume an Ubuntu server (e.g., Ubuntu 22.04).

1.  **Provision Server and Install Prerequisites:**
    * Ensure your Ubuntu server is up to date.
    * Install Python, pip, venv, FFmpeg, and Redis (as shown in the Prerequisites section).

2.  **Clone or Update the Repository:**
    ```bash
    # If cloning for the first time:
    # git clone <your-repository-url> /home/stt_api/whisper-api-service 
    # cd /home/stt_api/whisper-api-service
    
    # If updating an existing deployment:
    cd /home/stt_api/whisper-api-service # Adjust to your path
    git checkout main # Or your deployment branch
    git pull origin main
    ```

3.  **Set up Virtual Environment and Install/Update Dependencies:**
    ```bash
    cd /home/stt_api/whisper-api-service # Adjust to your path
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

4.  **Pre-download Whisper Models (Recommended):**
    Ensure the user that will run the Celery service (e.g., `stt_api`) runs this:
    ```bash
    source venv/bin/activate
    python download_prompt_models.py
    ```

5.  **Configure systemd Services:**
    We will create two systemd services: one for Gunicorn (to serve the Flask app) and one for the Celery worker.

    * **Gunicorn Service (`whisperapi-gunicorn.service`):**
        Create a file at `/etc/systemd/system/whisperapi-gunicorn.service`:
        ```bash
        sudo nano /etc/systemd/system/whisperapi-gunicorn.service
        ```
        Paste the following content (adjust `User`, `Group`, and paths if necessary):
        ```ini
        [Unit]
        Description=Whisper API Gunicorn Service
        After=network.target redis-server.service
        Requires=redis-server.service

        [Service]
        User=stt_api
        Group=stt_api
        WorkingDirectory=/home/stt_api/whisper-api-service
        ExecStart=/home/stt_api/whisper-api-service/venv/bin/gunicorn --workers 3 --bind 0.0.0.0:5050 app:app
        Restart=always
        RestartSec=10
        StandardOutput=journal
        StandardError=journal
        SyslogIdentifier=whisperapi-gunicorn

        [Install]
        WantedBy=multi-user.target
        ```

    * **Celery Worker Service (`whisperapi-celery.service`):**
        Create a file at `/etc/systemd/system/whisperapi-celery.service`:
        ```bash
        sudo nano /etc/systemd/system/whisperapi-celery.service
        ```
        Paste the following content (adjust `User`, `Group`, and paths if necessary):
        ```ini
        [Unit]
        Description=Whisper API Celery Worker Service
        After=network.target redis-server.service
        Requires=redis-server.service

        [Service]
        User=stt_api
        Group=stt_api
        WorkingDirectory=/home/stt_api/whisper-api-service
        # Setting PYTHON_MULTIPROCESSING_START_METHOD for CUDA compatibility with 'spawn'
        Environment="PYTHON_MULTIPROCESSING_START_METHOD=spawn"
        ExecStart=/home/stt_api/whisper-api-service/venv/bin/celery -A celery_worker_app.celery worker -l INFO -P solo
        Restart=always
        RestartSec=10
        StandardOutput=journal
        StandardError=journal
        SyslogIdentifier=whisperapi-celery

        [Install]
        WantedBy=multi-user.target
        ```
        *(Note: The `Environment="PYTHON_MULTIPROCESSING_START_METHOD=spawn"` and `-P solo` are critical for CUDA compatibility with Celery.)*

6.  **Reload systemd, Enable, and Start Services:**
    ```bash
    sudo systemctl daemon-reload
    sudo systemctl enable whisperapi-gunicorn.service
    sudo systemctl enable whisperapi-celery.service
    sudo systemctl start whisperapi-gunicorn.service
    sudo systemctl start whisperapi-celery.service
    ```

7.  **Check Service Status:**
    ```bash
    sudo systemctl status whisperapi-gunicorn.service
    sudo systemctl status whisperapi-celery.service
    ```
    Look for `Active: active (running)`.

8.  **View Logs (Troubleshooting):**
    If services fail or you need to check output:
    ```bash
    sudo journalctl -u whisperapi-gunicorn.service -f
    sudo journalctl -u whisperapi-celery.service -f
    ```

9.  **Configure Firewall (if necessary):**
    If you have a firewall (like `ufw`), ensure port 5050 (or your chosen port) is open:
    ```bash
    sudo ufw allow 5050/tcp
    ```

10. **Access the Application:**
    Navigate to `http://<your-server-ip>:5050` in your web browser.

## Code Structure

* `app.py`: Main Flask application, handles web requests, submits tasks to Celery.
* `celery_worker_app.py`: Defines the Celery application and transcription tasks. Includes logic to set multiprocessing start method to 'spawn' for CUDA compatibility.
* `whisper_wrapper.py`: Contains the core logic for loading Whisper models and performing transcription using the `openai-whisper` library.
* `requirements.txt`: Python dependencies.
* `templates/`: HTML templates for the web interface (`index.html`, `docs.html`).
* `static/`: Static files (e.g., `style.css`).
* `uploads/`: Directory for temporary audio file uploads (ensure it's writable by the Flask/Gunicorn user).
* `download_prompt_models.py`: Utility script to pre-download Whisper models.

## GPU and CUDA Considerations

* **Driver Installation:** Ensure you have the appropriate NVIDIA drivers installed on your server.
* **CUDA Toolkit:** PyTorch (a dependency of Whisper) needs a compatible CUDA toolkit version. The `requirements.txt` may specify a CUDA-enabled PyTorch build.
* **`celery_worker_app.py`:** Contains logic at the top to set `multiprocessing.set_start_method('spawn', force=True)` when CUDA is detected. This is crucial for preventing errors when Celery workers (child processes) try to initialize CUDA.
* **Celery Worker Command:** The `-P solo` flag for the Celery worker is used in the systemd service file to ensure tasks run in the main worker process, which has its multiprocessing context correctly set for CUDA. The `Environment="PYTHON_MULTIPROCESSING_START_METHOD=spawn"` in the service file provides an additional safeguard.

## Troubleshooting

* **Check Service Logs:** Use `sudo journalctl -u <service-name>.service -f` for detailed error messages.
* **Celery Worker Logs:** Pay close attention to messages related to model loading, CUDA initialization, and task execution.
* **Redis Connection:** Ensure Celery workers and the Flask app can connect to Redis.
* **File Permissions:** The `uploads/` directory must be writable by the user running Gunicorn/Flask. The user running Celery workers needs read access to these files.
* **CUDA Issues:** The "Cannot re-initialize CUDA in forked subprocess" error is common. The combination of setting the `spawn` start method and using the `solo` pool for Celery workers is the primary mitigation for this.
