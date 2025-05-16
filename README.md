# Whisper API Service

This repository provides a Flask-based API service for OpenAI's Whisper model, allowing for easy audio transcription and translation. It offers a user-friendly interface for testing and detailed documentation for API usage.

## Overview

This service wraps the OpenAI Whisper functionality into a web API. You can send audio files to the API and receive transcriptions or translations in various formats. The setup involves installing dependencies, configuring the environment, and running the Flask application. The API is designed for both local development and production deployment.

## Setting up in an IDE (e.g., PyCharm)

Follow these steps to set up the Whisper API service in an IDE like PyCharm:

1.  **Clone the Repository:**

    ```bash
    git clone <repository-url>
    cd whisper-api-service
    ```

2.  **Create a Virtual Environment:**
    It's highly recommended to use a virtual environment.

      * In PyCharm, you can create one via `File > Settings > Project: whisper-api-service > Python Interpreter > Add Interpreter > Add Local Interpreter`. Select `Virtualenv Environment` or `Conda Environment`.
      * Alternatively, use the terminal:
        ```bash
        python -m venv venv
        source venv/bin/activate  # On Windows: venv\Scripts\activate
        ```

3.  **Install Dependencies:**
    The required packages are listed in `requirements.txt`.

    ```bash
    pip install -r requirements.txt
    ```

    Ensure you have FFmpeg installed on your system as Whisper depends on it for audio processing.

4.  **Configure PyCharm Run Configuration:**

      * Open `app.py`. [cite: 1]
      * Click on the green play button next to the `if __name__ == '__main__':` line or create a new run configuration: `Run > Edit Configurations... > Add New Configuration > Python`.
      * **Script path:** Set to the path of `app.py`.
      * **Working directory:** Set to the project root directory (`whisper-api-service`).
      * Ensure your project's Python interpreter is the virtual environment created in step 2.

5.  **Run the Application:**

      * Click the Run button in PyCharm.
      * The application will start, typically on `http://0.0.0.0:5050` or `http://127.0.0.1:5050`. [cite: 1] You can access the service through your browser or API clients like Postman.

## Deploying to a Cloud Ubuntu 22.04 Server (Production)

To deploy the Whisper API service to an Ubuntu 22.04 server for production, follow these general steps:

1.  **Provision an Ubuntu 22.04 Server:**

      * Use any cloud provider (e.g., AWS, Google Cloud, Azure, DigitalOcean). Ensure the server has sufficient resources (CPU, RAM, and optionally GPU).

2.  **Install System Dependencies:**

      * Connect to your server via SSH.
      * Update package lists:
        ```bash
        sudo apt update
        sudo apt upgrade -y
        ```
      * Install Python, pip, and venv:
        ```bash
        sudo apt install python3 python3-pip python3-venv -y
        ```
      * Install FFmpeg:
        ```bash
        sudo apt install ffmpeg -y
        ```

3.  **Clone the Repository:**

    ```bash
    git clone <repository-url>
    cd whisper-api-service
    ```

4.  **Set up a Virtual Environment and Install Dependencies:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

5.  **Pre-download Whisper Models (Optional but Recommended):**
   The Whisper API service relies on OpenAI's Whisper models for transcription. These models are downloaded automatically by the whisper library on their first use and cached locally (typically in ~/.cache/whisper). To avoid runtime downloads, especially in production environments or for faster initial startup of Celery workers, you can pre-download the desired models. This project includes a utility script, pull_models.py, which allows you to interactively select and download standard Whisper models (including tiny, base, small, medium, large, and turbo, along with their English-only counterparts). To use it, activate your virtual environment and run `python download_prompt_models.py`, then follow the on-screen prompts for each model. This ensures the models are present in the local cache before the application or workers attempt to load them.

    ```bash
    python pull_models.py
    ```



6.  **Install a Production WSGI Server (e.g., Gunicorn):**
    Flask's built-in server is not suitable for production.

    ```bash
    pip install gunicorn
    ```

7.  **Configure Gunicorn:**

      * Test Gunicorn by running it from your project directory:
        ```bash
        gunicorn --bind 0.0.0.0:5050 app:app
        ```
        Replace `app:app` with `your_main_file_name:your_flask_app_instance_name` if different.

8.  **Set up a Systemd Service (for managing the application):**

      * Create a service file:
        ```bash
        sudo nano /etc/systemd/system/whisperapi.service
        ```
      * Add the following content (adjust paths and user as necessary):
        ```ini
        [Unit]
        Description=Gunicorn instance to serve Whisper API
        After=network.target

        [Service]
        User=your_username # Replace with the user running the application
        Group=www-data # Or your_username
        WorkingDirectory=/path/to/whisper-api-service # Replace with actual path
        Environment="PATH=/path/to/whisper-api-service/venv/bin" # Replace with actual path
        ExecStart=/path/to/whisper-api-service/venv/bin/gunicorn --workers 3 --bind unix:whisperapi.sock -m 007 app:app # Adjust workers as needed

        [Install]
        WantedBy=multi-user.target
        ```
      * Reload systemd, start, and enable the service:
        ```bash
        sudo systemctl daemon-reload
        sudo systemctl start whisperapi
        sudo systemctl enable whisperapi
        sudo systemctl status whisperapi
        ```


### Turbo Mode and GPU

**Note:** If your production server has a compatible NVIDIA GPU and the necessary drivers (CUDA toolkit) installed, Whisper can utilize it for significantly faster transcription speeds. This is often referred to as "turbo mode" or GPU-accelerated processing. [cite: 1] The `whisper_wrapper.py` script attempts to use CUDA if available. Ensure your Python environment (especially PyTorch, a dependency of Whisper) is installed with GPU support. You might need to install a specific PyTorch version:

```bash
# Example: Uninstall CPU-only PyTorch and install GPU version
pip uninstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # Or your CUDA version
```

Refer to the [PyTorch installation guide](https://pytorch.org/get-started/locally/) for the correct command based on your CUDA version.
The `app.py` also includes logic to check for GPU availability which can be surfaced in the UI. 