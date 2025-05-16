from flask import Flask, request, jsonify, render_template, url_for
import os
import tempfile
import torch  # To check for GPU
from whisper_wrapper import transcribe_audio, load_whisper_model, to_srt, to_vtt, to_tsv
from celery_worker_app import transcribe_audio_task

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'ogg', 'flac', 'aac', 'opus'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 300 * 1024 * 1024  # 300 MB

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    gpu_available = torch.cuda.is_available()
    return render_template('index.html', gpu_available=gpu_available)


@app.route('/docs')
def docs():
    return render_template('docs.html')


@app.route('/transcribe', methods=['POST'])
def transcribe_route(): # Kept original name
    if 'audio_file' not in request.files:
        return jsonify({"error": "No audio_file part in the request"}), 400

    file = request.files['audio_file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        _, temp_ext = os.path.splitext(file.filename)
        # The temporary file is saved. Celery task will be responsible for deleting it.
        temp_file_handler = tempfile.NamedTemporaryFile(delete=False, dir=app.config['UPLOAD_FOLDER'], suffix=temp_ext)
        file.save(temp_file_handler.name)
        temp_file_path = temp_file_handler.name
        temp_file_handler.close()

        try:
            model_name = request.form.get('model_name', 'base')
            task_type = request.form.get('task', 'transcribe') # Parameter for Celery task
            language = request.form.get('language')
            if language == "": language = None

            initial_prompt = request.form.get('initial_prompt')
            if initial_prompt == "": initial_prompt = None

            word_timestamps = request.form.get('word_timestamps', 'false').lower() in ['true', 'on', '1']

            verbose_form = request.form.get('verbose_output', 'default')
            verbose_param = None
            if verbose_form == 'true': verbose_param = True
            elif verbose_form == 'false': verbose_param = False

            temperature_str = request.form.get('temperature', '0.0')
            best_of_str = request.form.get('best_of', '5')
            temperature = float(temperature_str) if temperature_str else 0.0
            best_of = int(best_of_str) if best_of_str else 5

            print(f"API Request (to Celery): model='{model_name}', task='{task_type}', lang='{language}' for file {temp_file_path}")

            # Dispatch the task to Celery
            task_run = transcribe_audio_task.delay(
                audio_path=temp_file_path,
                model_name=model_name,
                task_type=task_type,
                language=language,
                initial_prompt=initial_prompt,
                temperature=temperature,
                best_of=best_of,
                word_timestamps=word_timestamps,
                verbose=verbose_param
            )

            return jsonify({
                "message": "Transcription task submitted successfully.",
                "task_id": task_run.id,
                "status_url": url_for('get_task_status', task_id=task_run.id, _external=True),
                "ui_status_url": url_for('index', task_id=task_run.id, _external=False)
            }), 202

        except Exception as e:
            app.logger.error(f"Error submitting task to Celery: {e}")
            # Clean up the temp file if task submission failed
            if os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception as e_del:
                    app.logger.error(f"Error deleting orphaned temp file {temp_file_path}: {e_del}")
            return jsonify({"error": f"Failed to submit task: {str(e)}"}), 500
    else:
        return jsonify({"error": "File type not allowed"}), 400

@app.route('/status/<task_id>', methods=['GET'])
def get_task_status(task_id):
    task = transcribe_audio_task.AsyncResult(task_id)
    response_data = {
        "task_id": task_id,
        "status": task.status,
        "result": None,
        "error_info": None
    }

    if task.successful():
        raw_result = task.result
        # Apply formatting based on original request parameters (could be passed or re-queried)
        output_format_preference = request.args.get('output_format', 'json')
        simplified_output_mode = request.args.get('simplified_output', 'false').lower() in ['true', 'on', '1']

        # Re-use your formatting logic here (simplified example)
        formatted_output = None
        if raw_result and ("text" in raw_result or "segments" in raw_result):
            if output_format_preference == 'txt':
                formatted_output = raw_result.get("text", "")
            elif output_format_preference == 'srt' and "segments" in raw_result:
                formatted_output = to_srt(raw_result["segments"], as_dict=simplified_output_mode)
            elif output_format_preference == 'vtt' and "segments" in raw_result:
                formatted_output = to_vtt(raw_result["segments"], as_dict=simplified_output_mode)
            elif output_format_preference == 'tsv' and "segments" in raw_result:
                formatted_output = to_tsv(raw_result["segments"])
            elif output_format_preference == 'json':
                if simplified_output_mode:
                    formatted_output = raw_result.get("text", "")
                else: # Full JSON for non-simplified
                    response_data["result"] = {"transcription_details": raw_result} # Set directly
                    return jsonify(response_data) # Early return for this specific case
            else:
                formatted_output = raw_result # Fallback to raw result

        if simplified_output_mode and output_format_preference != 'json': # JSON simplified is handled
             response_data["result"] = {"formatted_output": formatted_output}
        elif not simplified_output_mode and output_format_preference != 'json': # Full output
             response_data["result"] = {"transcription_details": raw_result, "formatted_output": formatted_output}
        elif simplified_output_mode and output_format_preference == 'json':
             response_data["result"] = {"formatted_output": formatted_output}


    elif task.failed():
        response_data["error_info"] = str(task.info) # .info contains the exception
        # The Celery task should have already tried to delete the temp file on failure

    return jsonify(response_data)


if __name__ == '__main__':
    # print("Pre-loading default 'base' Whisper model...")
    # load_whisper_model(model_name="base") moved to celery_worker_app
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5050)