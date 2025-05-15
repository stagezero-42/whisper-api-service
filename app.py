from flask import Flask, request, jsonify, render_template
import os
import tempfile
import torch  # To check for GPU
from whisper_wrapper import transcribe_audio, load_whisper_model, to_srt, to_vtt, to_tsv

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
def transcribe_route():
    if 'audio_file' not in request.files:
        return jsonify({"error": "No audio_file part in the request"}), 400

    file = request.files['audio_file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        _, temp_ext = os.path.splitext(file.filename)
        temp_file = tempfile.NamedTemporaryFile(delete=False, dir=app.config['UPLOAD_FOLDER'], suffix=temp_ext)
        file.save(temp_file.name)
        temp_file_path = temp_file.name
        temp_file.close()

        try:
            model_name = request.form.get('model_name', 'base')
            task = request.form.get('task', 'transcribe')
            language = request.form.get('language')
            if language == "": language = None

            initial_prompt = request.form.get('initial_prompt')
            if initial_prompt == "": initial_prompt = None

            word_timestamps = request.form.get('word_timestamps', 'false').lower() in ['true', 'on', '1']

            verbose_form = request.form.get('verbose_output', 'default')
            verbose_param = None
            if verbose_form == 'true':
                verbose_param = True
            elif verbose_form == 'false':
                verbose_param = False

            temperature_str = request.form.get('temperature', '0.0')
            best_of_str = request.form.get('best_of', '5')

            try:
                temperature = float(temperature_str) if temperature_str else 0.0
            except ValueError:
                return jsonify({"error": f"Invalid temperature value: {temperature_str}"}), 400

            try:
                best_of = int(best_of_str) if best_of_str else 5
            except ValueError:
                return jsonify({"error": f"Invalid best_of value: {best_of_str}"}), 400

            output_format_preference = request.form.get('output_format', 'json')
            simplified_output_mode = request.form.get('simplified_output', 'false').lower() in ['true', 'on', '1']

            # Handle "turbo" model selection logic if needed (e.g., check enable_turbo checkbox if sent)
            # enable_turbo_checkbox = request.form.get('enable_turbo', 'false').lower() in ['true', 'on', '1']
            # if model_name == "turbo" and not enable_turbo_checkbox and not torch.cuda.is_available():
            #     # This logic is mostly UI, but backend can double check if strict
            #     print("Warning: 'turbo' model selected without 'Enable Turbo' checked or GPU not confirmed by UI flag.")

            print(f"API Request: model='{model_name}', task='{task}', lang='{language}', "
                  f"prompt='{initial_prompt is not None}', temp={temperature}, best_of={best_of}, "
                  f"word_ts={word_timestamps}, verbose_param={verbose_param}, "
                  f"output_pref='{output_format_preference}', simplified={simplified_output_mode}")

            raw_result = transcribe_audio(
                audio_path=temp_file_path, model_name=model_name, task=task, language=language,
                initial_prompt=initial_prompt, temperature=temperature, best_of=best_of,
                word_timestamps=word_timestamps, verbose=verbose_param
            )

            if "error" in raw_result:
                return jsonify(raw_result), 500

            # Construct response based on simplified_output_mode
            response_data = {}
            if simplified_output_mode:
                if not raw_result or ("text" not in raw_result and "segments" not in raw_result):
                    return jsonify({"error": "Transcription produced no usable content."}), 500

                if output_format_preference == 'txt':
                    response_data = {"formatted_output": raw_result.get("text", "")}
                elif output_format_preference == 'srt' and "segments" in raw_result:
                    response_data = {"formatted_output": to_srt(raw_result["segments"], as_dict=True)}
                elif output_format_preference == 'vtt' and "segments" in raw_result:
                    response_data = {"formatted_output": to_vtt(raw_result["segments"], as_dict=True)}
                elif output_format_preference == 'tsv' and "segments" in raw_result:
                    response_data = {"formatted_output": to_tsv(raw_result["segments"])}
                elif output_format_preference == 'json':  # Simplified JSON is just the text
                    response_data = {"formatted_output": raw_result.get("text", "")}
                else:  # Fallback for simplified if segments missing but text exists
                    response_data = {"formatted_output": raw_result.get("text",
                                                                        "No segments found for chosen format, but text is available.")}

            else:  # Full output mode
                response_data = {"transcription_details": raw_result}
                if raw_result and ("segments" in raw_result or "text" in raw_result):
                    if output_format_preference == 'txt':
                        response_data["formatted_output"] = raw_result.get("text", "")
                    elif output_format_preference == 'srt' and "segments" in raw_result:
                        response_data["formatted_output"] = to_srt(raw_result["segments"],
                                                                   as_dict=False)  # String output
                    elif output_format_preference == 'vtt' and "segments" in raw_result:
                        response_data["formatted_output"] = to_vtt(raw_result["segments"],
                                                                   as_dict=False)  # String output
                    elif output_format_preference == 'tsv' and "segments" in raw_result:
                        response_data["formatted_output"] = to_tsv(raw_result["segments"])
                    # If 'json', transcription_details is already the main content. No separate formatted_output needed.
                elif output_format_preference == 'txt':  # Handle case where segments might be empty but text exists
                    response_data["formatted_output"] = raw_result.get("text", "")

            return jsonify(response_data)

        finally:
            try:
                os.remove(temp_file_path)
            except Exception as e:
                app.logger.error(f"Error deleting temporary file {temp_file_path}: {e}")

    else:
        return jsonify({"error": "File type not allowed"}), 400


if __name__ == '__main__':
    print("Pre-loading default 'base' Whisper model...")
    load_whisper_model(model_name="base")
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5050)