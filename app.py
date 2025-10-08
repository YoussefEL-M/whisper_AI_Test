from flask import Flask, request, jsonify, render_template, abort, Response
from tempfile import NamedTemporaryFile
from whisper.utils import get_writer
import whisper
import ffmpeg
import srt
import datetime
import os
import torch
import gc
from transformers import LlamaForCausalLM, LlamaTokenizer
import pathlib
import argostranslate.package
import argostranslate.translate
from pydub import AudioSegment
import io
from flask_socketio import SocketIO, emit
import asyncio
import websockets
import json
import numpy as np
import threading
from contextlib import asynccontextmanager

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # Commented out due to compatibility issue

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_name_or_path = "/app/TinyLlama-1.1B-Chat-v1.0"

# Global variables for models (lazy loading)
whisper_model = None
llama_model = None
tokenizer = None


def load_whisper_model():
    """Load Whisper model on demand"""
    global whisper_model
    if whisper_model is None:
        try:
            print("Loading Whisper model...")
            whisper_model = whisper.load_model("turbo", device=DEVICE)
            print("Whisper turbo model loaded successfully")
        except torch.cuda.OutOfMemoryError:
            print("Turbo model too large, falling back to base model")
            try:
                whisper_model = whisper.load_model("base", device=DEVICE)
                print("Whisper base model loaded successfully")
            except Exception as e:
                print(f"Failed to load base model: {e}")
                print("Falling back to small model...")
                whisper_model = whisper.load_model("small", device=DEVICE)
                print("Whisper small model loaded successfully")
        except Exception as e:
            print(f"Failed to load Whisper model: {e}")
            import traceback
            traceback.print_exc()
            return None
    return whisper_model


def unload_whisper_model():
    """Unload Whisper model from memory"""
    global whisper_model
    if whisper_model is not None:
        del whisper_model
        whisper_model = None
        torch.cuda.empty_cache()
        gc.collect()
        print("Whisper model unloaded")


def load_llama_model():
    """Load TinyLlama model on demand with error handling"""
    global llama_model, tokenizer
    if llama_model is None:
        try:
            print("Loading TinyLlama model...")
            tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
            
            # Check if CUDA is available and working
            if torch.cuda.is_available() and DEVICE == "cuda":
                try:
                    print("Attempting to load TinyLlama on GPU with float16...")
                    llama_model = LlamaForCausalLM.from_pretrained(
                        model_name_or_path,
                        torch_dtype=torch.float16,
                        device_map="auto"
                    )
                    print("TinyLlama loaded on GPU with float16")
                except Exception as e:
                    print(f"Failed to load on GPU with float16: {e}")
                    print("Trying GPU with float32...")
                    try:
                        llama_model = LlamaForCausalLM.from_pretrained(
                            model_name_or_path,
                            torch_dtype=torch.float32,
                            device_map="auto"
                        )
                        print("TinyLlama loaded on GPU with float32")
                    except Exception as e2:
                        print(f"Failed to load on GPU with float32: {e2}")
                        print("Falling back to CPU...")
                        llama_model = LlamaForCausalLM.from_pretrained(
                            model_name_or_path,
                            torch_dtype=torch.float32,
                            device_map="cpu"
                        )
                        print("TinyLlama loaded on CPU")
            else:
                print("CUDA not available, loading on CPU...")
                llama_model = LlamaForCausalLM.from_pretrained(
                    model_name_or_path,
                    torch_dtype=torch.float32,
                    device_map="cpu"
                )
                print("TinyLlama loaded on CPU")
        except Exception as e:
            print(f"Failed to load TinyLlama model: {e}")
            import traceback
            traceback.print_exc()
            return None, None
            
    return llama_model, tokenizer


def unload_llama_model():
    """Unload TinyLlama model from memory"""
    global llama_model, tokenizer
    if llama_model is not None:
        del llama_model
        llama_model = None
    if tokenizer is not None:
        del tokenizer
        tokenizer = None
    torch.cuda.empty_cache()
    gc.collect()
    print("TinyLlama model unloaded")
# ----------- CHECK TRANLATION INSTALLATION -----------

def ensure_model_installed(from_code: str, to_code: str):
    installed_langs = argostranslate.translate.get_installed_languages()
    from_lang = next((l for l in installed_langs if l.code == from_code), None)
    to_lang = next((l for l in installed_langs if l.code == to_code), None)

    if from_lang and to_lang:
        try:
            from_lang.get_translation(to_lang)
            print(f"Translation {from_code} → {to_code} is already installed")
            return
        except Exception:
            pass

    print(f"Downloading translation model from {from_code} to {to_code}...")
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()

    package_to_install = next(
        (p for p in available_packages if p.from_code == from_code and p.to_code == to_code), None)

    if package_to_install:
        path = package_to_install.download()
        argostranslate.package.install_from_path(path)
        print(f"Installed {from_code} → {to_code} model")
    else:
        print(f"No model available for {from_code} → {to_code}")

# Ensure Danish-English pair both directions
ensure_model_installed("da", "en")
ensure_model_installed("en", "da")


# Directory for your local models:
models_dir = os.path.join(os.path.dirname(__file__), "installed_models")

# Install local .argosmodel files
for filename in os.listdir(models_dir):
    if filename.endswith(".argosmodel"):
        model_path = os.path.join(models_dir, filename)
        argostranslate.package.install_from_path(model_path)

# Install local .argosmodel files
for filename in os.listdir(models_dir):
    if filename.endswith(".argosmodel"):
        model_path = os.path.join(models_dir, filename)
        argostranslate.package.install_from_path(model_path)

# Auto-download and install Danish->English model if missing
def ensure_model_installed(from_code, to_code):
    installed_langs = argostranslate.translate.get_installed_languages()
    from_lang = next((l for l in installed_langs if l.code == from_code), None)
    to_lang = next((l for l in installed_langs if l.code == to_code), None)
    if from_lang and to_lang:
        # Check if translation exists already
        try:
            from_lang.get_translation(to_lang)
            print(f"Translation {from_code} → {to_code} already installed")
            return
        except Exception:
            pass

    # Update package index and find package
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()
    package_to_install = next(
        (p for p in available_packages if p.from_code == from_code and p.to_code == to_code),
        None,
    )
    if package_to_install:
        print(f"Downloading and installing {from_code} → {to_code} model")
        pkg_path = package_to_install.download()
        argostranslate.package.install_from_path(pkg_path)
    else:
        print(f"No package found for {from_code} → {to_code}")

# Call your ensure function for Danish to English
ensure_model_installed("da", "en")
ensure_model_installed("en", "da")

installed_languages = argostranslate.translate.get_installed_languages()

for from_lang in installed_languages:
    for to_lang in installed_languages:
        if from_lang != to_lang:
            try:
                from_lang.get_translation(to_lang)
                print(f"{from_lang.code} → {to_lang.code}")
            except Exception:
                pass  # No translation available



# ----------- FIXED SUMMARIZATION LOGIC -----------

def chunk_text(text: str, max_chars: int = 1500):
    """Yield chunks of text up to max_chars for summarization"""
    for i in range(0, len(text), max_chars):
        yield text[i:i + max_chars]


def run_llama_summary(prompt: str) -> str:
    """Run TinyLlama on a given prompt and return its decoded output"""
    global llama_model, tokenizer
    try:
        if not prompt.strip():
            return ""
            
        model, tok = load_llama_model()
        if model is None or tok is None:
            print("[Summarizer] Model or tokenizer not loaded")
            return ""
            
        inputs = tok(prompt, return_tensors="pt", max_length=768, truncation=True)

        if next(model.parameters()).is_cuda:
            inputs = inputs.to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
                repetition_penalty=1.05
            )

        decoded = tok.decode(outputs[0], skip_special_tokens=True).strip()

        # Strip away prompt echo if present
        if prompt in decoded:
            decoded = decoded.split(prompt, 1)[-1].strip()

        return decoded
    except Exception as e:
        print(f"[Summarizer] Llama failed: {e}")
        import traceback
        traceback.print_exc()
        return ""


def summarize_transcript(full_text: str) -> str:
    """Create a summary of the transcript using TinyLlama (with chunking + fallback)."""
    try:
        if not full_text.strip():
            return "No text content to summarize."
            
        # Unload Whisper to save VRAM
        unload_whisper_model()

        partial_summaries = []
        for chunk in chunk_text(full_text, max_chars=1500):
            if not chunk.strip():
                continue
                
            prompt = (
                "Provide a clear and concise professional summary of the following meeting transcript. Focus on the main discussion points, key decisions, and any action items. The summary should be brief but complete — long enough to capture essential details without unnecessary repetition or filler.\n\n"
                f"{chunk.strip()}\n\nSummary:"
            )
            summary_piece = run_llama_summary(prompt)
            if summary_piece and summary_piece.strip():
                partial_summaries.append(summary_piece)

        if not partial_summaries:
            return "Unable to generate summary. The text may be too short or the model may be unavailable."

        # If transcript was long → combine partial summaries
        if len(partial_summaries) > 1:
            combined_text = " ".join(partial_summaries)
            final_prompt = (
                "Combine the following partial summaries into one clear coherent summary "
                "of about 8–12 sentences:\n\n"
                f"{combined_text}\n\nFinal Summary:"
            )
            final_summary = run_llama_summary(final_prompt)
            return final_summary.strip() if final_summary else combined_text
        else:
            return partial_summaries[0].strip()

    except Exception as e:
        print(f"[Summarizer] Error: {e}")
        # Simple fallback – first few sentences of transcript
        sentences = full_text.split('. ')[:5]
        return '. '.join(sentences) + '.'


# -------------------------------------------------

if torch.cuda.is_available():
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("CUDA not available, running on CPU")

# Video to audio helper functions
def is_audio(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_AUDIO

def is_video(filename):
    ext = os.path.splitext(filename).lower()
    return ext in ALLOWED_VIDEO

def video_to_audio(video_path, audio_path):
    ffmpeg.input(video_path).output(audio_path).run()

ALLOWED_AUDIO = {'.mp3', '.wav', '.flac', '.m4a', '.ogg', '.webm'}
ALLOWED_VIDEO = {'.mp4', '.mov', '.avi', '.webm', '.mkv'}

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins=["https://rosetta.semaphor.dk","https://meet2.semaphor.dk"])

app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB max request size
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Add request timeout handling
@app.before_request
def limit_remote_addr():
    # Log incoming requests for debugging
    print(f"Incoming request: {request.method} {request.path} - Content-Length: {request.content_length}")

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Please reduce file size and try again."}), 413

@app.errorhandler(Exception)
def handle_exception(e):
    print(f"Unhandled exception: {e}")
    import traceback
    traceback.print_exc()
    return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@socketio.on('audio_data', namespace='/live-cc')
def handle_audio(data):
    """
    Receives raw Opus audio chunks from Jitsi (via SkyNet),
    sends them to Whisper for streaming transcription,
    and emits partial captions back to the client.
    """
    global whisper_model
    if whisper_model is None:
        whisper_model = load_whisper_model()

    # data is binary Opus; convert to WAV in-memory
    audio = AudioSegment.from_file(io.BytesIO(data), format="opus")
    buf = io.BytesIO()
    audio.export(buf, format="wav", parameters=["-ar", "16000"])
    buf.seek(0)

    # Transcribe this chunk with Whisper (streaming or segment-level)
    result = whisper_model.transcribe(buf, **{})  # adjust kwargs if needed
    text = result["text"].strip()

    # Send caption back immediately
    emit('caption', {'text': text})

@app.route("/")
def index():
    return render_template("indexY.html")

@app.route("/status")
def status():
    """Simple status endpoint that doesn't require models to be loaded"""
    return jsonify({
        "status": "running",
        "message": "Flask app is running",
        "timestamp": int(datetime.datetime.now().timestamp() * 1000)
    })

@app.route("/progress")
def progress():
    """Progress endpoint to check if models are loading"""
    return jsonify({
        "whisper_loading": whisper_model is None,
        "llama_loading": llama_model is None,
        "message": "Models are loading..." if whisper_model is None else "Models ready"
    })

@app.route("/transcribe", methods=["POST"])
def transcribe():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        audio_file = request.files["file"]
        if audio_file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Add file size check (1GB limit)
        MAX_FILE_SIZE = 1024 * 1024 * 1024  # 1GB in bytes
        audio_file.seek(0, 2)  # Seek to end
        file_size = audio_file.tell()
        audio_file.seek(0)  # Reset to beginning
        
        if file_size > MAX_FILE_SIZE:
            return jsonify({"error": f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"}), 413
            
        print(f"Processing file: {audio_file.filename}, Size: {file_size / (1024*1024):.2f}MB")
        
        language = request.form.get("language") or None

        with NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.filename)[1]) as tmp_input:
            print(f"Saving to temporary file: {tmp_input.name}")
            audio_file.save(tmp_input.name)
            tmp_input_path = tmp_input.name

        with NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            tmp_wav_path = tmp_wav.name

        try:
            print("Converting audio to WAV format...")
            # Convert audio to WAV format
            ffmpeg.input(tmp_input_path).output(tmp_wav_path, format="wav", ar="16k").run(quiet=True, overwrite_output=True)
            
            print("Unloading Llama model to free memory...")
            # Unload Llama model to free memory
            unload_llama_model()
            
            print("Loading Whisper model...")
            # Load Whisper model
            model = load_whisper_model()
            
            if model is None:
                return jsonify({"error": "Failed to load Whisper model"}), 500
            
            # Prepare transcription parameters
            transcribe_kwargs = {}
            if language:
                transcribe_kwargs['language'] = language
            
            print("Starting transcription...")
            print(f"File size: {file_size / (1024*1024):.2f}MB - This may take several minutes...")

            # Check available memory before transcription
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"GPU memory before transcription: {allocated:.2f}GB / {total:.2f}GB")
                
                if allocated > (total * 0.8):  # If using more than 80% of GPU memory
                    torch.cuda.empty_cache()
                    gc.collect()
                    print("Cleared GPU cache due to high memory usage")
            
            # Perform transcription with error handling
            # Perform transcription with error handling
            try:
                print(f"Starting Whisper transcription on file size: {os.path.getsize(tmp_wav_path) / (1024*1024):.2f}MB")
                
                # Add progress callback for very large files
                def progress_callback(progress):
                    print(f"Transcription progress: {progress:.1%}")
                
                result = model.transcribe(
                    tmp_wav_path, 
                    verbose=True,  # Enable verbose logging
                    **transcribe_kwargs
                )
                
            except torch.cuda.OutOfMemoryError:
                print("GPU out of memory, clearing cache and retrying...")
                torch.cuda.empty_cache()
                gc.collect()
                # Try again with smaller model or CPU
                unload_whisper_model()
                print("Loading smaller Whisper model on CPU...")
                model = whisper.load_model("base", device="cpu")
                result = model.transcribe(tmp_wav_path, verbose=True, **transcribe_kwargs)
            except Exception as transcribe_error:
                print(f"Transcription failed: {transcribe_error}")
                import traceback
                traceback.print_exc()
                raise transcribe_error

            print("Transcription completed successfully")
            return jsonify({
                "transcript": result["text"],
                "language": result.get("language", ""),
                "segments": result.get("segments", [])
            })

        except Exception as e:
            print(f"Transcription error: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({"error": f"Transcription failed: {str(e)}"}), 500

        finally:
            # Clean up temporary files
            try:
                if os.path.exists(tmp_input_path):
                    os.remove(tmp_input_path)
                if os.path.exists(tmp_wav_path):
                    os.remove(tmp_wav_path)
            except Exception as cleanup_error:
                print(f"Cleanup error: {cleanup_error}")

    except Exception as e:
        print(f"Transcribe endpoint error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route("/summarize_from_json", methods=["POST"])
def summarize_from_json():
    try:
        json_data = request.get_json()
        if not json_data or 'transcript' not in json_data:
            abort(400, description="No transcript data provided")

        full_transcript = json_data['transcript']
        summary = summarize_transcript(full_transcript)

        installed_languages = argostranslate.translate.get_installed_languages()
        from_lang = next((l for l in installed_languages if l.code == "en"), None)
        to_lang = next((l for l in installed_languages if l.code == "da"), None)

        if from_lang and to_lang:
            try:
                translation = from_lang.get_translation(to_lang)
                if translation:
                    danish_summary = translation.translate(summary)
                else:
                    print("Translation not available, returning English summary")
                    danish_summary = summary
            except Exception as e:
                print(f"Translation error: {e}, returning English summary")
                danish_summary = summary
        else:
            print("Language pair not available, returning English summary")
            danish_summary = summary

        return Response(
            danish_summary,
            mimetype="text/plain",
            headers={
                "Content-Disposition": 'attachment; filename="summary.txt"'
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/paraphrase_srt", methods=["POST"])
def paraphrase_srt():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        srt_file = request.files["file"]
        if srt_file.filename == '':
            return jsonify({"error": "No file selected"}), 400
            
        with NamedTemporaryFile(delete=False, suffix=".srt") as tmp_file:
            srt_file.save(tmp_file.name)
            tmp_path = tmp_file.name

        try:
            with open(tmp_path, "r", encoding="utf-8") as f:
                srt_content = f.read()

            subtitles = list(srt.parse(srt_content))
            full_text = " ".join([sub.content for sub in subtitles])
            
            if not full_text.strip():
                return jsonify({"error": "No text content found in SRT file"}), 400
                
            summary = summarize_transcript(full_text)

            return Response(
                summary,
                mimetype="text/plain",
                headers={
                    "Content-Disposition": 'attachment; filename="summary.txt"'
                }
            )
        except Exception as e:
            print(f"Paraphrase error: {e}")
            return jsonify({"error": f"Paraphrase failed: {str(e)}"}), 500
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
    except Exception as e:
        print(f"Paraphrase endpoint error: {e}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500


@app.route("/transcribe/srt", methods=["POST"])
def transcribe_srt():
    if "file" not in request.files:
        abort(400, description="No file Uploaded")
    video_file = request.files["file"]
    language = request.form.get("language") or None

    with NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.filename)[1]) as tmp_video:
        video_file.save(tmp_video.name)
        tmp_video_path = tmp_video.name
    with NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
        tmp_audio_path = tmp_audio.name

    try:
        ffmpeg.input(tmp_video_path).output(tmp_audio_path, format="wav", ar="16k").run(quiet=True, overwrite_output=True)
        unload_llama_model()
        model = load_whisper_model()
        transcribe_kwargs = {}
        if language:
            transcribe_kwargs["language"] = language
        result = model.transcribe(tmp_audio_path, **transcribe_kwargs)

        subs = []
        for i, seg in enumerate(result.get("segments", []), start=1):
            start = datetime.timedelta(seconds=seg["start"])
            end = datetime.timedelta(seconds=seg["end"])
            subs.append(srt.Subtitle(index=i, start=start, end=end, content=seg["text"].strip()))

        srt_data = srt.compose(subs)

        return (
            srt_data,
            200,
            {
                "Content-Type": "application/x-subrip",
                "Content-Disposition": 'attachment; filename="subtitles.srt"'
            }
        )

    finally:
        if os.path.exists(tmp_video_path):
            os.remove(tmp_video_path)
        if os.path.exists(tmp_audio_path):
            os.remove(tmp_audio_path)


@app.route("/memory_status", methods=["GET"])
def memory_status():
    try:
        status = {
            "device": DEVICE,
            "cuda_available": torch.cuda.is_available(),
            "whisper_loaded": whisper_model is not None,
            "llama_loaded": llama_model is not None,
            "tokenizer_loaded": tokenizer is not None
        }
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            status.update({
                "allocated_gb": round(allocated, 2),
                "reserved_gb": round(reserved, 2),
                "total_gb": round(total, 2),
                "free_gb": round(total - allocated, 2),
                "utilization_percent": round((allocated / total) * 100, 1)
            })
        else:
            status["error"] = "CUDA not available - running on CPU"
            
        return jsonify(status)
    except Exception as e:
        return jsonify({"error": f"Memory status error: {str(e)}"}), 500

@app.route("/test_summary", methods=["POST"])
def test_summary():
    """Test summary functionality without translation"""
    try:
        json_data = request.get_json()
        if not json_data or 'transcript' not in json_data:
            return jsonify({"error": "No transcript data provided"}), 400

        full_transcript = json_data['transcript']
        summary = summarize_transcript(full_transcript)
        
        return jsonify({
            "summary": summary,
            "original_length": len(full_transcript),
            "summary_length": len(summary)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/test_models", methods=["GET"])
def test_models():
    """Test if models can be loaded and work properly"""
    try:
        results = {
            "whisper_test": False,
            "llama_test": False,
            "errors": []
        }
        
        # Test Whisper
        try:
            whisper_model_test = load_whisper_model()
            if whisper_model_test is not None:
                results["whisper_test"] = True
            else:
                results["errors"].append("Whisper model failed to load")
        except Exception as e:
            results["errors"].append(f"Whisper test failed: {str(e)}")
        
        # Test Llama
        try:
            llama_model_test, tokenizer_test = load_llama_model()
            if llama_model_test is not None and tokenizer_test is not None:
                # Try a simple generation
                test_prompt = "Hello, this is a test."
                inputs = tokenizer_test(test_prompt, return_tensors="pt", max_length=50, truncation=True)
                if next(llama_model_test.parameters()).is_cuda:
                    inputs = inputs.to(DEVICE)
                
                with torch.no_grad():
                    outputs = llama_model_test.generate(
                        **inputs,
                        max_new_tokens=10,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer_test.eos_token_id,
                        eos_token_id=tokenizer_test.eos_token_id
                    )
                
                decoded = tokenizer_test.decode(outputs[0], skip_special_tokens=True).strip()
                if decoded:
                    results["llama_test"] = True
                else:
                    results["errors"].append("Llama model generated empty output")
            else:
                results["errors"].append("Llama model or tokenizer failed to load")
        except Exception as e:
            results["errors"].append(f"Llama test failed: {str(e)}")
        
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": f"Model test error: {str(e)}"}), 500

@app.route("/translate", methods=["POST"])
def translate():
    if "file" not in request.files:
        abort(400, description="No file uploaded")

    audio_file = request.files["file"]
    target_language = request.form.get("target_language") or "en"
    source_language = request.form.get("source_language")
    if source_language is not None:
        source_language = source_language.strip()
    if not source_language:
        source_language = None

    # Save uploaded file to temp location
    with NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.filename)[1]) as tmp_input:
        audio_file.save(tmp_input.name)
        tmp_input_path = tmp_input.name

    with NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
        tmp_wav_path = tmp_wav.name

    try:
        # Convert input audio file to WAV for Whisper
        ffmpeg.input(tmp_input_path).output(tmp_wav_path, format="wav", ar="16k").run(quiet=True, overwrite_output=True)

        model = load_whisper_model()

        # Prepare Whisper transcription kwargs
        transcribe_kwargs = {}
        if source_language:
            transcribe_kwargs["language"] = source_language

        result = model.transcribe(tmp_wav_path, **transcribe_kwargs)

        detected_language = result.get("language", "")
        used_source_lang = source_language if source_language else detected_language
        original_text = result.get("text", "").strip()

        # Move installed_languages assignment here before logging
        installed_languages = argostranslate.translate.get_installed_languages()

        # Now logging is safe
        app.logger.info(f"Requested source_language={source_language}, target_language={target_language}")
        app.logger.info(f"Used source language for translation: {used_source_lang}")
        app.logger.info(f"Installed languages: {[lang.code for lang in installed_languages]}")
        app.logger.info(f"Transcription result text length: {len(original_text)}")

        if not original_text:
            return jsonify({"error": "No speech or text detected in audio."}), 400

        from_lang = next((l for l in installed_languages if l.code == used_source_lang), None)
        to_lang = next((l for l in installed_languages if l.code == target_language), None)

        if not from_lang or not to_lang:
            return jsonify({
                "error": "Selected language(s) not installed",
                "available_languages": [lang.code for lang in installed_languages]
            }), 400

        try:
            translation = from_lang.get_translation(to_lang).translate(original_text)
        except Exception as e:
            app.logger.error("Translation error", exc_info=True)
            return jsonify({"error": f"Translation failed: {str(e)}"}), 400


        return jsonify({
            "detected_language": detected_language,
            "original_text": original_text,
            "translated_text": translation
        })

    finally:
        if os.path.exists(tmp_input_path):
            os.remove(tmp_input_path)
        if os.path.exists(tmp_wav_path):
            os.remove(tmp_wav_path)

@app.route("/stt", methods=["POST"])
def stt():
    audio_data = io.BytesIO(request.data)
    audio = AudioSegment.from_file(audio_data, format="opus")  # or "raw" depending on Jigasi
    audio.export("temp.wav", format="wav")
    result = model.transcribe("temp.wav", language="da")
    os.remove("temp.wav")
    return jsonify({"transcript": result["text"]})

# ==== SKYNET-COMPATIBLE STREAMING WHISPER SERVER ====

# Enhanced connection management for WebSocket with Skynet compatibility
class WhisperConnectionManager:
    def __init__(self):
        self.connections = {}
        self.flush_audio_task = None
        
    async def connect(self, websocket, room_name):
        self.connections[room_name] = {
            'websocket': websocket,
            'participants': {},
            'previous_transcription_tokens': [],
            'previous_transcription_store': [],
            'meeting_language': None,
            'tokenizer': None
        }
        print(f"Connected to room: {room_name}")
        
        # Start flush audio worker if not already running
        if self.flush_audio_task is None:
            import asyncio
            loop = asyncio.get_running_loop()
            self.flush_audio_task = loop.create_task(self.flush_working_audio_worker())
        
    def disconnect(self, room_name):
        if room_name in self.connections:
            del self.connections[room_name]
        print(f"Disconnected from room: {room_name}")
    
    async def flush_working_audio_worker(self):
        """Force transcription for participants with accumulated audio"""
        while True:
            try:
                for room_name, room_data in self.connections.items():
                    for participant_id, participant_data in room_data['participants'].items():
                        if (participant_data.get('working_audio') and 
                            len(participant_data['working_audio']) > 0 and 
                            not participant_data.get('is_transcribing', False)):
                            
                            # Check if enough time has passed since last chunk
                            current_time = int(datetime.datetime.now().timestamp() * 1000)
                            last_chunk_time = participant_data.get('last_received_chunk', 0)
                            if current_time - last_chunk_time > 2000:  # 2 seconds
                                print(f"Forcing transcription for {participant_id} in room {room_name}")
                                await self.force_transcription(room_name, participant_id)
                
                await asyncio.sleep(1)
            except Exception as e:
                print(f"Error in flush worker: {e}")
                await asyncio.sleep(1)
    
    async def force_transcription(self, room_name, participant_id):
        """Force transcription for a participant"""
        if room_name not in self.connections:
            return
        
        room_data = self.connections[room_name]
        if participant_id not in room_data['participants']:
            return
        
        participant_data = room_data['participants'][participant_id]
        working_audio = participant_data.get('working_audio', b'')
        
        if len(working_audio) > 0:
            try:
                # Transcribe the accumulated audio
                result = await self.transcribe_audio(working_audio, participant_data.get('language', 'en'))
                
                if result and result.get('text', '').strip():
                    # Send transcription result
                    response = {
                        "type": "final",
                        "text": result['text'].strip(),
                        "participant": participant_id,
                        "language": result.get('language', participant_data.get('language', 'en')),
                        "timestamp": int(datetime.datetime.now().timestamp() * 1000)
                    }
                    
                    # Send response based on room type
                    if room_name.startswith('socketio-'):
                        self.send_to_socketio_room(room_name, response)
                    else:
                        await self.send_to_room(room_name, response)
                    
                    # Reset participant audio buffer
                    participant_data['working_audio'] = b''
                    participant_data['working_audio_starts_at'] = 0
                    
            except Exception as e:
                print(f"Error in force transcription: {e}")
    
    async def transcribe_audio(self, audio_bytes, language='en'):
        """Transcribe audio bytes using Whisper"""
        try:
            # Convert bytes to numpy array
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float = audio_data.astype(np.float32) / 32768.0
            
            # Load whisper model
            model = load_whisper_model()
            
            # Transcribe
            transcribe_kwargs = {}
            if language and language != 'auto':
                transcribe_kwargs['language'] = language
            
            result = model.transcribe(
                audio_float,
                **transcribe_kwargs,
                fp16=False,
                beam_size=1,
                temperature=0.0
            )
            
            return {
                'text': result.get('text', ''),
                'language': result.get('language', language)
            }
            
        except Exception as e:
            print(f"Transcription error: {e}")
            return None
    
    async def send_to_room(self, room_name, message):
        """Send message to all participants in a room"""
        if room_name in self.connections:
            try:
                websocket = self.connections[room_name]['websocket']
                await websocket.send(json.dumps(message))
            except Exception as e:
                print(f"Error sending message to room {room_name}: {e}")
    
    def send_to_socketio_room(self, room_name, message):
        """Send message to SocketIO room"""
        try:
            # Extract session ID from room name (format: socketio-{session_id})
            if room_name.startswith('socketio-'):
                session_id = room_name.replace('socketio-', '')
                socketio.emit('transcription', message, room=session_id, namespace='/local-whisper/ws')
        except Exception as e:
            print(f"Error sending message to SocketIO room {room_name}: {e}")

connection_manager = WhisperConnectionManager()

async def handle_websocket_client(websocket, path):
    """
    Handle WebSocket connections for live transcription.
    Compatible with Skynet's streaming whisper format.
    """
    print(f"New WebSocket connection from {websocket.remote_address} - Path: {path}")
    
    # Extract room name from path - handle /local-whisper/ws/roomname format from nginx proxy
    path_parts = path.strip('/').split('/')
    if len(path_parts) >= 3 and path_parts[0] == 'local-whisper' and path_parts[1] == 'ws':
        room_name = path_parts[2]  # /local-whisper/ws/roomname format
    elif len(path_parts) >= 2 and path_parts[0] == 'ws':
        room_name = path_parts[1]  # /ws/roomname format
    elif len(path_parts) >= 1 and path_parts[0]:
        room_name = path_parts[0]  # /roomname format
    else:
        room_name = 'default-room'
    
    print(f"Room name extracted: {room_name}")
    
    try:
        await connection_manager.connect(websocket, room_name)
        
        # Send confirmation
        try:
            await websocket.send(json.dumps({
                "type": "status",
                "message": "Connected to local whisper transcription service",
                "room": room_name,
                "timestamp": int(datetime.datetime.now().timestamp() * 1000)
            }))
        except Exception as e:
            print(f"Failed to send confirmation: {e}")
        
        print(f"WebSocket connection established for room: {room_name}")
        
        async for message in websocket:
            try:
                if isinstance(message, bytes):
                    # Parse the Skynet format: 60-byte header + PCM audio data
                    if len(message) < 60:
                        continue
                        
                    # Extract header (participant_id|language)
                    header_bytes = message[:60]
                    header_str = header_bytes.decode('utf-8', errors='ignore').rstrip('\0')
                    
                    # Extract participant and language
                    if '|' in header_str:
                        participant_id, language = header_str.split('|', 1)
                    else:
                        participant_id = 'unknown'
                        language = 'en'
                    
                    # Extract PCM audio data
                    pcm_data = message[60:]
                    
                    if len(pcm_data) > 0:
                        # Process audio chunk with enhanced connection manager
                        await process_audio_chunk(room_name, participant_id, language, pcm_data)
                        
            except Exception as e:
                print(f"Error processing audio: {e}")
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": f"Transcription error: {str(e)}"
                }))
                
    except websockets.exceptions.ConnectionClosed:
        print(f"WebSocket connection closed for room: {room_name}")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        connection_manager.disconnect(room_name)

async def process_audio_chunk(room_name, participant_id, language, pcm_data):
    """Process audio chunk with Skynet-compatible logic"""
    try:
        if room_name not in connection_manager.connections:
            return
        
        room_data = connection_manager.connections[room_name]
        
        # Initialize participant if not exists
        if participant_id not in room_data['participants']:
            room_data['participants'][participant_id] = {
                'working_audio': b'',
                'working_audio_starts_at': 0,
                'language': language,
                'last_received_chunk': int(datetime.datetime.now().timestamp() * 1000),
                'is_transcribing': False,
                'chunk_count': 0
            }
        
        participant_data = room_data['participants'][participant_id]
        
        # Update participant data
        participant_data['last_received_chunk'] = int(datetime.datetime.now().timestamp() * 1000)
        participant_data['chunk_count'] += 1
        
        # Add audio to working buffer
        if not participant_data['working_audio']:
            participant_data['working_audio_starts_at'] = participant_data['last_received_chunk']
        
        participant_data['working_audio'] += pcm_data
        
        # Check if we should transcribe (accumulated enough audio)
        audio_length_samples = len(participant_data['working_audio']) // 2  # 2 bytes per sample
        audio_length_seconds = audio_length_samples / 16000  # 16kHz sample rate
        
        # Transcribe if we have at least 1 second of audio and not already transcribing
        if audio_length_seconds >= 1.0 and not participant_data['is_transcribing']:
            participant_data['is_transcribing'] = True
            
            try:
                # Transcribe the accumulated audio
                result = await connection_manager.transcribe_audio(
                    participant_data['working_audio'], 
                    participant_data['language']
                )
                
                if result and result.get('text', '').strip():
                    # Send transcription result
                    response = {
                        "type": "final",
                        "text": result['text'].strip(),
                        "participant": participant_id,
                        "language": result.get('language', participant_data['language']),
                        "timestamp": int(datetime.datetime.now().timestamp() * 1000)
                    }
                    
                    # Send response based on room type
                    if room_name.startswith('socketio-'):
                        connection_manager.send_to_socketio_room(room_name, response)
                    else:
                        await connection_manager.send_to_room(room_name, response)
                    
                    # Reset audio buffer after successful transcription
                    participant_data['working_audio'] = b''
                    participant_data['working_audio_starts_at'] = 0
                    
                    print(f"Transcribed from {participant_id} ({language}): {result['text'].strip()}")
                
            except Exception as e:
                print(f"Transcription error for {participant_id}: {e}")
            finally:
                participant_data['is_transcribing'] = False
        
    except Exception as e:
        print(f"Error processing audio chunk: {e}")

# WebSocket server
async def start_websocket_server():
    """Start the WebSocket server for Skynet-compatible streaming whisper"""
    print("Starting local WebSocket server on port 5001...")
    
    async def websocket_handler(websocket, path):
        await handle_websocket_client(websocket, path)
    
    try:
        # Configure server with basic settings
        server = await websockets.serve(
            websocket_handler, 
            "0.0.0.0",  # Listen on IPv4 only for better compatibility
            5001,
            # Set ping/pong settings
            ping_interval=30,
            ping_timeout=10,
            # Max message size (10MB for large audio chunks)
            max_size=10 * 1024 * 1024
        )
        print("Local whisper WebSocket server started on ws://0.0.0.0:5001")
        print("Server configured for proxy compatibility")
        print(f"Server is running: {server}")
        await server.wait_closed()
    except Exception as e:
        print(f"Error starting WebSocket server: {e}")
        import traceback
        traceback.print_exc()

# Add health endpoint for the local whisper websocket server
@app.route("/local-whisper/health")
def local_whisper_health():
    return jsonify({
        "status": "healthy", 
        "service": "local_streaming_whisper",
        "connections": len(connection_manager.connections),
        "whisper_loaded": whisper_model is not None,
        "llama_loaded": llama_model is not None
    })

# Add health endpoint for the local whisper websocket server (alternative path)
@app.route("/health")
def health():
    return jsonify({
        "status": "healthy", 
        "service": "local_streaming_whisper",
        "connections": len(connection_manager.connections),
        "whisper_loaded": whisper_model is not None,
        "llama_loaded": llama_model is not None,
        "message": "Server is running and ready to accept requests"
    })

# Add broadcast endpoint for compatibility with Skynet
@app.route("/local-whisper/broadcast", methods=["POST"])
def broadcast_caption():
    """Broadcast a caption to all WebSocket clients in a room (Skynet compatibility)"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        room_name = data.get('room')
        text = data.get('text')
        participant = data.get('participant', 'System')
        
        if not room_name or not text:
            return jsonify({"error": "Missing room or text"}), 400
        
        # Send to WebSocket clients in the room
        if room_name in connection_manager.connections:
            message = {
                "type": "broadcast",
                "text": text,
                "participant": participant,
                "timestamp": int(datetime.datetime.now().timestamp() * 1000)
            }
            
            # Send message based on room type
            if room_name.startswith('socketio-'):
                connection_manager.send_to_socketio_room(room_name, message)
            else:
                # Use asyncio to send the message
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(connection_manager.send_to_room(room_name, message))
                loop.close()
            
            return jsonify({
                "broadcasted": True, 
                "room": room_name, 
                "text": text,
                "participant": participant
            })
        else:
            return jsonify({"error": f"No active connections for room {room_name}"}), 404
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Add room status endpoint
@app.route("/local-whisper/rooms")
def get_rooms():
    """Get status of all active rooms"""
    rooms_info = {}
    for room_name, room_data in connection_manager.connections.items():
        rooms_info[room_name] = {
            "participants": list(room_data['participants'].keys()),
            "participant_count": len(room_data['participants']),
            "meeting_language": room_data.get('meeting_language'),
            "connected": True
        }
    
    return jsonify({
        "rooms": rooms_info,
        "total_rooms": len(rooms_info)
    })

# Add WebSocket endpoint for Skynet compatibility
@socketio.on('connect', namespace='/local-whisper/ws')
def handle_local_whisper_connect(auth):
    """Handle WebSocket connections for local whisper via SocketIO"""
    from flask import request
    session_id = request.sid if hasattr(request, 'sid') else 'unknown'
    print(f"Local whisper WebSocket connection from {session_id}")
    emit('status', {
        'type': 'status',
        'message': 'Connected to local whisper transcription service',
        'timestamp': int(datetime.datetime.now().timestamp() * 1000)
    })

@socketio.on('audio_data', namespace='/local-whisper/ws')
def handle_local_whisper_audio(data):
    """Handle audio data for local whisper transcription via SocketIO"""
    try:
        from flask import request
        session_id = request.sid if hasattr(request, 'sid') else 'unknown'
        room_name = f"socketio-{session_id}"
        
        # Parse the Skynet format: 60-byte header + PCM audio data
        if isinstance(data, bytes) and len(data) >= 60:
            # Extract header (participant_id|language)
            header_bytes = data[:60]
            header_str = header_bytes.decode('utf-8', errors='ignore').rstrip('\0')
            
            # Extract participant and language
            if '|' in header_str:
                participant_id, language = header_str.split('|', 1)
            else:
                participant_id = 'unknown'
                language = 'en'
            
            # Extract PCM audio data
            pcm_data = data[60:]
            
            if len(pcm_data) > 0:
                # Use asyncio to process the audio chunk
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(process_audio_chunk(room_name, participant_id, language, pcm_data))
                loop.close()
        
    except Exception as e:
        print(f"Error processing local whisper audio via SocketIO: {e}")
        emit('error', {
            'type': 'error',
            'message': f"Transcription error: {str(e)}"
        })

@socketio.on('disconnect', namespace='/local-whisper/ws')
def handle_local_whisper_disconnect():
    """Handle WebSocket disconnection for local whisper"""
    from flask import request
    session_id = request.sid if hasattr(request, 'sid') else 'unknown'
    print(f"Local whisper WebSocket disconnected: {session_id}")

# Add a simple WebSocket test endpoint
@app.route("/ws-test")
def ws_test():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>WebSocket Test</title>
    </head>
    <body>
        <h1>Local Whisper WebSocket Test</h1>
        <div id="status">Disconnected</div>
        <div id="messages"></div>
        <script>
            const ws = new WebSocket('ws://localhost:5001/test-room');
            const status = document.getElementById('status');
            const messages = document.getElementById('messages');
            
            ws.onopen = function() {
                status.textContent = 'Connected';
                status.style.color = 'green';
            };
            
            ws.onmessage = function(event) {
                const div = document.createElement('div');
                div.textContent = 'Received: ' + event.data;
                messages.appendChild(div);
            };
            
            ws.onclose = function() {
                status.textContent = 'Disconnected';
                status.style.color = 'red';
            };
            
            ws.onerror = function(error) {
                status.textContent = 'Error: ' + error;
                status.style.color = 'red';
            };
        </script>
    </body>
    </html>
    """

# ==== END WEBSOCKET SERVER ====

def run_flask_server():
    """Run Flask-SocketIO server with WebSocket support"""
    print("Starting Flask-SocketIO server on 0.0.0.0:5000 with WebSocket support")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True)

# Create a separate WebSocket endpoint using SocketIO instead
@socketio.on('connect', namespace='/ws')
def handle_ws_connect(auth):
    """Handle WebSocket connections via SocketIO"""
    from flask import request
    session_id = request.sid if hasattr(request, 'sid') else 'unknown'
    print(f"SocketIO WebSocket connection from {session_id}")
    emit('status', {
        'type': 'status',
        'message': 'Connected to local whisper transcription service via SocketIO',
        'timestamp': int(datetime.datetime.now().timestamp() * 1000)
    })

@socketio.on('audio_chunk', namespace='/ws')
def handle_audio_chunk(data):
    """Handle audio chunks via SocketIO"""
    try:
        # Extract data from SocketIO format
        if isinstance(data, dict):
            audio_data = data.get('audio_data')
            participant_id = data.get('participant_id', 'unknown')
            language = data.get('language', 'en')
            room_name = data.get('room_name', 'default')
        else:
            # Handle raw binary data (backwards compatibility)
            audio_data = data
            participant_id = 'unknown'
            language = 'en'
            room_name = 'default'
        
        if audio_data:
            # Convert base64 or process binary data
            import base64
            if isinstance(audio_data, str):
                # Base64 encoded
                audio_bytes = base64.b64decode(audio_data)
            else:
                audio_bytes = audio_data
            
            # Process like the WebSocket version
            if len(audio_bytes) >= 60:
                # Parse header if present
                if isinstance(audio_bytes, bytes) and len(audio_bytes) >= 60:
                    header_bytes = audio_bytes[:60]
                    header_str = header_bytes.decode('utf-8', errors='ignore').rstrip('\0')
                    
                    if '|' in header_str:
                        participant_id, language = header_str.split('|', 1)
                    
                    pcm_data = audio_bytes[60:]
                else:
                    pcm_data = audio_bytes
                
                if len(pcm_data) >= 8000:  # At least 0.5 seconds
                    # Convert PCM to float for whisper
                    audio_data_np = np.frombuffer(pcm_data, dtype=np.int16)
                    audio_float = audio_data_np.astype(np.float32) / 32768.0
                    
                    # Transcribe
                    model = load_whisper_model()
                    
                    transcribe_kwargs = {}
                    if language and language != 'auto':
                        transcribe_kwargs['language'] = language
                    
                    result = model.transcribe(
                        audio_float,
                        **transcribe_kwargs,
                        fp16=False,
                        temperature=0.0
                    )
                    
                    text = result["text"].strip()
                    
                    if text and len(text) > 2:
                        print(f"SocketIO Transcribed [{participant_id}, {language}]: {text}")
                        
                        # Send response via SocketIO
                        emit('transcription', {
                            'type': 'final',
                            'text': text,
                            'participant': participant_id,
                            'language': result.get('language', language),
                            'timestamp': int(datetime.datetime.now().timestamp() * 1000)
                        })
    
    except Exception as e:
        print(f"Error processing SocketIO audio chunk: {e}")
        emit('error', {
            'type': 'error',
            'message': str(e)
        })

@socketio.on('disconnect', namespace='/ws')
def handle_ws_disconnect():
    """Handle WebSocket disconnection"""
    from flask import request
    session_id = request.sid if hasattr(request, 'sid') else 'unknown'
    print(f"SocketIO WebSocket disconnected: {session_id}")

if __name__ == "__main__":    
    # Start WebSocket server in a separate thread
    import threading
    
    def start_websocket_server_thread():
        """Start the raw WebSocket server for Skynet compatibility"""
        try:
            print("Creating WebSocket server thread...")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            print("Starting WebSocket server in thread...")
            loop.run_until_complete(start_websocket_server())
        except Exception as e:
            print(f"Error in WebSocket server thread: {e}")
            import traceback
            traceback.print_exc()
    
    # Start WebSocket server in background thread
    ws_thread = threading.Thread(target=start_websocket_server_thread, daemon=True)
    ws_thread.start()
    
    # Run Flask-SocketIO server
    run_flask_server()
