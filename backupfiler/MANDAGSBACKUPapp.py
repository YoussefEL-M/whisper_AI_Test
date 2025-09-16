from flask import Flask, request, jsonify, render_template, abort, send_file, Response
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

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_name_or_path = "/opt/praktik/whisper/TinyLlama-1.1B-Chat-v1.0"

# Global variables for models (lazy loading)
whisper_model = None
llama_model = None
tokenizer = None

def load_whisper_model():
    """Load Whisper model on demand"""
    global whisper_model
    if whisper_model is None:
        print("Loading Whisper model...")
        # Try smaller model if turbo fails
        try:
            whisper_model = whisper.load_model("turbo", device=DEVICE)
        except torch.cuda.OutOfMemoryError:
            print("Turbo model too large, falling back to base model")
            whisper_model = whisper.load_model("base", device=DEVICE)
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
    """Load TinyLlama model on demand"""
    global llama_model, tokenizer
    if llama_model is None:
        print("Loading TinyLlama model...")
        tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
        # Load without quantization - just use float16 for memory efficiency
        llama_model = LlamaForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            device_map="auto",
            # Removed load_in_8bit=True to avoid bitsandbytes
        )
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

def paraphrase_text_hf(text: str) -> str:
    """Paraphrase text using TinyLlama"""
    # Unload Whisper first to make room
    unload_whisper_model()
    
    # Load TinyLlama
    model, tok = load_llama_model()
    
    prompt = f"Paraphrase this text:\n{text}\nParaphrased:"
    inputs = tok(prompt, return_tensors="pt", max_length=512, truncation=True).to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tok.eos_token_id
        )
    
    decoded = tok.decode(outputs[0], skip_special_tokens=True)
    
    # Extract paraphrased text
    if "Paraphrased:" in decoded:
        _, paraphrased = decoded.split("Paraphrased:", 1)
        return paraphrased.strip()
    return text  # Return original if parsing fails

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
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_VIDEO

def video_to_audio(video_path, audio_path):
    ffmpeg.input(video_path).output(audio_path).run()

ALLOWED_AUDIO = {'.mp3', '.wav', '.flac', '.m4a', '.ogg', '.webm'}
ALLOWED_VIDEO = {'.mp4', '.mov', '.avi', '.webm', '.mkv'}

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        abort(400, description="No file uploaded")
    audio_file = request.files["file"]
    language = request.form.get("language") or None

    # Save input to temp file
    with NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.filename)[1]) as tmp_input:
        audio_file.save(tmp_input.name)
        tmp_input_path = tmp_input.name

    # Convert to WAV
    with NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
        tmp_wav_path = tmp_wav.name

    try:
        ffmpeg.input(tmp_input_path).output(tmp_wav_path, format="wav", ar="16k").run(quiet=True, overwrite_output=True)

        # Unload TinyLlama if loaded to make room for Whisper
        unload_llama_model()
        
        # Load and use Whisper
        model = load_whisper_model()
        
        # Transcribe
        transcribe_kwargs = {}
        if language:
            transcribe_kwargs['language'] = language
        
        result = model.transcribe(tmp_wav_path, **transcribe_kwargs)

        return jsonify({
            "transcript": result["text"],
            "language": result.get("language", ""),
            "segments": result.get("segments", [])
        })

    finally:
        # Clean up
        if os.path.exists(tmp_input_path):
            os.remove(tmp_input_path)
        if os.path.exists(tmp_wav_path):
            os.remove(tmp_wav_path)

@app.route("/paraphrase_srt", methods=["POST"])
def paraphrase_srt():
    if "file" not in request.files:
        abort(400, description="No file uploaded")

    srt_file = request.files["file"]
    with NamedTemporaryFile(delete=False, suffix=".srt") as tmp_file:
        srt_file.save(tmp_file.name)
        tmp_path = tmp_file.name

    try:
        # Read SRT subtitles
        with open(tmp_path, "r", encoding="utf-8") as f:
            subtitles = list(srt.parse(f.read()))

        # Paraphrase each subtitle block
        paraphrased_subs = []
        for sub in subtitles:
            paraphrased_text = paraphrase_text_hf(sub.content)
            paraphrased_subs.append(srt.Subtitle(
                index=sub.index,
                start=sub.start,
                end=sub.end,
                content=paraphrased_text
            ))

        result_srt = srt.compose(paraphrased_subs)

        # Send result as downloadable SRT
        return Response(
            result_srt,
            mimetype="application/x-subrip",
            headers={
                "Content-Disposition": 'attachment; filename="paraphrased.srt"'
            }
        )
    finally:
        os.remove(tmp_path)

@app.route("/transcribe/srt", methods=["POST"])
def transcribe_srt():
    if "file" not in request.files:
        abort(400, description="No file Uploaded")
    video_file = request.files["file"]
    language = request.form.get("language") or None

    # Temp files
    with NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.filename)[1]) as tmp_video:
        video_file.save(tmp_video.name)
        tmp_video_path = tmp_video.name
    with NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
        tmp_audio_path = tmp_audio.name

    try:
        # Extract audio from video
        ffmpeg.input(tmp_video_path).output(tmp_audio_path, format="wav", ar="16k").run(quiet=True, overwrite_output=True)

        # Unload TinyLlama if loaded to make room for Whisper
        unload_llama_model()
        
        # Load and use Whisper
        model = load_whisper_model()
        
        # Transcribe with whisper
        transcribe_kwargs = {}
        if language:
            transcribe_kwargs["language"] = language

        result = model.transcribe(tmp_audio_path, **transcribe_kwargs)

        # Build SRT subtitles
        subs = []
        for i, seg in enumerate(result.get("segments", []), start=1):
            start = datetime.timedelta(seconds=seg["start"])
            end = datetime.timedelta(seconds=seg["end"])
            subs.append(srt.Subtitle(index=i, start=start, end=end, content=seg["text"].strip()))

        srt_data = srt.compose(subs)

        # Return downloadable file
        return (
            srt_data,
            200,
            {
                "Content-Type": "application/x-subrip",
                "Content-Disposition": 'attachment; filename="subtitles.srt"'
            }
        )

    finally:
        # Clean up temp files
        if os.path.exists(tmp_video_path):
            os.remove(tmp_video_path)
        if os.path.exists(tmp_audio_path):
            os.remove(tmp_audio_path)

# Add a route to check memory usage
@app.route("/memory_status", methods=["GET"])
def memory_status():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        return jsonify({
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "total_gb": round(total, 2),
            "free_gb": round(total - allocated, 2),
            "whisper_loaded": whisper_model is not None,
            "llama_loaded": llama_model is not None
        })
    return jsonify({"error": "CUDA not available"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)  # Set debug=False to avoid model reloading
