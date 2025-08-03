import os
import tempfile
import requests
from flask import Flask, render_template, request, jsonify


ffmpeg_dir = r"C:\ffmpeg-7.1.1-essentials_build\bin"
os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")


import whisper  

app = Flask(__name__)


whisper_model = whisper.load_model("small")



OPENROUTER_API_KEY = "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"  
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"
MODEL              = "openai/gpt-3.5-turbo"


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    audio = request.files.get("audio")
    if not audio:
        return jsonify(error="No audio provided"), 400

    
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
    audio.save(tmp.name)
    tmp.close()

    
    if os.path.getsize(tmp.name) < 1024:
        os.remove(tmp.name)
        return jsonify(error="Audio too short, please try again."), 400

    
    try:
        result = whisper_model.transcribe(tmp.name)
        text = result.get("text", "").strip()
    except Exception as e:
        os.remove(tmp.name)
        return jsonify(error=f"STT error: {e}"), 500

    os.remove(tmp.name)
    return jsonify(transcription=text)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json() or {}
    user_msg = data.get("message", "").strip()
    if not user_msg:
        return jsonify(error="No message provided"), 400

    
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert assistant who ONLY answers questions about "
                "Related Indian Government Schemes , Govt plans useful for common citizen or students anything and related to them Dont be way too abstract. If asked anything else, reply: "
                "'I’m sorry, I can only answer questions about Indian Government Schemes.'"
            )
        },
        {"role": "user", "content": user_msg}
    ]

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"model": MODEL, "messages": messages}

    resp = requests.post(OPENROUTER_URL, headers=headers, json=payload)
    if not resp.ok:
        return jsonify(error="AI service error"), 500

    try:
        reply = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return jsonify(error="Malformed AI response"), 500

    return jsonify(reply=reply)

@app.route("/favicon.ico")
def favicon():
    return "", 204

if __name__ == "__main__":
    app.run(debug=True)
