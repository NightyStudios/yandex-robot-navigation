import os
import tempfile
import subprocess
import threading

from backend.stt_tts import get_transcribrion, summarize_objects_from_text_request, text_to_speach
from fastapi import FastAPI, UploadFile, File

api = FastAPI(
    root_path='/api/voice/v1'
)


@api.post('/upload')
async def upload_audio(audio: UploadFile = File(...)) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    result = get_transcribrion(tmp_path)
    summary = summarize_objects_from_text_request(result)

    # print(summary)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tts_file:
        tts_path = tts_file.name

    text_to_speach(summary, output_path=tts_path)

    def play_and_cleanup(path_to_audio, path_to_input):
        subprocess.run(["aplay", path_to_audio])
        os.remove(path_to_audio)
        os.remove(path_to_input)

    threading.Thread(target=play_and_cleanup, args=(tts_path, tmp_path)).start()

    return {"summary": summary}