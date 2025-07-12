import os
import subprocess
import tempfile
import threading

from backend.stt_tts import get_stt_speechkit, summarize_objects_from_text_request_yandex, get_tts_speechkit, convert_raw_to_wav
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

    result = get_stt_speechkit(tmp_path)['result']
    summary = summarize_objects_from_text_request_yandex(result)

    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as tts_file:
        tts_path_raw = tts_file.name

    get_tts_speechkit(summary, output_path=tts_path_raw)
    tts_path_wav = tts_path_raw.split('.')[-1] + '.wav'
    convert_raw_to_wav(tts_path_raw, tts_path_wav)

    def play_and_cleanup(path_to_audio, path_to_input, path_to_cleanup):
        subprocess.run(["afplay", path_to_audio])
        os.remove(path_to_audio)
        os.remove(path_to_input)
        os.remove(path_to_cleanup)

    threading.Thread(target=play_and_cleanup, args=(tts_path_wav, tmp_path, tts_path_raw)).start()

    return {"summary": summary}
