import os
import subprocess
import tempfile

from backend.stt_tts import get_stt_speechkit, summarize_objects_from_text_request_yandex, get_tts_speechkit, \
    convert_raw_to_wav
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

    tmp_ogg_path = tmp_path.replace(".webm", ".ogg")
    subprocess.run(["ffmpeg", "-i", tmp_path, "-acodec", "libopus", tmp_ogg_path], check=True)

    result = get_stt_speechkit(tmp_ogg_path)['result']
    print(f'LOG: Расшифрованная фраза: {result}')

    summary = summarize_objects_from_text_request_yandex(result)
    print(f'LOG: Выделенный объект: {summary}')

    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as tts_file:
        tts_path_raw = tts_file.name

    get_tts_speechkit(summary, output_path=tts_path_raw, find=True)
    tts_path_wav = tts_path_raw.split('.')[-1] + '.wav'
    convert_raw_to_wav(tts_path_raw, tts_path_wav)

    def play_and_cleanup(path_to_audio, *paths_to_cleanup):
        subprocess.run(["aplay", path_to_audio])
        for path in paths_to_cleanup:
            os.remove(path)
        os.remove(path_to_audio)

    play_and_cleanup(tts_path_wav, tmp_path, tmp_ogg_path, tts_path_raw)

    return {"result": f"Привет! Сейчас найду тебе {summary.split(';')[0]}!"}


@api.post('/play')
async def play_custom_sound(audio: UploadFile = File(...)):
    filetype = audio.content_type
    suffix = filetype.split('/')[-1]

    with tempfile.NamedTemporaryFile(suffix=f'.{suffix}', delete=False) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    subprocess.run(["aplay", tmp_path])
    os.remove(tmp_path)


@api.post('/say')
async def say_custom_phrase(text: str):
    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as tts_file:
        tts_path_raw = tts_file.name

    get_tts_speechkit(text, output_path=tts_path_raw)
    tts_path_wav = tts_path_raw.split('.')[-1] + '.wav'
    convert_raw_to_wav(tts_path_raw, tts_path_wav)

    def play_and_cleanup(path_to_audio, *paths_to_cleanup):
        subprocess.run(["aplay", path_to_audio])
        for path in paths_to_cleanup:
            os.remove(path)
        os.remove(path_to_audio)

    play_and_cleanup(tts_path_wav, tts_path_raw)


@api.get('/ping')
async def ping():
    subprocess.run(["aplay", 'voice_web_service/car.mp3'])


@api.get('/goal')
async def goal():
    subprocess.run(["aplay", 'voice_web_service/goal.mp3'])
