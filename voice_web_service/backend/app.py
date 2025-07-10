import tempfile

from backend.stt_tts import get_transcribrion, summarize_objects_from_text_request
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
    return {"summary": summary}