from io import BytesIO

from app.core.transformer import f

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse

app = FastAPI(root_path='/api/v1')


@app.get("/ping")
async def ping() -> dict:
    return {"status": "GOOOOOOOOL"}


@app.post("/frame")
async def get_coordinates(phrase: str, request: Request):
    body = await request.body()

    if not body:
        raise HTTPException(status_code=400, detail="Empty body")

    if body.startswith(b"\xff\xd8"):
        file_type = "jpeg"
    elif body.startswith(b"\x89PNG\r\n\x1a\n"):
        file_type = "png"
    else:
        file_type = "unknown"

    result = f(body)
    print(result)

    return {"status": "GOOOOOOOOL"}