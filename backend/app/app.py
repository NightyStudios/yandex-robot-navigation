from app.core.transformer import f
from fastapi import FastAPI, Request, HTTPException

app = FastAPI(openapi_prefix='/api/v1')


@app.get("/ping")
async def ping() -> dict:
    return {"status": "GOOOOOOOOL"}


@app.post("/frame")
async def upload_raw(image: Request):
    body = await image.body()

    if len(body) == 0:
        raise HTTPException(status_code=400, detail="Empty body")

    if len(body) > 2 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Too big")

    if body.startswith(b"\x89PNG\r\n\x1a\n"):
        file_type = "png"
    elif body.startswith(b"\xff\xd8"):
        file_type = "jpeg"
    else:
        file_type = "ne znayu"
        # raise HTTPException(status_code=400, detail="Unknown file format")

    # f - ОБРАБОТКА ТРАНСФОРМЕРОМ
    data = f(body)
    return data
