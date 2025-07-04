from app.core.transformer import f, set_title
from app.models import Title

from fastapi import FastAPI, Request, HTTPException

app = FastAPI(root_path='/api/v1')


@app.get("/ping")
async def ping() -> dict:
    return {"status": "GOOOOOOOOL"}


@app.post("/frame")
async def get_coordinates(request: Request):
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
    # print(result)

    return {"status": "GOOOOOOOOL"}


@app.post("/title")
async def set_title(title: Title):
    title = title.title
    res = set_title(title)

    return res
