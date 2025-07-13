from app.core.OWLV2 import run_owlv2_inference_hardcoded

from fastapi import FastAPI, Request, HTTPException

app = FastAPI(root_path='/api/v1')


@app.get("/ping")
async def ping() -> dict:
    return {"status": "GOOOOOOOOL"}


@app.post("/frame")
async def get_coordinates(phrase: str, image: Request):
    body = await image.body()

    if not body:
        raise HTTPException(status_code=400, detail="Empty body")

    if body.startswith(b"\xff\xd8"):
        file_type = "jpeg"
    elif body.startswith(b"\x89PNG\r\n\x1a\n"):
        file_type = "png"
    else:
        file_type = "unknown"

    result = test(body, phrase)
    print(result)

    return {"result": result}


