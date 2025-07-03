import os
import uvicorn

from app.app import app

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=5050)
