[!NOTE] Placeholder ¯\_(ツ)_/¯


**Для запуска бэка (пока без докера)**

`cd yandex-robot-navigation`
`python3 backend/run.py`

**Запустить tts-stt с веб страничкой чисто:**

создаем венв

`cd yandex-robot-navigation`

`pip3 install -r voice_web_service/requirements.txt`

Делаем .env файл с ключом апи:

`nano .env` # и в нано делаем что хотим

`python3 voice_web_service/run.py`