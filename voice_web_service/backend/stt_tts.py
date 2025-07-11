import os
import subprocess

import openai
import torch
from dotenv import load_dotenv
from vosk_tts import Synth, Model as ttsModel
from yandex_cloud_ml_sdk import YCloudML

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

yandex_api_key = os.getenv("YANDEX_API_KEY")
yandex_folder_id = os.getenv("YANDEX_FOLDER_ID")

SAMPLE_RATE = 16000


def get_transcription_gpu(audio_path: str) -> str:
    """
    Использует GPU-версию Kaldi из репозитория vosk-api-gpu
    """
    process = subprocess.Popen(
        ["./vosk-api-gpu/bin/transcriber", "--model", "path_to_model", "--audio", audio_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    out, err = process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"Ошибка в transcriber: {err.decode()}")

    return out.decode().strip()


def summarize_objects_from_text_request_openai(prompt):
    system_promt = "You are an intermediate module for a robot, responsible for processing natural-language voice commands from users. Your primary function is to extract concrete, visually detectable objects that the robot should identify and approach.\
        Follow these exact instructions:\
        🔍 OBJECT EXTRACTION\
        Identify physical objects explicitly mentioned in the input.\
        Retain only descriptive attributes that are:\
        Objective\
        Visually observable (e.g., color, shape, size)\
        Remove all subjective, emotional, or personal descriptors, such as:\
        “my favorite”, “beautiful”, “scary”, “funny”, etc.\
        Example:\
        Input: 'Find my favorite red cat'\
        Output: кот;cat (“red” is ignored if not essential for visual identification or is subjective in context)\
        🌐 TRANSLATION FORMAT\
        For each object, output the pair:\
        [original phrase];[English translation]\
        Separate each object with a comma.\
        ✅ OUTPUT RULES\
        Use only essential adjectives: colors, shapes, sizes.\
        Each item must be formatted as:\
        [original adjective + noun];[translated adjective + noun]\
        If no valid attributes exist, output just the noun:\
        кошка;cat\
        🧠 EXAMPLES\
        User command: 'Покажи мою любимую красную пирамиду и огромную бутылку воды'\
        → Output: красная пирамида;red pyramid, большая бутылка;large bottle'\
        User command: 'Найди страшную игрушку и зелёный куб'\
        → Output: игрушка;toy, зелёный куб;green cube\
        User command: 'Подойди к моему милому синему ведру и любимому столу'\
        → Output: синее ведро;blue bucket, стол;table\
        This format ensures clarity, consistency, and high compatibility with downstream object detection systems.\
        Let me know if you'd like this as a structured JSON response format or used in a live NLP pipeline."
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": system_promt,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        return list(map(lambda x: x.split(";"), response.choices[0].message.content.split(", ")))

    except Exception as e:
        return f"An error occurred: {e}"


def summarize_objects_from_text_request_yandex(prompt: str) -> str:
    sdk = YCloudML(
        folder_id=yandex_folder_id,
        auth=yandex_api_key
    )

    model = sdk.models.completions("yandexgpt", model_version="rc")
    model = model.configure(temperature=0.3)
    try:
        result = model.run(
            [
                {"role": "system", "text": "You are an intermediate module for a robot, responsible for processing natural-language voice commands from users. Your primary function is to extract concrete, visually detectable objects that the robot should identify and approach.\
                Follow these exact instructions:\
                🔍 OBJECT EXTRACTION\
                Identify physical objects explicitly mentioned in the input.\
                Retain only descriptive attributes that are:\
                Objective\
                Visually observable (e.g., color, shape, size)\
                Remove all subjective, emotional, or personal descriptors, such as:\
                “my favorite”, “beautiful”, “scary”, “funny”, etc.\
                Example:\
                Input: 'Find my favorite red cat'\
                Output: кот;cat (“red” is ignored if not essential for visual identification or is subjective in context)\
                🌐 TRANSLATION FORMAT\
                For each object, output the pair:\
                [original phrase];[English translation]\
                Separate each object with a comma.\
                ✅ OUTPUT RULES\
                Use only essential adjectives: colors, shapes, sizes.\
                Each item must be formatted as:\
                [original adjective + noun];[translated adjective + noun]\
                If no valid attributes exist, output just the noun:\
                кошка;cat\
                🧠 EXAMPLES\
                User command: 'Покажи мою любимую красную пирамиду и огромную бутылку воды'\
                → Output: красная пирамида;red pyramid, большая бутылка;large bottle'\
                User command: 'Найди страшную игрушку и зелёный куб'\
                → Output: игрушка;toy, зелёный куб;green cube\
                User command: 'Подойди к моему милому синему ведру и любимому столу'\
                → Output: синее ведро;blue bucket, стол;table\
                This format ensures clarity, consistency, and high compatibility with downstream object detection systems.\
                Let me know if you'd like this as a structured JSON response format or used in a live NLP pipeline."},
                {
                    "role": "user",
                    "text": prompt,
                },
            ]
        )

        return result[0].text

    except Exception as e:
        return f"An error occurred: {e}"


def text_to_speach(text: str, output_path: str) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используемое устройство: {device}")

    model = ttsModel(model_name="vosk-model-tts-ru-0.8-multi")

    # Перемещаем веса на GPU
    model.model = model.model.to(device)

    # Некоторые TTS модели требуют передачи device в forward
    synth = Synth(model)
    synth.device = device  # <- добавим это, если не работает — перепишем forward

    synth.synth(f"Привет! Сейчас найду тебе {text[0][0]}!", output_path, speaker_id=2)

    print(f"Ответ сохранён в {output_path}")
