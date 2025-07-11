import os
import subprocess

import openai
import torch
from dotenv import load_dotenv
from vosk import Model, KaldiRecognizer
from vosk_tts import Synth, Model as ttsModel

from yandex_cloud_ml_sdk import YCloudML

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

yandex_api_key = os.getenv("YANDEX_API_KEY")
yandex_folder_id = os.getenv("YANDEX_FOLDER_ID")

SAMPLE_RATE = 16000


def get_transcribrion(path: str) -> str:
    '''
    :param path: path to file that have to be transcribed
    :return: transcribed text
    '''
    model = Model(lang="ru")
    rec = KaldiRecognizer(model, SAMPLE_RATE)

    with subprocess.Popen(["ffmpeg", "-loglevel", "quiet", "-i",
                           path,
                           "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "s16le", "-"],
                          stdout=subprocess.PIPE) as process:

        while True:
            data = process.stdout.read(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                rec.Result()
            else:
                rec.PartialResult()

    text = rec.FinalResult()
    return text


def summarize_objects_from_text_request_openai(prompt):
    # Without translation
    # system_promt = "You are an intermediate processing module for a robot. Your task is to analyze the text of a user's voice command and extract the key objects that the robot should visually identify and physically approach.\
    #     For each object mentioned, you must also extract any associated descriptive words (e.g., colors, sizes, shapes, or other adjectives) that help define the object.\
    #     Your output should always be a concise, comma-separated list of these fully described objects, with each item formatted as: [adjective] [object] (e.g., 'red cube', 'green bottle').\
    #     Examples:\
    #     User command: 'Find me the red cube'\
    #     → Output: red cube\
    #     User command: 'Go to the red cube, the red pyramid, and the green bottle'\
    #     → Output: red cube, red pyramid, green bottle\
    #     Only include objects with clear descriptors or meaningful context, and keep the output as short and precise as possible."

    # With translation
    # system_promt = "You are an intermediate natural language processing module for a robot. Your primary function is to process the text of a user's voice command and output a list of key objects that the robot should visually locate and approach.\
    #     Before extracting the objects, automatically translate the command into English (if it's not already). Then perform the following steps:\
    #     Tasks:\
    #     Identify target objects mentioned in the command (e.g., cube, bottle, pyramid).\
    #     Extract any descriptive adjectives that modify or clarify the objects (e.g., color, size, shape).\
    #     Format each result as: [adjective] [object], and output them as a comma-separated list.\
    #     Do not include irrelevant words, verbs, or prepositions—only fully described, actionable object phrases.\
    #     Output Format:\
    #     A clean, comma-separated list of objects with descriptors.\
    #     Example:\
    #     Input: 'Найди мне красный куб'\
    #     → Translate: 'Find me the red cube'\
    #     → Output: red cube\
    #     Input: 'Подойди к зелёной бутылке и красной пирамиде'\
    #     → Translate: 'Approach the green bottle and the red pyramid'\
    #     → Output: green bottle, red pyramid\
    #     Notes:\
    #     Always output in English, even if the original command is in another language.\
    #     Do not explain or include additional text—only return the list of described objects.\
    #     Let me know if you'd like to integrate it into a code pipeline or connect with a speech-to-text layer!"

    # With translation and original
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

    model.model = model.model.to(device)
    synth = Synth(model)

    synth.synth(f"Привет! Сейчас найду тебе {text[0][0]}!", output_path, speaker_id=2)
    print(f"Ответ сохранён в {output_path}")

    return None

# user_input = text
# generated_text = summarize_objects_from_text_request(user_input)
# print(f"Extracted: {generated_text}")


# model = ttsModel(model_name="vosk-model-tts-ru-0.8-multi")
# synth = Synth(model)
#
# synth.synth(f"Привет! Сейчас найду тебе {generated_text[0][0]}!", "out.wav", speaker_id=2)
# print("Ответ сохранён в out.wav")
# Пожалуйста найди красный куб. Где то здесь зелёная бутылка. О, крутой белый попугай. Подойди к пирамидке. найди новый ноутбук
