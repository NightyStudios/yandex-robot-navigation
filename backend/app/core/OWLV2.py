import numpy as np
import torch
from PIL import Image
from io import BytesIO
import requests
from transformers import Owlv2Processor, Owlv2ForObjectDetection


def run_owlv2_inference_hardcoded(image_bytes: bytes, text_prompt: str):
    EXPECTED_SHAPE = (480, 640, 3)
    EXPECTED_DTYPE = np.uint8

    expected_size = np.prod(EXPECTED_SHAPE) * np.dtype(EXPECTED_DTYPE).itemsize
    if len(image_bytes) != expected_size:
        raise ValueError(
            f"Неверный размер байтовой строки! Ожидалось {expected_size} байт "
            f"(для формы {EXPECTED_SHAPE}), а получено {len(image_bytes)}."
        )

    image_array = np.frombuffer(image_bytes, dtype=EXPECTED_DTYPE).reshape(EXPECTED_SHAPE)

    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

    texts = [[text_prompt]]
    inputs = processor(text=texts, images=image_array, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    unnormalized_image = Image.fromarray(image_array)
    target_sizes = torch.Tensor([unnormalized_image.size[::-1]])
    results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.2)

    found_objects = []
    i = 0
    text = texts[i]
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

    print(f"\nРезультаты для запроса: '{text[0]}'")
    for box, score, label in zip(boxes, scores, labels):
        box_coords = [round(coord, 2) for coord in box.tolist()]
        obj_data = [text[label], round(score.item(), 3), box_coords]
        print(f"  - Найден объект: {obj_data[0]}, Уверенность: {obj_data[1]}, Координаты: {obj_data[2]}")
        found_objects.append(obj_data)

    return found_objects