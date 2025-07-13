import numpy as np
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

print(f"PyTorch version: {torch.__version__}")
print(f"Is CUDA available: {torch.cuda.is_available()}")


model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
model = model.to('cuda')

processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")

import os
from PIL import Image, ImageDraw

# === Константы ===
INPUT_DIR = "simple_objects_for_demo2"
OUTPUT_DIR = "result3"
# TEXTS = [["striped pyramid", "yellow cube", "green cube", "red cube", "blue cube","lime cube", "ball"]]
# TEXTS = [["striped pyramid", "box", "yellow cube", "green cube", "red cube", "blue cube","lime cube", "ball", "window", "white rectangle", "chair", "desk", "heater"]]
TEXTS = [["striped pyramid", "box", "yellow cube", "green cube", "red cube", "blue cube", "lime cube", "ball", "window",
          "white rectangle", "headphones", "chair", "desk", "heater"]]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# === Функция для преобразования тензора обратно в изображение ===
def get_preprocessed_image(pixel_values):
    pixel_values = pixel_values.squeeze().cpu().numpy()
    unnormalized_image = (pixel_values * np.array(OPENAI_CLIP_STD)[:, None, None]) + np.array(OPENAI_CLIP_MEAN)[:, None,
                                                                                     None]
    unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
    unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
    return Image.fromarray(unnormalized_image)


# === Основной цикл по изображениям ===
for filename in os.listdir(INPUT_DIR):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(INPUT_DIR, filename)
    image = Image.open(image_path).convert("RGB")

    # Подготовка входов
    inputs = processor(text=TEXTS, images=image, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # Предсказание
    with torch.no_grad():
        outputs = model(**inputs)

    # Обработка результатов
    unnormalized_image = get_preprocessed_image(inputs["pixel_values"])
    target_sizes = torch.Tensor([unnormalized_image.size[::-1]])
    results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.2)[0]

    boxes, scores, labels = results["boxes"], results["scores"], results["labels"]

    # Визуализация как у тебя
    visualized_image = unnormalized_image.copy()
    draw = ImageDraw.Draw(visualized_image)
    text = TEXTS[0]

    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        # if score < 0.35:
        #     continue
        x1, y1, x2, y2 = tuple(box)
        draw.rectangle(xy=((x1, y1), (x2, y2)), outline="red")
        draw.text(xy=(x1, y1), text=text[label])

    # Сохранение результата
    output_path = os.path.join(OUTPUT_DIR, f"vis_{filename}")
    visualized_image.save(output_path)
    print(f"[✔] Сохранено: {output_path}")