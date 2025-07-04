import cv2 as cv
import time

import requests

def send_image_bytes_to_server(image_bytes, server_url, file_name='image.jpg', content_type='image/jpeg'):
    """
    Отправляет байты изображения на сервер по указанному URL.

    :param image_bytes: Байты изображения (bytes)
    :param server_url: URL сервера, на который нужно отправить изображение
    :param file_name: Имя файла, которое будет передано серверу
    :param content_type: MIME-тип изображения ('image/jpeg', 'image/png' и т.д.)
    :return: Ответ от сервера (Response object)
    """
    # Подготовка данных для отправки
    files = {
        'file': (file_name, image_bytes, content_type)
    }

    try:
        response = requests.post(server_url, files=files)
        return response
    except requests.exceptions.RequestException as e:
        print(f"[Ошибка] Не удалось отправить изображение: {e}")
        return None

cap = cv.VideoCapture(0)


# ADDRESS_FOR_POST_IMAGE = "127.0.0.1:8080"
ADDRESS_FOR_POST_IMAGE = "http://localhost:8080/upload/"
WEIDTH_OF_VIDEO = 640
HEIGHT_OF_VIDEO = 480


# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'XVID')
cap.set(cv.CAP_PROP_FRAME_WIDTH, WEIDTH_OF_VIDEO)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, HEIGHT_OF_VIDEO)
fps = cap.get(cv.CAP_PROP_FPS)
print(f"Camera FPS: {fps}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    # Параметры качества сжатия JPEG
    quality_params = [int(cv.IMWRITE_JPEG_QUALITY), 20]  # качество от 0 до 100

    # Кодирование изображения в формат JPEG
    success, img_buf = cv.imencode('.jpg', frame, quality_params)
    if not success:
        raise RuntimeError("Ошибка при кодировании изображения")

    # Теперь img_buf содержит сжатое изображение в формате JPEG (в виде массива байтов)
    # Его можно сохранить в файл или использовать далее по назначению

    # Например, сохранение в файл:
    with open('output.jpg', 'wb') as f:
        bytes_image = img_buf.tobytes()
        f.write(bytes_image)

    send_image_bytes_to_server(bytes_image, ADDRESS_FOR_POST_IMAGE)

    cv.imshow('frame', frame)
    time.sleep(1)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()