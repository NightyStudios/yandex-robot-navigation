import cv2 as cv
import time

import requests

def send_frame(image_bytes: str, server_url: str, phrase: str):
    """
    Отправляет байты изображения на сервер по указанному URL.

    :param image_bytes: Байты изображения (str)
    :param server_url: URL сервера, на который нужно отправить изображение
    :param phrase: Название объекта, который будет найден на сервере
    :return: Ответ от сервера (Response object)
    """
    try:
        response = requests.post(
            url=server_url,
            params={"phrase": phrase},
            data=image_bytes,  # просто байты JPEG
            headers={"Content-Type": "application/octet-stream"}
        )
        return response
    except requests.exceptions.RequestException as e:
        print(f"[Ошибка] Не удалось отправить изображение: {e}")
        return None


def main():
    cap = cv.VideoCapture(1)


    # ADDRESS_FOR_POST_IMAGE = "127.0.0.1:8080"
    ADDRESS_FOR_POST_IMAGE = "http://0.0.0.0:8000/api/v1/frame"
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
        quality_params = [int(cv.IMWRITE_JPEG_QUALITY), 100]  # качество от 0 до 100

        bytes_image = frame.tobytes()
        send_frame(bytes_image, ADDRESS_FOR_POST_IMAGE, 'phone')

        cv.imshow('frame', frame)
        time.sleep(1)
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


main()