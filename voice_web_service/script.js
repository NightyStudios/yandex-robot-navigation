let mediaRecorder;
let audioChunks = [];
let isRecording = false;

const button = document.getElementById("recordButton");
const icon = document.getElementById("buttonIcon");
const status = document.getElementById("status");

button.addEventListener("click", async () => {
  if (!isRecording) {
    // Начинаем запись
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];

    mediaRecorder.ondataavailable = e => {
      if (e.data.size > 0) audioChunks.push(e.data);
    };

    mediaRecorder.start();
    isRecording = true;
    button.classList.add("recording");
    icon.textContent = "■";
    status.textContent = "Идёт запись... Нажмите ещё раз, чтобы остановить.";
  } else {
    // Останавливаем запись и отправляем
    mediaRecorder.stop();
    isRecording = false;
    button.classList.remove("recording");
    icon.textContent = "●";
    status.textContent = "Отправка...";

mediaRecorder.onstop = async () => {
  const blob = new Blob(audioChunks, { type: 'audio/webm' });
  const formData = new FormData();
  formData.append("audio", blob, "recording.webm");

  try {
    const res = await fetch("http://127.0.0.1:5050/api/voice/v1/upload", {
      method: "POST",
      body: formData
    });

    // Логируем ответ
    console.log("Ответ от сервера:", res);

    if (res.ok) {
      const result = await res.json();
      console.log("JSON:", result);
      status.textContent = "Файл отправлен!";
    } else {
      const errText = await res.text();
      console.error("Ошибка ответа:", res.status, errText);
      status.textContent = `Ошибка при отправке: ${res.status}`;
    }
  } catch (err) {
    console.error("Ошибка fetch:", err);
    status.textContent = "Ошибка сети.";
  }
};

  }
});
