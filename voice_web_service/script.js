mediaRecorder.onstop = async () => {
  const blob = new Blob(audioChunks, { type: 'audio/webm' });
  const formData = new FormData();
  formData.append("audio", blob, "recording.webm");

  try {
    const res = await fetch("http://127.0.0.1:5050/upload", {
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
