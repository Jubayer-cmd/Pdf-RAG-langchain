document.addEventListener('DOMContentLoaded', () => {
  const uploadForm = document.getElementById('upload-form');
  const uploadStatus = document.getElementById('upload-status');
  const chatForm = document.getElementById('chat-form');
  const messageInput = document.getElementById('message-input');
  const chatBox = document.getElementById('chat-box');
  const fileInput = document.getElementById('file-input');
  const fileNameElement = document.getElementById('file-name');

  fileInput.addEventListener('change', async () => {
    if (fileInput.files.length > 0) {
      const file = fileInput.files[0];
      fileNameElement.textContent = file.name;

      // Auto-process the file
      await processFile(file);
    } else {
      fileNameElement.textContent = 'No file chosen';
    }
  });

  async function processFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    showStatusMessage('Processing PDF...', '#94a3b8');

    try {
      const response = await fetch('/upload', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();

      if (response.ok) {
        showStatusMessage('PDF processed successfully!', 'green', 3000);
        addMessage(
          'assistant',
          `PDF "${result.filename}" has been processed. You can now ask questions about it.`,
        );
      } else {
        showStatusMessage(result.error || 'Processing failed.', 'red');
      }
    } catch (error) {
      console.error('Upload error:', error);
      showStatusMessage('An error occurred during processing.', 'red');
    }
  }

  chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const message = messageInput.value.trim();
    if (!message) return;

    addMessage('user', message);
    messageInput.value = '';

    const assistantMessageElement = addMessage('assistant', '');
    const assistantTextElement = assistantMessageElement.querySelector('p');

    try {
      const response = await fetch('/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: message }),
      });

      if (!response.body) return;
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let accumulatedResponse = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk
          .split('\n')
          .filter((line) => line.trim().startsWith('data:'));
        for (const line of lines) {
          const data = line.replace(/^data: /, '').trim();
          if (data) {
            accumulatedResponse += data;
            assistantTextElement.textContent = accumulatedResponse;
            chatBox.scrollTop = chatBox.scrollHeight;
          }
        }
      }
    } catch (error) {
      console.error('Chat error:', error);
      assistantTextElement.textContent =
        "Sorry, I couldn't connect to the server.";
    }
  });

  function showStatusMessage(message, color, duration = 0) {
    uploadStatus.textContent = message;
    uploadStatus.style.color = color;
    uploadStatus.style.opacity = 1;

    if (duration > 0) {
      setTimeout(() => {
        uploadStatus.style.opacity = 0;
      }, duration);
    }
  }

  function addMessage(sender, text) {
    const messageElement = document.createElement('div');
    messageElement.classList.add('message', sender);

    const p = document.createElement('p');
    p.textContent = text;
    messageElement.appendChild(p);

    chatBox.appendChild(messageElement);
    chatBox.scrollTop = chatBox.scrollHeight;
    return messageElement;
  }
});
