const chatMessages = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const loadingIndicator = document.getElementById('loading-indicator');
const loadingTextContainer = document.getElementById('loading-text-container');
const customModal = document.getElementById('custom-modal');
const modalMessage = document.getElementById('modal-message');
const modalOkButton = document.getElementById('modal-ok-button');

let currentBotMessageDiv = null;
let controller = null;

// Theme management
const themeToggle = document.querySelector('.theme-toggle');
const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
const savedTheme = localStorage.getItem('theme');

// Initialize theme
let currentTheme = savedTheme || (prefersDark ? 'dark' : 'light');
document.documentElement.setAttribute('data-theme', currentTheme);

function toggleTheme() {
    currentTheme = currentTheme === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', currentTheme);
    localStorage.setItem('theme', currentTheme);
    const icon = document.getElementById("header-icon")
    const light_themeIcon = document.getElementById("light-themeIcon")
    const dark_themeIcon = document.getElementById("dark-themeIcon")
    const send_icon = document.getElementById("send-icon")
    if(currentTheme=='light'){
        light_themeIcon.src = "/static/icons/light.svg"
        dark_themeIcon.src = "/static/icons/dark-mode.svg"
        icon.src = "/static/icons/logo-bg-removed.png"
        send_icon.src = "/static/icons/send-dark.svg"
    }else{
        light_themeIcon.src = "/static/icons/light-light.svg"
        dark_themeIcon.src = "/static/icons/dark-dark.svg"
        icon.src = "/static/icons/logo-dark-removebg-preview-cp.png"
        send_icon.src = "/static/icons/send-light.svg"
    }
}

// Auto-resize textarea
// userInput.addEventListener('input', function () {
//     this.style.height = 'auto';
//     this.style.height = Math.min(this.scrollHeight, 128) + 'px';
// });

// function showCustomModal(message) {
//     modalMessage.textContent = message;
//     customModal.style.display = 'flex';
// }

// modalOkButton.onclick = function () {
//     customModal.style.display = 'none';
// }

// window.onclick = function (event) {
//     if (event.target == customModal) {
//         customModal.style.display = 'none';
//     }
// }

/**
 * Formats a plain text string with Markdown syntax to HTML using marked.js.
 * @param {string} text The input text with Markdown syntax.
 * @returns {string} The HTML-formatted string.
 */
function formatMarkdown(text) {
    // Add this check: If text is null, undefined, or empty, return an empty string.
    if (!text) {
        return "";
    }

    // This line will now only run if 'text' is a valid string.
    return marked.parse(text);
}


// in script.js

// in script.js

function appendMessage(sender, messageContent) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message-bubble');

    if (sender === 'user') {
        messageDiv.classList.add('user-message');
        messageDiv.textContent = messageContent;
    } else {
        messageDiv.classList.add('bot-message');
        // [+] ADD A DEDICATED CONTAINER FOR THE RESPONSE CONTENT
        const responseContentDiv = document.createElement('div');
        responseContentDiv.classList.add('bot-response-content');
        messageDiv.appendChild(responseContentDiv);
    }

    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return messageDiv;
}

async function sendMessage() {
    const message = userInput.value.trim();
    if (message === '') {
        showCustomModal("Please enter a message before sending.");
        return;
    }

    appendMessage('user', message);
    userInput.value = '';
    userInput.style.height = 'auto'; // Reset textarea height after sending
    sendButton.disabled = true;
    loadingIndicator.style.display = 'block';
    loadingTextContainer.textContent = "Processing your query...";

    currentBotMessageDiv = appendMessage('bot', '');
    const responseContentTarget = currentBotMessageDiv.querySelector('.bot-response-content');


    if (controller) {
        controller.abort();
    }
    controller = new AbortController();
    const signal = controller.signal;

    try {
        const response = await fetch('/stream_chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message }),
            signal: signal
        });

        if (!response.ok) {
            const errorText = await response.text();
            let errorMessage = `HTTP error! status: ${response.status}`;
            try {
                const sseDataPart = errorText.substring(errorText.indexOf('data:') + 5).trim();
                const errorData = JSON.parse(sseDataPart);
                errorMessage = errorData.error || errorMessage;
            } catch (e) {
                errorMessage = errorText || errorMessage;
            }
            throw new Error(errorMessage);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';
        let streamedContent = '';
        let sourcesAdded = false;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            let lastNewlineIndex = buffer.lastIndexOf('\n\n');

            while (lastNewlineIndex !== -1) {
                const eventString = buffer.substring(0, lastNewlineIndex + 2);
                buffer = buffer.substring(lastNewlineIndex + 2);

                const dataMatch = eventString.match(/data: (.*)/);
                if (dataMatch && dataMatch[1]) {
                    let parsedData;
                    try {
                        parsedData = JSON.parse(dataMatch[1]);
                    } catch (e) {
                        console.warn("Failed to parse SSE data as JSON:", dataMatch[1], e);
                        continue;
                    }

                    if (parsedData.status === 'DONE') {
                        loadingIndicator.style.display = 'none';
                        sendButton.disabled = false;
                        chatMessages.scrollTop = chatMessages.scrollHeight;
                        break;
                    }

                    if (parsedData.error) {
                        showCustomModal(`Streaming error: ${parsedData.error}`);
                        currentBotMessageDiv.innerHTML = formatMarkdown(`⚠️ Error: ${parsedData.error}`);
                        return;
                    }

                    if (parsedData.chunk) {
                        streamedContent += parsedData.chunk;
                        responseContentTarget.innerHTML = formatMarkdown(streamedContent);
                        // currentBotMessageDiv.innerHTML = formatMarkdown(streamedContent);
                    }

                    if (parsedData.sql_query) {
                        console.log("Appending SQL Query:", parsedData.sql_query);
                        const sqlContainer = document.createElement('div');
                        sqlContainer.classList.add('sql-query-container');
                        sqlContainer.innerHTML = `<strong>Generated SQL Query:</strong><pre>${parsedData.sql_query}</pre>`;

                        currentBotMessageDiv.appendChild(sqlContainer);
                    }

                    if (parsedData.sources && !sourcesAdded) {
                        console.log("Appending sources:", parsedData.sources);
                        const sourcesHtml = parsedData.sources.map(src => `<div class="source-line">${src}</div>`).join('');
                        const sourcesContainer = document.createElement('div');
                        sourcesContainer.classList.add('source-container');
                        sourcesContainer.innerHTML = `<strong>Sources:</strong>${sourcesHtml}`;

                        currentBotMessageDiv.appendChild(sourcesContainer);
                        sourcesAdded = true; // Only add once
                    }
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                }
                lastNewlineIndex = buffer.lastIndexOf('\n\n');
            }
        }


    } catch (error) {
        if (error.name === 'AbortError') {
            console.log('Fetch aborted.');
            return;
        }
        // console.error('Error initiating or reading stream:', error);

        fetch('/frontend_log', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                type: "fetch_stream_error",
                error_message: error.message,
                stack: error.stack,
                user_prompt: message,
                user_agent: navigator.userAgent
            })
        });

        showCustomModal(`Failed to get response: ${error.message}. Please check the server logs.`);
        if (currentBotMessageDiv) {
            currentBotMessageDiv.innerHTML = formatMarkdown("Sorry, something went wrong. Please try again.");
        }
    } finally {
        loadingIndicator.style.display = 'none';
        sendButton.disabled = false;
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
}

sendButton.addEventListener('click', sendMessage);

userInput.addEventListener('keypress', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault(); // Prevent new line in textarea
        sendMessage();
    }
});

// // Focus input on load
userInput.focus();

// Global frontend error logger
window.onerror = function (message, source, lineno, colno, error) {
    console.error("Global JS Error:", { message, source, lineno, colno, error });
    fetch('/frontend_log', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            type: "js_runtime_error",
            message: message,
            source: source,
            lineno: lineno,
            colno: colno,
            error_stack: error ? error.stack : null,
            user_agent: navigator.userAgent
        })
    });
};


window.addEventListener("beforeunload", () => {
  navigator.sendBeacon('/end_session');
});

window.onload = () => {
    fetch('/start_session', { method: 'GET' })
      .then(res => res.json())
      .then(data => console.log('Session started:', data))
      .catch(console.error);
};

// Capture unhandled promise rejections
window.addEventListener('unhandledrejection', function (event) {
    console.error("Unhandled Promise Rejection:", event.reason);
    fetch('/frontend_log', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            type: "unhandled_promise_rejection",
            reason: event.reason,
            user_agent: navigator.userAgent
        })
    });
});

