async function sendMessage() {

    const input = document.getElementById("message");
    const message = input.value.trim();

    if (message === "") return;

    const chatBox = document.getElementById("chat-box");

    // User Message
    chatBox.innerHTML += `
        <div class="user-message">
            ${message}
        </div>
    `;

    input.value = "";

    // Typing Animation
    const typing = document.createElement("div");
    typing.className = "bot-message";
    typing.id = "typing";
    typing.innerHTML = "🤖 Typing...";
    chatBox.appendChild(typing);

    chatBox.scrollTop = chatBox.scrollHeight;

    try {

        const response = await fetch("/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                message: message
            })
        });

        const data = await response.json();

        document.getElementById("typing").remove();

        chatBox.innerHTML += `
            <div class="bot-message">
                🤖 ${data.answer}
            </div>
        `;

    } catch (error) {

        document.getElementById("typing").remove();

        chatBox.innerHTML += `
            <div class="bot-message">
                ❌ Error connecting to server.
            </div>
        `;

    }

    chatBox.scrollTop = chatBox.scrollHeight;
}

function enterKey(event){

    if(event.key==="Enter"){
        sendMessage();
    }

}