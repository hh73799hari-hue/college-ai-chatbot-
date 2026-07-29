async function sendMessage(){

    let message=document.getElementById("message").value;

    if(message=="") return;

    let chat=document.getElementById("chat-box");

    chat.innerHTML+=`
    <div class="user">
        ${message}
    </div>
    `;

    chat.innerHTML+=`
    <div class="bot" id="typing">
        🤖 Typing...
    </div>
    `;

    chat.scrollTop=chat.scrollHeight;

    let response=await fetch("/chat",{
        method:"POST",
        headers:{
            "Content-Type":"application/json"
        },
        body:JSON.stringify({
            message:message
        })
    });

    let data=await response.json();

    document.getElementById("typing").remove();

    chat.innerHTML+=`
    <div class="bot">
        ${data.answer}
    </div>
    `;

    document.getElementById("message").value="";

    chat.scrollTop=chat.scrollHeight;
}

function enterKey(event){
    if(event.key==="Enter"){
        sendMessage();
    }
}