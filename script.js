let keyData = [];
const box = document.getElementById("typingBox");
const resultBox = document.getElementById("result");

box.addEventListener("keydown", e => {
    const now = performance.now();
    keyData.push({ keydown: now });
});

box.addEventListener("keyup", e => {
    const now = performance.now();
    keyData[keyData.length - 1].keyup = now;
});

async function sendData() {

    if (keyData.length < 5) {
        resultBox.style.display = "block";
        resultBox.innerHTML = "Please type a little more 😄";
        return;
    }

    const res = await fetch("/submit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ keystrokes: keyData })
    });

    const data = await res.json();

    let emotion = data.prediction || "Unknown";

    // Emotion color assignment
    let cssClass = "";
    if (emotion === "Happy") cssClass = "happy";
    else if (emotion === "Sad") cssClass = "sad";
    else if (emotion === "Calm") cssClass = "calm";
    else if (emotion === "Stressed") cssClass = "stressed";

    resultBox.style.display = "block";
    resultBox.innerHTML = `
        Detected Emotion:
        <span class="badge ${cssClass}">${emotion}</span>
    `;

    // reset keystroke buffer
    keyData = [];
}
