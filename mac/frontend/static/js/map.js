const canvas = document.getElementById('scoutMap');
const ctx = canvas.getContext('2d');
const centerX = canvas.width / 2;
const centerY = canvas.height / 2;
const scale = 3; // 1cm in real life = 3 pixels on screen

let robotPath = []; // Stores {x, y}
let resources = []; // Stores {x, y, type}

function drawGrid() {
    ctx.strokeStyle = '#003300';
    ctx.beginPath();
    // Draw Axis
    ctx.moveTo(0, centerY); ctx.lineTo(canvas.width, centerY);
    ctx.moveTo(centerX, 0); ctx.lineTo(centerX, canvas.height);
    ctx.stroke();
}
async function updateData() {
    try {
        const response = await fetch('/get_telemetry');
        const data = await response.json();

        // Update robot position
        robotPath.push({x: data.x, y: data.y});
        
        // Sync the resource list
        resources = data.resources; 

        render();
    } catch (e) { console.log("Searching for Scout..."); }
}

function render() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawGrid();

    // 1. Draw Path (Green Line)
    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 2;
    ctx.beginPath();
    robotPath.forEach((p, i) => {
        const sx = centerX + (p.x * scale);
        const sy = centerY - (p.y * scale); // Subtract because Y is up in math but down in JS
        if (i === 0) ctx.moveTo(sx, sy);
        else ctx.lineTo(sx, sy);
    });
    ctx.stroke();

    // 2. Draw Resources
    // 2. Draw Object Labels
    ctx.font = "bold 14px 'Courier New'";
    resources.forEach(res => {
        const rx = centerX + (res.x * scale);
        const ry = centerY - (res.y * scale);
        
        // Draw a small marker dot
        ctx.fillStyle = "#ffff00"; // Bright yellow for visibility
        ctx.beginPath();
        ctx.arc(rx, ry, 4, 0, Math.PI * 2);
        ctx.fill();

        // Draw the label text next to the dot
        ctx.fillStyle = "#ffffff";
        ctx.fillText(res.label, rx + 8, ry + 4);
    });

    // 3. Draw Robot (Current Position)
    const current = robotPath[robotPath.length - 1];
    if (current) {
        ctx.fillStyle = '#fff';
        ctx.fillRect(centerX + (current.x * scale) - 5, centerY - (current.y * scale) - 5, 10, 10);
    }
}

setInterval(updateData, 500); // Refresh every 0.5s