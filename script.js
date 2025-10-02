// Canvas and drawing setup
const canvas = document.getElementById('drawingCanvas');
const ctx = canvas.getContext('2d');
let isDrawing = false;
let lastX = 0;
let lastY = 0;

// API endpoints
const API_URL = 'http://localhost:5000/predict';
const FEEDBACK_URL = 'http://localhost:5000/feedback';
const RETRAIN_URL = 'http://localhost:5000/retrain';

// Store current prediction data
let currentPrediction = null;
let currentImageData = null;

// Initialize canvas
function initCanvas() {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 15;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
}

// Clear canvas
function clearCanvas() {
    initCanvas();
    document.getElementById('results').style.display = 'none';
    document.getElementById('feedbackSection').style.display = 'block';
    document.getElementById('correctionSection').style.display = 'none';
    document.getElementById('learningStatus').style.display = 'none';
    currentPrediction = null;
    currentImageData = null;
    setStatus('', '');
}

// Get mouse/touch position relative to canvas
function getPosition(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    if (e.touches) {
        return {
            x: (e.touches[0].clientX - rect.left) * scaleX,
            y: (e.touches[0].clientY - rect.top) * scaleY
        };
    }
    return {
        x: (e.clientX - rect.left) * scaleX,
        y: (e.clientY - rect.top) * scaleY
    };
}

// Start drawing
function startDrawing(e) {
    isDrawing = true;
    const pos = getPosition(e);
    lastX = pos.x;
    lastY = pos.y;
}

// Draw on canvas
function draw(e) {
    if (!isDrawing) return;

    const pos = getPosition(e);

    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();

    lastX = pos.x;
    lastY = pos.y;
}

// Stop drawing
function stopDrawing() {
    isDrawing = false;
}

// Set status message
function setStatus(message, type) {
    const statusEl = document.getElementById('status');
    statusEl.textContent = message;
    statusEl.className = `status ${type}`;
}

// Display prediction results
function displayResults(data) {
    const resultsEl = document.getElementById('results');
    const predictedDigitEl = document.getElementById('predictedDigit');
    const confidenceEl = document.getElementById('confidence');
    const probBarsEl = document.getElementById('probBars');

    // Store prediction data
    currentPrediction = data;

    // Show predicted digit
    predictedDigitEl.textContent = data.prediction;
    confidenceEl.textContent = `Confidence: ${(data.confidence * 100).toFixed(2)}%`;

    // Create probability bars
    probBarsEl.innerHTML = '';
    data.probabilities.forEach((prob, digit) => {
        const barDiv = document.createElement('div');
        barDiv.className = 'prob-bar';

        const percentage = (prob * 100).toFixed(1);
        const isHighest = digit === data.prediction;

        barDiv.innerHTML = `
            <span class="prob-label">${digit}:</span>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill" style="width: ${percentage}%">
                    <span class="prob-value">${percentage}%</span>
                </div>
            </div>
        `;

        if (isHighest) {
            barDiv.style.fontWeight = 'bold';
        }

        probBarsEl.appendChild(barDiv);
    });

    // Reset feedback sections
    document.getElementById('feedbackSection').style.display = 'block';
    document.getElementById('correctionSection').style.display = 'none';
    document.getElementById('learningStatus').style.display = 'none';

    // Show results
    resultsEl.style.display = 'block';
    resultsEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Predict digit
async function predictDigit() {
    try {
        setStatus('Analyzing your drawing...', 'loading');

        // Convert canvas to base64 image
        currentImageData = canvas.toDataURL('image/png');

        // Send to API
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: currentImageData })
        });

        if (!response.ok) {
            throw new Error('Server error. Make sure the Flask server is running!');
        }

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        // Display results
        displayResults(data);
        setStatus('Prediction complete! Is it correct?', 'success');

    } catch (error) {
        console.error('Error:', error);
        setStatus(`Error: ${error.message}`, 'error');
        document.getElementById('results').style.display = 'none';
    }
}

// Handle correct feedback
async function handleCorrectFeedback() {
    if (!currentPrediction || !currentImageData) return;

    try {
        setStatus('Saving training data...', 'loading');

        const response = await fetch(FEEDBACK_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: currentImageData,
                label: currentPrediction.prediction,
                was_correct: true
            })
        });

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        // Hide feedback buttons
        document.getElementById('feedbackSection').style.display = 'none';

        // Show learning status
        const learningStatus = document.getElementById('learningStatus');
        const learningMessage = document.getElementById('learningMessage');
        learningStatus.className = 'learning-status success';
        learningMessage.textContent = `Great! Learned from your drawing. Total samples: ${data.samples_collected}`;
        learningStatus.style.display = 'block';

        setStatus('Training data saved successfully!', 'success');

        // Check if we should retrain (every 10 correct samples)
        if (data.samples_collected % 10 === 0 && data.samples_collected >= 10) {
            await triggerRetraining();
        }

    } catch (error) {
        console.error('Error:', error);
        setStatus(`Error: ${error.message}`, 'error');
    }
}

// Handle wrong feedback
function handleWrongFeedback() {
    // Hide feedback buttons
    document.getElementById('feedbackSection').style.display = 'none';

    // Show correction section
    document.getElementById('correctionSection').style.display = 'block';

    setStatus('Please select the correct digit', 'loading');
}

// Handle digit correction
async function handleDigitCorrection(correctDigit) {
    if (!currentImageData) return;

    try {
        setStatus('Learning from correction...', 'loading');

        const response = await fetch(FEEDBACK_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: currentImageData,
                label: correctDigit,
                was_correct: false
            })
        });

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        // Hide correction section
        document.getElementById('correctionSection').style.display = 'none';

        // Show learning status
        const learningStatus = document.getElementById('learningStatus');
        const learningMessage = document.getElementById('learningMessage');
        learningStatus.className = 'learning-status info';
        learningMessage.textContent = `Thanks for the correction! Learned that this is a ${correctDigit}. Total samples: ${data.samples_collected}`;
        learningStatus.style.display = 'block';

        setStatus('Correction saved! Draw another digit to continue.', 'success');

        // Check if we should retrain
        if (data.samples_collected % 10 === 0 && data.samples_collected >= 10) {
            await triggerRetraining();
        }

    } catch (error) {
        console.error('Error:', error);
        setStatus(`Error: ${error.message}`, 'error');
    }
}

// Trigger model retraining
async function triggerRetraining() {
    try {
        setStatus('Retraining model with your data... This may take a moment.', 'loading');

        const response = await fetch(RETRAIN_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });

        const data = await response.json();

        if (data.error) {
            console.error('Retraining error:', data.error);
            return;
        }

        const learningStatus = document.getElementById('learningStatus');
        const learningMessage = document.getElementById('learningMessage');
        learningStatus.className = 'learning-status success';
        learningMessage.innerHTML = `<strong>Model Retrained!</strong><br>Learned from ${data.samples_used} of your drawings.<br>Training accuracy: ${(data.accuracy * 100).toFixed(2)}%`;
        learningStatus.style.display = 'block';

        setStatus('Model successfully retrained with your data!', 'success');

    } catch (error) {
        console.error('Retraining error:', error);
    }
}

// Event listeners for mouse
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

// Event listeners for touch
canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    startDrawing(e);
});
canvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    draw(e);
});
canvas.addEventListener('touchend', (e) => {
    e.preventDefault();
    stopDrawing();
});

// Button event listeners
document.getElementById('clearBtn').addEventListener('click', clearCanvas);
document.getElementById('predictBtn').addEventListener('click', predictDigit);

// Feedback button listeners
document.getElementById('correctBtn').addEventListener('click', handleCorrectFeedback);
document.getElementById('wrongBtn').addEventListener('click', handleWrongFeedback);

// Digit selector button listeners
document.querySelectorAll('.digit-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const digit = parseInt(btn.getAttribute('data-digit'));
        handleDigitCorrection(digit);
    });
});

// Initialize canvas on load
initCanvas();

// Show a helpful message on first load
setStatus('Draw a digit and click "Predict Digit" to start!', '');
