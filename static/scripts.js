console.log("RespiroGuide scripts.js loaded.");

document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const resultDiv = document.getElementById('result');

    if (form) {
        form.addEventListener('submit', async (event) => {
            event.preventDefault();

            resultDiv.style.display = 'block';
            resultDiv.className = 'result-container'; 
            resultDiv.innerHTML = 'Predicting...';

            const formData = new FormData(form);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json(); 
                console.log("Received from backend:", result);
                console.log("Prediction status:", result.prediction_status);
                console.log("Prediction status type:", typeof result.prediction_status);
                console.log("Probability:", result.probability);
                console.log("Probability text:", result.probability_text);

                if (response.ok) {
                    // Normalize prediction status to handle any case issues
                    const normalizedStatus = result.prediction_status ? result.prediction_status.toLowerCase() : '';
                    console.log("Normalized prediction status:", normalizedStatus);
                    
                    // Determine the output message based on prediction_status
                    let statusMessage = '';
                    if (normalizedStatus === 'positive') {
                        console.log("Prediction matched 'Positive'");
                        statusMessage = '<strong>Suffering from COPD</strong>';
                    } else if (normalizedStatus === 'negative') {
                        console.log("Prediction matched 'Negative'");
                        statusMessage = '<strong>Not Suffering from COPD</strong>';
                    } else {
                        console.log("Prediction did not match expected values");
                        statusMessage = `Status: ${result.prediction_status}`; // Fallback
                    }

                    // Format the output with the new message
                    resultDiv.innerHTML = `
                        <strong>COPD Prediction Result</strong><br>
                        Based on the provided data, the prediction is: ${statusMessage}<br>
                        Probability: ${result.probability_text}
                    `;
                    resultDiv.classList.remove('error');
                } else {
                    resultDiv.innerHTML = `<strong>Error:</strong> ${result.error || 'Prediction failed'}`;
                    resultDiv.classList.add('error');
                }

            } catch (error) {
                console.error("Error submitting form:", error);
                resultDiv.innerHTML = '<strong>Error:</strong> An error occurred while submitting the form.';
                resultDiv.classList.add('error');
            }
        });
    }
}); 