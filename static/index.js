document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    const form = e.target;
    const btn = document.getElementById('submit-btn');
    const loader = btn.querySelector('.loader');
    const btnText = btn.querySelector('.btn-text');
    const resultContainer = document.getElementById('result-container');
    const predictionText = document.getElementById('prediction-text');

    // Show loading state
    btn.disabled = true;
    loader.classList.remove('hidden');
    btnText.textContent = 'Processing...';

    const formData = new FormData(form);
    const data = Object.fromEntries(formData.entries());

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });

        const result = await response.json();

        if (result.success) {
            resultContainer.classList.remove('hidden');
            predictionText.textContent = result.prediction;

            // Add styling based on result
            if (result.prediction.toLowerCase().includes('not')) {
                predictionText.className = 'not-canceled';
            } else {
                predictionText.className = 'canceled';
            }

            // Update Chart
            renderChart(result.probabilities);

            // Scroll to result
            resultContainer.scrollIntoView({ behavior: 'smooth' });
        } else {
            alert('Error: ' + result.error);
        }
    } catch (error) {
        console.error('Fetch error:', error);
        alert('An error occurred while connecting to the server.');
    } finally {
        // Reset button state
        btn.disabled = false;
        loader.classList.add('hidden');
        btnText.textContent = 'Predict Status';
    }
});

let myChart = null;

function renderChart(probabilities) {
    const ctx = document.getElementById('predictionChart').getContext('2d');

    if (myChart) {
        myChart.destroy();
    }

    myChart = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: ['Not Canceled', 'Canceled'],
            datasets: [{
                data: [probabilities.not_canceled * 100, probabilities.canceled * 100],
                backgroundColor: ['#22c55e', '#ef4444'],
                borderColor: 'rgba(255, 255, 255, 0.1)',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: '#94a3b8'
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            return `${context.label}: ${context.raw.toFixed(1)}%`;
                        }
                    }
                }
            }
        }
    });
}
