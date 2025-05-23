document.addEventListener('DOMContentLoaded', function() {
  let pollingIntervalId = null; // To store the interval ID for polling
  let currentTaskId = null;     // To store the current task ID

  // --- Load models (unchanged) ---
  fetch('/api/models')
    .then(response => response.json())
    .then(data => {
      const modelsDiv = document.getElementById('known-models');
      data.models.forEach(model => {
        const modelEl = document.createElement('div');
        modelEl.classList.add('model-tag');
        modelEl.textContent = model;
        modelsDiv.appendChild(modelEl);
      });
    })
    .catch(error => console.error('Error loading models:', error));

  // Toggle visibility functionality
  const toggleVisibilityBtn = document.getElementById('toggle-visibility');
  const inputSection = document.getElementById('input-section');
  if (toggleVisibilityBtn && inputSection) {
    toggleVisibilityBtn.addEventListener('click', function() {
      if (inputSection.classList.contains('hidden')) {
        inputSection.classList.remove('hidden');
        this.textContent = 'Hide Input Fields';
      } else {
        inputSection.classList.add('hidden');
        this.textContent = 'Show Input Fields';
      }
    });
  }

  // Test connection button
  document.getElementById('test-connection').addEventListener('click', function() {
    const apiKey = document.getElementById('api-key').value;
    const provider = document.getElementById('provider').value;
    const model = document.getElementById('model').value;
    const temperature = parseFloat(document.getElementById('temperature').value);

    if (!apiKey || !provider || !model) {
      showError('Please fill in all fields: API key, provider, and model name.');
      return;
    }
    showCard('testing-card');
    fetch('/api/test-connection', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ api_key: apiKey, provider: provider, model: model, temperature: temperature })
    })
      .then(response => response.json())
      .then(data => {
        hideCard('testing-card');
        if (data.status === 'success') {
          alert('Connection successful! Sample response: ' + data.response_preview);
        } else {
          showError(data.message || 'Error connecting to provider');
        }
      })
      .catch(error => {
        hideCard('testing-card');
        showError('Error: ' + error.message);
      });
  });

  // Identify model button
  document.getElementById('identify-model').addEventListener('click', function() {
    // Clear any previous polling
    stopPolling();

    const apiKey = document.getElementById('api-key').value;
    const provider = document.getElementById('provider').value;
    const model = document.getElementById('model').value;
    const numSamples = document.getElementById('num-samples').value;
    const temperature = parseFloat(document.getElementById('temperature').value);

    if (!apiKey || !provider || !model) {
      showError('Please fill in all fields: API key, provider, and model name.');
      return;
    }

    // Show progress card and reset UI immediately
    showCard('progress-card');
    updateProgressUI(0, parseInt(numSamples) || 0); // Initialize with 0 progress

    // Start the identification task (expects task_id in response)
    fetch('/api/identify-model', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        api_key: apiKey,
        provider: provider,
        model: model,
        num_samples: parseInt(numSamples),
        temperature: temperature
      })
    })
      .then(response => {
        if (!response.ok) {
          // Handle immediate errors from starting the task
          return response.json().then(data => {
            throw new Error(data.error || `Error starting identification: ${response.statusText}`);
          });
        }
        return response.json();
      })
      .then(data => {
        if (data.task_id) {
          currentTaskId = data.task_id;
          // Start polling for status updates
          startPolling(currentTaskId, parseInt(numSamples));
        } else {
          // If backend doesn't return task_id, treat as error
          throw new Error('Backend did not return a task ID.');
        }
      })
      .catch(error => {
        hideCard('progress-card');
        showError('Error starting identification task: ' + error.message);
      });
  });

  // --- Polling function ---
  function pollStatus(taskId, totalSamples) {
    fetch(`/api/task-status/${taskId}`)
      .then(response => {
        if (!response.ok) {
          // Handle errors during polling
           return response.json().then(data => {
               throw new Error(data.message || `Polling failed: ${response.statusText}`);
           }).catch(() => {
               // If response is not JSON or empty
               throw new Error(`Polling failed: ${response.status} ${response.statusText}`);
           });
        }
        return response.json();
      })
      .then(data => {
        switch (data.status) {
          case 'processing':
          case 'pending':
            // Update progress bar
            const completed = data.completed_samples || 0;
            // Use totalSamples passed from initial request if backend doesn't provide it
            const total = data.total_samples || totalSamples || 1; // Avoid division by zero
            updateProgressUI(completed, total);
            break;
          case 'completed':
            // Task finished successfully
            stopPolling();
            hideCard('progress-card');
            displayResults(data.result); // Assuming results are nested under 'result'
            break;
          case 'error':
            // Task failed on backend
            stopPolling();
            hideCard('progress-card');
            showError(`Identification failed: ${data.message || 'Unknown error'}`);
            break;
          default:
            // Unknown status
             console.warn('Received unknown task status:', data.status);
             // Optionally stop polling or continue? For now, continue.
            // stopPolling();
            // showError(`Received unknown task status: ${data.status}`);
            break;
        }
      })
      .catch(error => {
        // Handle fetch errors during polling
        console.error('Polling error:', error);
        stopPolling();
        hideCard('progress-card');
        showError(`Error checking task status: ${error.message}`);
      });
  }

  // --- Helper to start polling ---
  function startPolling(taskId, totalSamples) {
    // Ensure any old interval is cleared first
    stopPolling();
    // Poll immediately, then set interval
    pollStatus(taskId, totalSamples);
    pollingIntervalId = setInterval(() => pollStatus(taskId, totalSamples), 2000); // Poll every 2 seconds
  }

  // --- Helper to stop polling ---
  function stopPolling() {
    if (pollingIntervalId) {
      clearInterval(pollingIntervalId);
      pollingIntervalId = null;
      currentTaskId = null; // Clear task ID when stopping
      console.log('Polling stopped.');
    }
  }

  // --- Helper to update progress UI ---
  function updateProgressUI(completed, total) {
     const progressElement = document.getElementById('sample-progress');
     const countElement = document.getElementById('sample-count');
     const totalElement = document.getElementById('sample-total'); // Assuming you add this element

     if (!progressElement || !countElement) return; // Exit if elements not found

     total = total || 1; // Avoid division by zero if total is 0 or undefined
     const percentage = Math.min(100, Math.max(0, (completed / total) * 100));

     countElement.textContent = completed;
     progressElement.style.width = percentage.toFixed(2) + '%';
     progressElement.setAttribute('aria-valuenow', completed);
     progressElement.setAttribute('aria-valuemax', total);

     // Optional: Update total samples display if you have an element for it
     if (totalElement) {
         totalElement.textContent = total;
     }
  }

  // Cancel button
  document.getElementById('cancel-identification').addEventListener('click', function() {
    stopPolling(); // Stop polling the backend
    hideCard('progress-card');
    // Note: This doesn't cancel the task on the *backend*.
    // A separate API call would be needed for true cancellation.
  });

  // Back button from error
  document.getElementById('back-to-form').addEventListener('click', function() {
    hideCard('error-card');
  });

  // Helper functions
  function showCard(cardId) {
    // Stop polling if we are showing a card that isn't the progress card
    if (cardId !== 'progress-card') {
        stopPolling();
    }
    // Hide all relevant cards
    document.querySelectorAll('.card').forEach(card => {
      if (['testing-card', 'progress-card', 'results-card', 'error-card'].includes(card.id)) {
        card.classList.add('hidden');
      }
    });
    // Show requested card
    const cardToShow = document.getElementById(cardId);
    if (cardToShow) {
        cardToShow.classList.remove('hidden');
    } else {
        console.error(`Card with ID ${cardId} not found.`);
    }
  }

  function hideCard(cardId) {
    const cardToHide = document.getElementById(cardId);
     if (cardToHide) {
        cardToHide.classList.add('hidden');
    } else {
        console.error(`Card with ID ${cardId} not found for hiding.`);
    }
  }

  function showError(message) {
    stopPolling(); // Ensure polling stops on error
    const errorMessageElement = document.getElementById('error-message');
    if(errorMessageElement){
        errorMessageElement.textContent = message;
        showCard('error-card');
    } else {
        console.error("Error message element not found. Error was:", message);
        alert("An error occurred: " + message); // Fallback alert
    }
  }

  // Display results
   function displayResults(data) {
     if (!data) {
       showError("Received empty results data.");
       return;
     }
    // Fill in basic info
    document.getElementById('input-model').textContent = data.input_model ?? 'N/A';
    document.getElementById('provider-name').textContent = data.provider ?? 'N/A';
    document.getElementById('predicted-model').textContent = data.predicted_model ?? 'N/A';
    document.getElementById('confidence').textContent = data.confidence ?? 'N/A';

    // Fill in top predictions table
    const predictionsTable = document.getElementById('top-predictions')?.querySelector('tbody');
    if (predictionsTable) {
        predictionsTable.innerHTML = ''; // Clear previous results
        if(data.top_predictions && Array.isArray(data.top_predictions)) {
            data.top_predictions.forEach(prediction => {
                const row = document.createElement('tr');
                const modelCell = document.createElement('td');
                modelCell.textContent = prediction.model ?? '?';
                row.appendChild(modelCell);
                const probCell = document.createElement('td');
                probCell.textContent = (typeof prediction.probability === 'number') ? (prediction.probability * 100).toFixed(2) + '%' : 'N/A';
                row.appendChild(probCell);
                predictionsTable.appendChild(row);
            });
        } else {
             console.warn("Top predictions data is missing or not an array.");
        }
    } else {
         console.error("Top predictions table body not found.");
    }


    // Fill in word frequencies table
    const wordFreqTable = document.getElementById('word-frequencies')?.querySelector('tbody');
     if (wordFreqTable) {
        wordFreqTable.innerHTML = ''; // Clear previous results
        if (data.word_frequencies_top && typeof data.word_frequencies_top === 'object') {
            // Sort word frequencies by frequency (descending)
            const sortedWords = Object.entries(data.word_frequencies_top)
                                      .sort((a, b) => (b[1] ?? 0) - (a[1] ?? 0));
            const totalWords = sortedWords.reduce((sum, [_, freq]) => sum + (freq ?? 0), 0);

            // Show top 20 words (or fewer if less available)
            sortedWords.slice(0, 20).forEach(([word, freq]) => {
                const row = document.createElement('tr');
                const wordCell = document.createElement('td');
                wordCell.textContent = word;
                row.appendChild(wordCell);
                const freqCell = document.createElement('td');
                freqCell.textContent = freq ?? 0;
                row.appendChild(freqCell);
                const percentCell = document.createElement('td');
                percentCell.textContent = (totalWords > 0 && typeof freq === 'number')
                                          ? ((freq / totalWords) * 100).toFixed(2) + '%'
                                          : 'N/A';
                row.appendChild(percentCell);
                wordFreqTable.appendChild(row);
            });
        } else {
             console.warn("Word frequencies data is missing or not an object.");
        }
     } else {
         console.error("Word frequencies table body not found.");
     }

    // Show the results card
    showCard('results-card');
  }

});
