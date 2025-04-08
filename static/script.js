// DOM Elements
const form = document.getElementById('model-form');
const fileInput = document.getElementById('input-file');
const fileLabel = document.getElementById('fileInput-label');
const algorithmSelect = document.getElementById('dropdown_algo');
const variableSelect = document.getElementById('variable');
const modelSelect = document.getElementById('model');
const modelSelection = document.querySelector('.model-selection');
const submitBtn = document.getElementById('submit-btn');
const spinner = document.getElementById('spinner');
const codePart = document.getElementById('code-part');

// Event Listeners
fileInput.addEventListener('change', handleFileSelect);
fileInput.addEventListener('dragover', handleDragOver);
fileInput.addEventListener('drop', handleFileDrop);
algorithmSelect.addEventListener('change', handleAlgorithmChange);
variableSelect.addEventListener('change', handleVariableChange);
modelSelect.addEventListener('change', handleModelChange);
form.addEventListener('submit', handleSubmit);

// File Upload Handlers
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) validateFile(file);
}

function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    fileLabel.classList.add('drag-over');
}

function handleFileDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    fileLabel.classList.remove('drag-over');
    
    const file = e.dataTransfer.files[0];
    if (file) validateFile(file);
}

function validateFile(file) {
    if (!file.name.endsWith('.csv')) {
        showError('Please upload a CSV file');
        fileInput.value = '';
        return;
    }
    
    fileLabel.querySelector('.file-text').textContent = file.name;
    fileLabel.classList.add('file-selected');
    retrieveColumnNames();
    enableAlgorithmSelect();
}

// Form Handlers
function handleAlgorithmChange() {
    const selectedAlgorithm = algorithmSelect.value;
    variableSelect.disabled = !selectedAlgorithm;
    if (!selectedAlgorithm) resetSelects();
}

function handleVariableChange() {
    const selectedVariable = variableSelect.value;
    if (!selectedVariable) resetModelSelect();
}

function handleModelChange() {
    const selectedModel = modelSelect.value;
    if (selectedModel) generateAndDisplayCode();
}

async function handleSubmit(e) {
    e.preventDefault();
    
    if (!validateForm()) return;

    submitBtn.disabled = true;
    spinner.hidden = false;
    
    try {
        await simulateApiCall();
    } catch (error) {
        showError('Failed to generate model. Please try again.');
        console.error('API Error:', error);
    } finally {
        submitBtn.disabled = false;
    }
}

// Form Validation
function validateForm() {
    if (!fileInput.files[0]) {
        showError('Please upload a CSV file');
        return false;
    }
    
    if (!algorithmSelect.value) {
        showError('Please select an algorithm');
        return false;
    }
    
    if (!variableSelect.value) {
        showError('Please select a variable');
        return false;
    }
    
    return true;
}

// UI Updates
function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.textContent = message;
    
    form.insertBefore(errorDiv, submitBtn);
    setTimeout(() => errorDiv.remove(), 3000);
}

function enableAlgorithmSelect() {
    algorithmSelect.disabled = false;
}

function resetSelects() {
    variableSelect.innerHTML = '<option value="" disabled selected>Select a variable</option>';
    modelSelection.hidden = true;
    resetModelSelect();
}

function resetModelSelect() {
    modelSelect.innerHTML = '<option value="" disabled selected>Choose a model</option>';
}

function enablePlots() {
    document.querySelectorAll('.plot').forEach(plot => plot.classList.remove('loading'));
}

function generateAndDisplayCode() {
    codePart.textContent = generateCode();
}

// API Calls
async function simulateApiCall() {
    const data = new FormData();
    data.append('file', fileInput.files[0]);
    data.append('target', variableSelect.value);
    data.append('algorithm', algorithmSelect.value);

    try {
        const response = await fetch('/generate', {
            method: 'POST',
            body: data,
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const models = await response.json();
        console.log('Response:', models);

        models.forEach(model => {
            const modelOption = document.createElement('option');
            modelOption.value = model.model_name;
            modelOption.textContent = `${model.model_name} (${model.metric_value.toFixed(2)})`;
            modelSelect.appendChild(modelOption);
        });

        modelSelection.hidden = false;
        spinner.style.display = 'none';

        modelSelect.addEventListener('change', () => {
            const selectedModel = models.find(m => m.model_name === modelSelect.value);
            if (selectedModel) {
                codePart.textContent = selectedModel.code_snippets;
                const plotContent = document.querySelector('.plot-content');
                plotContent.innerHTML = `<img src="data:image/png;base64,${selectedModel.image}" alt="Model performance plot">`;
                enablePlots();
            }
        });
    } catch (error) {
        spinner.style.display = 'none';
        throw new Error(`API call failed: ${error.message}`);
    }
}

async function retrieveColumnNames() {
    const data = new FormData();
    data.append('file', fileInput.files[0]);

    try {
        const response = await fetch('/retrieve_column_names', {
            method: 'POST',
            body: data,
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const { column_names } = await response.json();
        console.log('Column names:', column_names);

        variableSelect.innerHTML = '<option value="" disabled selected>Select a variable</option>';
        column_names.forEach(column => {
            const option = document.createElement('option');
            option.value = column.replace(/\s+/g, '_');
            option.textContent = column;
            variableSelect.appendChild(option);
        });
    } catch (error) {
        console.error('Error retrieving column names:', error);
        showError('Failed to retrieve column names. Please try again.');
    }
}