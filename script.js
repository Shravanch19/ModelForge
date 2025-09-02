// ===== DOM Elements =====
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

// ===== Event Listeners =====
fileInput.addEventListener('change', e => handleFile(e.target.files[0]));
fileInput.addEventListener('dragover', e => toggleDrag(e, true));
fileInput.addEventListener('drop', e => {
    e.preventDefault();
    toggleDrag(e, false);
    handleFile(e.dataTransfer.files[0]);
});
algorithmSelect.addEventListener('change', handleAlgorithmChange);
variableSelect.addEventListener('change', () => resetModelSelect());
modelSelect.addEventListener('change', () => generateAndDisplayCode());
form.addEventListener('submit', handleSubmit);

// ===== File Handling =====
function handleFile(file) {
    if (!file) return;
    if (!file.name.endsWith('.csv')) return showError('Please upload a CSV file');

    fileLabel.querySelector('.file-text').textContent = file.name;
    fileLabel.classList.add('file-selected');
    algorithmSelect.disabled = false;

    retrieveColumnNames();
}

function toggleDrag(e, active) {
    e.preventDefault();
    e.stopPropagation();
    fileLabel.classList.toggle('drag-over', active);
}

// ===== Form Handling =====
function handleAlgorithmChange() {
    variableSelect.disabled = !algorithmSelect.value;
    if (!algorithmSelect.value) resetSelects();
}

async function handleSubmit(e) {
    e.preventDefault();
    if (!validateForm()) return;

    submitBtn.disabled = true;
    spinner.hidden = false;

    try {
        await simulateApiCall();
    } catch (err) {
        showError('Failed to generate model. Please try again.');
        console.error(err);
    } finally {
        submitBtn.disabled = false;
    }
}

// ===== Validation =====
function validateForm() {
    if (!fileInput.files[0]) return showError('Please upload a CSV file'), false;
    if (!algorithmSelect.value) return showError('Please select an algorithm'), false;
    if (!variableSelect.value) return showError('Please select a variable'), false;
    return true;
}

// ===== UI Helpers =====
function showError(message) {
    const error = document.createElement('div');
    error.className = 'error-message';
    error.textContent = message;
    form.insertBefore(error, submitBtn);
    setTimeout(() => error.remove(), 3000);
}

function resetSelects() {
    variableSelect.innerHTML = `<option disabled selected>Select a variable</option>`;
    resetModelSelect();
    modelSelection.hidden = true;
}

function resetModelSelect() {
    modelSelect.innerHTML = `<option disabled selected>Choose a model</option>`;
}

function enablePlots() {
    document.querySelectorAll('.plot').forEach(plot => plot.classList.remove('loading'));
}

function generateAndDisplayCode() {
    codePart.textContent = modelSelect.value ? generateCode() : '';
}

// ===== API Calls =====
async function simulateApiCall() {
    const data = new FormData();
    data.append('file', fileInput.files[0]);
    data.append('target', variableSelect.value);
    data.append('algorithm', algorithmSelect.value);

    console.log(variableSelect.value);
    console.log(algorithmSelect.value);

    const res = await fetch('http://127.0.0.1:5000/generate', { method: 'POST', body: data });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const models = await res.json();
    updateModelOptions(models);
}

function updateModelOptions(models) {
    modelSelect.innerHTML = `<option disabled selected>Choose a model</option>`;
    models.forEach(({ model_name, metric_value }) => {
        const opt = document.createElement('option');
        opt.value = model_name;
        opt.textContent = `${model_name} (${metric_value.toFixed(2)})`;
        modelSelect.appendChild(opt);
    });

    modelSelection.hidden = false;
    spinner.style.display = 'none';

    modelSelect.addEventListener('change', () => {
        const selected = models.find(m => m.model_name === modelSelect.value);
        if (!selected) return;
        codePart.textContent = selected.code_snippets;
        document.querySelector('.plot-content').innerHTML =
            `<img src="data:image/png;base64,${selected.image}" alt="Model performance plot">`;
        enablePlots();
    });
}

async function retrieveColumnNames() {
    const data = new FormData();
    data.append('file', fileInput.files[0]);

    try {
        const res = await fetch('http://127.0.0.1:5000/retrieve_column_names', { method: 'POST', body: data });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);

        const { column_names } = await res.json();
        variableSelect.innerHTML = `<option disabled selected>Select a variable</option>`;
        column_names.forEach(col => {
            const opt = document.createElement('option');
            opt.value = col.replace(/\s+/g, '_');
            opt.textContent = col;
            variableSelect.appendChild(opt);
        });
        variableSelect.disabled = false;
    } catch (err) {
        console.error(err);
        showError('Failed to retrieve column names.');
    }
}
