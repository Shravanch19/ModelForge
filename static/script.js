let datas;
async function fetchcolumn(file) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('/get_columns', {
        method: 'POST',
        body: formData
    });
    const data = await response.json();
    Columns_set(data.columns);
}



let isFormReady = false;

const fileInput = document.getElementById('input-file');
const selectElement1 = document.getElementById('variable');
const selectElement2 = document.getElementById('dropdown_algo');
const submitBtn = document.getElementById('submit-btn');



function validateForm(event) {
    if (selectElement1.value === "" || selectElement2.value === "" || !isFormReady) {
        alert("Please select valid options.");
        event.preventDefault();
        return false
    }
    else {
        return true
    }
}
function handleFileSelection() {
    const file = fileInput.files[0];

    if (file && file.name.endsWith('.csv')) {
        console.log('CSV file selected: ', file.name);
        isFormReady = true;

        // then disable this input and call function abc
        fileInput.disabled = true;
        fetchcolumn(file);

        if (file.name.length > 15) {
            document.getElementById('fileInput-label').innerText = file.name.slice(0, 11) + "...";
        }
        else {
            document.getElementById('fileInput-label').innerText = file.name;
        }

    } else {
        alert("Please select a valid file.");
    }
}


fileInput.addEventListener('change', handleFileSelection);


submitBtn.addEventListener('click', function (event) {
    selectElement1.setAttribute('required', '');
    selectElement2.setAttribute('required', '');

    if (!validateForm(event)) {
        event.preventDefault();
    }
    else {
        document.getElementById("spinner").style.display = "block";
        document.getElementById("code-part").style.display = "none";
        const file = fileInput.files[0];
        const variable = document.getElementById("variable").value;
        submitBtn.disabled = true;
        fetchMessage(file, variable);
    }
});


selectElement1.addEventListener('change', function (event) {
    if (!isFormReady) {
        alert("Please select a file first.");
        event.preventDefault();
    }
});

selectElement2.addEventListener('change', function (event) {
    if (!isFormReady) {
        alert("Please select a file first.");
        event.preventDefault();
    }
});

async function fetchMessage(file, targetVariable) {

    const formData = new FormData();
    formData.append('file', file);
    formData.append("target_variable", targetVariable);

    const response = await fetch('/get_message', {
        method: 'POST',
        body: formData
    });

    const data = await response.json();
    datas = data;

    const img = document.createElement("img");
    img.src = `data:image/png;base64,${data.heat}`;

    document.getElementById('Heat_para').style.display = "none";
    document.getElementById('Heat_plot').appendChild(img);

    document.getElementById('code-part').innerText = data.code_snippets[0].code;
    document.getElementById("R2_para").style.display = "none";

    const R2_Img = document.createElement("img");
    R2_Img.src = `data:image/png;base64,${data.code_snippets[0].score_plot}`;

    document.getElementById('R2_plot').appendChild(R2_Img);

    document.getElementById("spinner").style.display = "none";
    document.getElementById("code-part").style.display = "block";

    document.querySelectorAll('.plot').forEach(element => {
        element.classList.remove('plot');
        element.classList.add("Plot");
    });

    Model_set = document.getElementById("model");
    Model_set.disabled = false;
    for (let i = 0; i < data.code_snippets.length; i++) {
        let option = document.createElement("option");
        option.value = data.code_snippets[i].model_name;
        option.text = data.code_snippets[i].model_name + " (R2 Score: " + data.code_snippets[i].accuracy + "%)";
        Model_set.appendChild(option);
    }
}

async function Columns_set(data) {
    for (let i = 0; i < data.length; i++) {
        let option = document.createElement("option");
        option.value = data[i];
        option.text = data[i];
        selectElement1.appendChild(option);
        console.log(data[i]);
    }
}

document.getElementById("model").addEventListener("change", function () {
    for (let i = 0; i < datas.code_snippets.length; i++) {
        if (datas.code_snippets[i].model_name === this.value) {
            document.getElementById("code-part").innerText = datas.code_snippets[i].code;
            document.getElementById('R2_plot').removeChild(document.getElementById('R2_plot').lastElementChild);
            const R2_Img = document.createElement("img");
            R2_Img.src = `data:image/png;base64,${datas.code_snippets[i].score_plot}`;
            document.getElementById('R2_plot').appendChild(R2_Img);
        }
    }
})