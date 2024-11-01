// async function fetchMessage() {
//     const response = await fetch('/get_message');
//     const data = await response.json();
//     document.getElementById('PRT').innerText = data.message;
// }

let isFormReady = false;

const fileInput = document.getElementById('fileInput');
const selectElement1 = document.getElementById('target-menu');
const selectElement2 = document.getElementById('Model-type');
const submitBtn = document.getElementById('submit-main');
const messageDisplay = document.getElementById('PRT');




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

        if(file.name.length > 15){
            document.getElementById('fileInput-label').innerText = file.name.slice(0, 11) + "...";
        }
        else{
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
        fetchMessage();
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

async function fetchMessage() {
    messageDisplay.innerText = "Code generated for the dataset!!";
}
