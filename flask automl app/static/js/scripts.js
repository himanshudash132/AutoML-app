document.addEventListener('DOMContentLoaded', function () {
    const rangeInput = document.querySelector('input[name="split_size"]');
    const splitValue = document.getElementById('splitValue');
    
    rangeInput.addEventListener('input', function () {
        splitValue.textContent = `Training: ${rangeInput.value}%`;
    });
});
