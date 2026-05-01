document.getElementById('myForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const formData = new FormData(e.target);
    const payload = {
        step_size: formData.get("step_size"),
        dimensionality: formData.get("dimensionality"),
        start_position: formData.get("start_position"),
        n_steps: formData.get("n_steps"),
    };

    try {
        const response = await fetch('/api/p2/state', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const result = await response.json();
        console.log('Success:', result);
    } catch (error) {
        console.error('Error:', error);
    }
});