function predict() {
    let temperature = document.getElementById("temperature").value;
    let pressure = document.getElementById("pressure").value;
    let catalyst = document.getElementById("catalyst").value;
    let structure = document.getElementById("structure").value;

    fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            "Temperature": temperature,
            "Pressure": pressure,
            "Catalyst": catalyst,
            "Structure": structure
        })
    })
        .then(response => response.json())
        .then(data => {
            document.getElementById("result").innerText = "Estimated Yield: " + data["Reaction Yield"] + "%";
        })
        .catch(error => console.error("Error:", error));
}
