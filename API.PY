import requests

API_KEY = "tYfZrX4vNiT40941w2U8z5zPUBETUqw2"  # Replace with your actual API key
MATERIALS_PROJECT_URL = "https://materialsproject.org/rest/v2/materials/mp-149/vasp?API_KEY=" + API_KEY

response = requests.get(MATERIALS_PROJECT_URL)

if response.status_code == 200:
    data = response.json()
    print(data)
else:
    print("Error fetching data")
