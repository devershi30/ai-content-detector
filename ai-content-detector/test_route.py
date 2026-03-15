import requests

# Test the analyze_sentences endpoint
url = "http://localhost:5000/analyze_sentences"
data = {"text": "This is a test sentence. This is another sentence for testing AI detection."}

try:
    response = requests.post(url, data=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
