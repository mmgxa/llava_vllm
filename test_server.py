import base64
import requests

with open("img.jpg", "rb") as f:
    image_file = f.read()
    encoded = base64.b64encode(image_file).decode("utf-8")

data = {
    "prompt": "<image>\n say something",
    "max_tokens": 256,
    "images": [
        encoded
    ],  # str or a list of str. can be **url** or **base64.**  must match the number of '<image>'
}

res = requests.post(f"http://localhost:8000/generate", json=data)
print(res.text)

