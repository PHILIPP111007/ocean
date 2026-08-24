import requests
import os

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
file_path = "tinyshakespeare.txt"

if not os.path.exists(file_path):
    print("Загружаем текст Шекспира...")
    response = requests.get(url)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(response.text)
    print("Файл сохранён.")

with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

print(f"Загружено {len(text):,} символов.")