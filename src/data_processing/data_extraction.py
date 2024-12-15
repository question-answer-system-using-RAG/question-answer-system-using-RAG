import requests
import markdown
from bs4 import BeautifulSoup
import pandas as pd
import google.generativeai as genai

def extract_text_from_md(url):
    # Загрузка содержимого Markdown с URL
    response = requests.get(url)
    response.raise_for_status()  # Проверка на наличие ошибок при загрузке

    # Конвертация содержимого Markdown в HTML
    html_content = markdown.markdown(response.text)

    # Использование BeautifulSoup для извлечения текста
    soup = BeautifulSoup(html_content, features="html.parser")
    text = soup.get_text()

    # Удаление лишних пробелов и переносов строк
    cleaned_text = ' '.join(text.split())

    return cleaned_text
