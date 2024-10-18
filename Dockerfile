# Используем базовый образ с Python
FROM python:3.9-slim

# Устанавливаем рабочую директорию в контейнере
WORKDIR /app

# Устанавливаем poetry
RUN pip install poetry

# Копируем файлы проекта
COPY pyproject.toml poetry.lock ./

# Устанавливаем зависимости
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

# Копируем весь код в контейнер
COPY . .

# Скачиваем модель spacy
RUN python -m spacy download ru_core_news_sm

# Указываем переменные окружения
ENV YANDEX_GPT_IAM_TOKEN=your_yandex_gpt_iam_token
ENV YANDEX_GPT_FOLDER_ID=your_yandex_gpt_folder_id
ENV YANDEX_MODEL_URI=your_yandex_model_uri

# Запускаем программу
CMD ["python", "your_script.py"]