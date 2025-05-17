FROM python:3.10

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# Download nltk punkt tokenizer data
RUN python -c "import nltk; nltk.download('punkt')"

# Create cache folder and give write permission to all users
RUN mkdir -p /tmp/huggingface/transformers_cache && chmod -R 777 /tmp/huggingface

ENV HF_HOME=/tmp/huggingface
ENV TRANSFORMERS_CACHE=/tmp/huggingface/transformers_cache

COPY . .

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
