# Runs the HTTP API. The harness itself needs no key; set ANTHROPIC_API_KEY at
# run time only if you want the LLM judge instead of the rubric judge.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Fail the image build if the suite is red. Same gate as CI, one layer earlier.
RUN python -m pytest -q

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
