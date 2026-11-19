FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python -m factory_copilot.data.build && python -m factory_copilot.analytics.train
EXPOSE 8000 8501
# API by default; `docker run ... streamlit run dashboard/app.py --server.address 0.0.0.0` for the dashboard
CMD ["uvicorn", "factory_copilot.api:app", "--host", "0.0.0.0", "--port", "8000"]
