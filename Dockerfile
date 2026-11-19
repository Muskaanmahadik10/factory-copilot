FROM python:3.15-slim@sha256:65a65fc7b30b06cb6df4930fd4650697cb23f4c2698ab81fc87e4b031bc36928
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python -m factory_copilot.data.build && python -m factory_copilot.analytics.train
EXPOSE 8000 8501
# API by default; `docker run ... streamlit run dashboard/app.py --server.address 0.0.0.0` for the dashboard
CMD ["uvicorn", "factory_copilot.api:app", "--host", "0.0.0.0", "--port", "8000"]
