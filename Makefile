PY ?= python

.PHONY: data train test eval eval-real api dashboard mcp all

data:
	$(PY) -m factory_copilot.data.build

train:
	$(PY) -m factory_copilot.analytics.train

test:
	$(PY) -m pytest -q

eval:
	$(PY) -m evaluation.cases
	$(PY) -m evaluation.run --provider mock

eval-real:
	$(PY) -m evaluation.run --provider env --limit 10

api:
	uvicorn factory_copilot.api:app --reload

dashboard:
	streamlit run dashboard/app.py

mcp:
	$(PY) -m factory_copilot.mcp_server

all: data train test eval
