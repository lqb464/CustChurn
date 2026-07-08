.PHONY: install validate train predict test api frontend compose

install:
	python -m pip install -e ".[api,frontend,notebooks,dev]"

validate:
	python scripts/validate_data.py

train:
	python scripts/train.py

predict:
	python scripts/predict.py

test:
	python -m pytest

api:
	uvicorn backend.main:app --reload

frontend:
	streamlit run frontend/app.py

compose:
	docker compose up --build
