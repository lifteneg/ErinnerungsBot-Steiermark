.PHONY: setup run compose up qdrant down eval eval-llm

QUESTIONS ?= data/eval_questions.txt

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

run:
	streamlit run app.py

compose: up

up:
	docker compose up --build

qdrant:
	docker compose up -d qdrant

eval:
	@if [ -f .env ]; then export $$(grep -v '^#' .env | xargs); fi; \
	python eval.py --questions $(QUESTIONS)

eval-llm:
	@if [ -f .env ]; then export $$(grep -v '^#' .env | xargs); fi; \
	python eval.py --questions $(QUESTIONS) --llm

down:
	docker compose down
