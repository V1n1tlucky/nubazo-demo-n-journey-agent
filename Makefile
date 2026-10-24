.DEFAULT_GOAL := help
.PHONY: help install browser dev stop backend frontend model-mlx lint format typecheck test check eval clean

BE := uv run --directory backend
BACKEND_PORT ?= 8000  # keep in sync with JA_PORT
FRONTEND_PORT ?= 5173

help: ## Show targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Create backend/.venv and install all deps (incl. dev + browser extra)
	uv sync --directory backend --all-extras

browser: ## Download headless Chromium for JS-rendered pages + XHR payload capture
	$(BE) playwright install chromium

dev: ## Run backend (:8000) and frontend (:5173) together; Ctrl-C stops both
	@for p in $(BACKEND_PORT) $(FRONTEND_PORT); do \
	  if lsof -nP -iTCP:$$p -sTCP:LISTEN >/dev/null 2>&1; then \
	    echo "Port $$p is already in use (another 'make dev' running?). Free it with: make stop"; exit 1; \
	  fi; \
	done
	@trap 'kill 0' INT TERM EXIT; $(MAKE) --no-print-directory backend & $(MAKE) --no-print-directory frontend & wait

stop: ## Stop whatever is listening on the backend and frontend ports
	@for p in $(BACKEND_PORT) $(FRONTEND_PORT); do \
	  pids=$$(lsof -tiTCP:$$p -sTCP:LISTEN); \
	  if [ -n "$$pids" ]; then kill $$pids && echo "stopped :$$p (pid $$pids)"; else echo ":$$p already free"; fi; \
	done

backend: ## Start the API on http://127.0.0.1:8000
	$(BE) journey-agent

frontend: ## Serve the console on http://127.0.0.1:$(FRONTEND_PORT) (static, no build step)
	python3 scripts/frontend_server.py $(FRONTEND_PORT)

model-mlx: ## Serve a 7B model with MLX (alternative to Ollama), port 8080
	uvx --python 3.12 --from mlx-lm mlx_lm.server --model mlx-community/Qwen2.5-7B-Instruct-4bit --port 8080

lint: ## Ruff lint
	$(BE) ruff check src tests

format: ## Ruff format + autofix
	$(BE) ruff format src tests && $(BE) ruff check --fix src tests

typecheck: ## mypy --strict
	$(BE) mypy src

test: ## Unit + integration tests (no model downloads needed)
	$(BE) pytest

check: lint typecheck test ## Everything CI runs
	$(BE) ruff format --check src tests

eval: ## Calibrate the relevance gate (EVAL=evals/x.json, PAGE_ID=<id> to use a stored index)
	$(BE) python -m journey_agent.evals $(or $(EVAL),evals/nubazo.json) $(if $(PAGE_ID),--page-id $(PAGE_ID))

clean: ## Delete runtime data (SQLite, FAISS indexes, snapshots)
	rm -rf backend/data
