.PHONY: install install-full dev test lint run digest docker-build docker-run

install:            ## 安装核心依赖（轻量，无本地嵌入模型）
	pip install -r requirements.txt

install-full:       ## 安装全部依赖（含 fastembed 本地语义嵌入）
	pip install -r requirements-embed.txt

dev:                ## 开发环境（含 pytest/ruff）
	pip install -r requirements-dev.txt

test:               ## 运行测试（全离线）
	pytest -q

lint:               ## 代码检查
	ruff check .

run:                ## 启动 Web 服务
	python app.py

digest:             ## 生成本地论文日报
	python scripts/daily_digest.py

docker-build:
	docker build -t papermind .

docker-run:
	docker run -p 5000:5000 -v $(PWD)/data:/app/data papermind
