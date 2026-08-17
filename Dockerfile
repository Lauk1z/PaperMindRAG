FROM python:3.11-slim

WORKDIR /app

# 先装依赖（利用镜像层缓存）
COPY requirements-embed.txt requirements.txt ./
RUN pip install --no-cache-dir -r requirements-embed.txt gunicorn

COPY . .

# 数据卷：论文与索引持久化
VOLUME ["/app/data"]
EXPOSE 5000

# gunicorn 生产级 WSGI（Flask 内置 server 仅适合开发）
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "--access-logfile", "-", "app:app"]
