FROM python:3.11-slim
WORKDIR /app
RUN pip install uv
COPY pyproject.toml uv.lock ./
RUN uv pip install --system .
COPY research_server.py .
EXPOSE 8001
CMD ["uv", "run", "research_server.py"]