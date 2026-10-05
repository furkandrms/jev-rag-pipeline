# Cloud Run images are billed by CPU-time while serving requests, not by
# image size -- but a smaller image still means faster cold starts (less to
# pull before the first request can be served), which matters more here
# than build-time convenience.
FROM python:3.12-slim

WORKDIR /app

# git: pip needs it to clone rag-guard's git+https dependency (see
# requirements.txt). build-essential: a couple of chromadb's transitive
# dependencies (e.g. hnswlib) ship as source distributions on some
# platforms and need a C compiler to build their wheel.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run sets $PORT at runtime (defaults to 8080 if unset, e.g. for
# local `docker run`); the app must listen on whatever it provides.
ENV PORT=8080
EXPOSE 8080

CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT}
