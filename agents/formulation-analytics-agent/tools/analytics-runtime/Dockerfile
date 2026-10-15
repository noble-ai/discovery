FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app /input /output /state

RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    scipy==1.14.1 \
    pandas==2.2.3 \
    pyarrow==18.1.0 \
    scikit-learn==1.5.2 \
    statsmodels==0.14.4 \
    plotly==5.24.1

COPY analytics_utils.py /app/analytics_utils.py

WORKDIR /app
ENV PYTHONPATH="/app:${PYTHONPATH}"
ENV PYTHONUNBUFFERED=1

CMD ["python3"]
