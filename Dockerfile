
FROM us-docker.pkg.dev/vertex-ai/training/ray-gpu.2-9.py310:latest

# Install training libraries.
ENV PIP_ROOT_USER_ACTION=ignore
COPY requirements.txt .
COPY ./models/ models/
COPY attacks.py .
COPY client.py .
COPY evaluation.py .
COPY main.py .
COPY perplexity.py .
COPY strategy.py .
COPY tensorword.py .
ADD ./parser/ data/
RUN pip install -r requirements.txt
