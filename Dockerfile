# Serves the Gradio demo. Build after training:
#   docker build -t ucf101-vit .
#   docker run --gpus all -p 7860:7860 ucf101-vit
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY data/ data/
COPY models/ models/
COPY utils.py inference.py app.py ./
# Only the best checkpoint(s) are copied in (see .dockerignore).
COPY checkpoints/ checkpoints/

ENV HOST=0.0.0.0 PORT=7860
EXPOSE 7860
CMD ["python", "app.py", "--examples_dir", "/nonexistent"]
