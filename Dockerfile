# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Create a non-root user with UID 1000 (required by Hugging Face Spaces)
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Install system dependencies (including libxrender, libxext for RDKit headless if needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container, setting ownership to our new user
COPY --chown=user requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container, setting ownership to the user
COPY --chown=user . .

# Switch to the non-root user for security and to comply with HF limits
USER user

# Make port 7860 available to the world outside this container
EXPOSE 7860

# Define environment variables
ENV FLASK_APP=app.py \
    PORT=7860 \
    MAX_CONCURRENT_ANALYSIS_JOBS=1 \
    BO_TOTAL_JOB_TIMEOUT_SEC=300 \
    BO_CANCEL_GRACE_SEC=3 \
    BO_NEHVI_MAX_OBJECTIVES=3 \
    BO_NEHVI_ALPHA=0.01 \
    BO_MC_SAMPLES_CPU=64 \
    BO_NUM_RESTARTS_CPU=8 \
    BO_RAW_SAMPLES_CPU=128 \
    BO_MODEL_FIT_MAXITER=75 \
    BO_MODEL_FIT_MAXLS=50 \
    BO_MODEL_FIT_RETRIES=1 \
    BO_MODEL_FIT_PHASE_TIMEOUT_SEC=60 \
    BO_ACQ_OPT_MAXITER=75 \
    BO_CANDIDATE_TIMEOUT_SEC=20 \
    BO_FEASIBLE_IC_MAX_DRAWS=8192 \
    BO_FEASIBLE_IC_CHUNK_SIZE=512 \
    BO_MAX_BATCH_SIZE=20 \
    BO_WORKLOAD_REVIEW_SCORE=0.55 \
    BO_WORKLOAD_UNSAFE_SCORE=1.0 \
    BO_REPRESENTATIVE_TARGET_SCORE=0.95 \
    BO_REPRESENTATIVE_MIN_ROWS=20 \
    BO_REPRESENTATIVE_MAX_ROWS=250 \
    BO_SUBSET_TARGET_FIDELITY=90 \
    BO_SUBSET_SELECTION_ATTEMPTS=3 \
    BO_SUBSET_SELECTION_TIMEOUT_SEC=12 \
    BO_SAFE_MEMORY_MB=6144 \
    BO_TORCH_NUM_THREADS=1 \
    BO_TORCH_INTEROP_THREADS=1 \
    TRANSFER_GP_ATTEMPT_TIMEOUT_SECONDS=75 \
    TRANSFER_JOB_HARD_TIMEOUT_SECONDS=360 \
    ANALYSIS_JOB_CANCEL_GRACE_SECONDS=30 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# Run the application with gunicorn for production reliability
CMD ["gunicorn", "--workers", "1", "--threads", "8", "--bind", "0.0.0.0:7860", "--timeout", "120", "app:app"]
