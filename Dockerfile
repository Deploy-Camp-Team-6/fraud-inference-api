# --- Builder Stage ---
# This stage installs dependencies into a virtual environment.
FROM python:3.11-slim as builder

ENV POETRY_VERSION=1.8.2
ENV POETRY_HOME="/opt/poetry"
ENV POETRY_VENV="/opt/poetry-venv"
ENV POETRY_CACHE_DIR="/opt/poetry-cache"
# Ensure the Poetry executable installed inside the virtual environment is on PATH
ENV PATH="$POETRY_VENV/bin:$PATH"

# Install poetry
RUN python -m venv $POETRY_VENV \
    && $POETRY_VENV/bin/pip install -U pip setuptools \
    && $POETRY_VENV/bin/pip install poetry==${POETRY_VERSION}

# Set poetry configuration
RUN poetry config virtualenvs.in-project true

WORKDIR /app

# Copy dependency definition files
COPY pyproject.toml poetry.lock ./

# Install production dependencies
# Using --no-dev instead of --only main as it's more common in older versions
# that might be encountered, though the goal is the same.
RUN poetry install --no-dev --no-root --no-interaction

# Diagnostic step to verify uvicorn installation
RUN ls -al /app/.venv/bin


# --- Final Stage ---
# This stage creates the final, lean production image.
FROM python:3.11-slim as final

# Create a non-root user for security
RUN groupadd --system app && useradd --system --gid app app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV APP_HOME="/home/app"

WORKDIR $APP_HOME

# Copy virtual environment from builder stage
COPY --from=builder /app/.venv $APP_HOME/.venv

# Copy application code
COPY ./app $APP_HOME/app

# Activate virtual environment
ENV PATH="$APP_HOME/.venv/bin:$PATH"

# Change ownership and switch to non-root user
RUN chown -R app:app $APP_HOME
USER app

# Expose port and run the application
EXPOSE 80
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80", "--workers", "${WORKERS:-2}"]
