FROM python:3.11.12-slim

# Copy local code to the container image.
ENV PYTHONUNBUFFERED=True
WORKDIR /app

# Install Python dependencies and Gunicorn
COPY poetry.lock pyproject.toml /app/

# Install Poetry and configure it not to create virtual environments.
RUN pip3 install --no-cache-dir poetry \
    && poetry config virtualenvs.create false

# Install the dependencies specified in the pyproject.toml file.
RUN poetry install --no-interaction --no-ansi --no-root

# Copy the rest of the codebase into the image
COPY ./python /app/

# Document the port on which the app will run.
EXPOSE 8050

# Start the app using Gunicorn with one worker and 8 threads.
CMD ["gunicorn", "--bind", ":8050", "--log-level", "info", "--workers", "1", "--threads", "8", "--timeout", "0", "app:server"]
