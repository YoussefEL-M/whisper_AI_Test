FROM python:3.11-slim

# Install websockets library
RUN pip install websockets

# Set working directory
WORKDIR /app

# Copy the WebSocket server
COPY simple_ws_server.py /app/

# Expose port
EXPOSE 5001

# Run the server
CMD ["python", "simple_ws_server.py"]