# Use the official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy app code
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Streamlit/Flask will run on
EXPOSE 5000

# Start the Flask app using gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:5000", "CasedemoAPI:app"]
