FROM python:3.12.2-slim

# Set the working directory inside the container
WORKDIR /movie_recommendation_system

# Copy the requirements file and install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . . 

# Expose the port for Streamlit
EXPOSE 8501

# Command to run your Streamlit app
CMD ["streamlit", "run", "recommendation_app.py"]