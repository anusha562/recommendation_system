FROM python:3.12.2-slim

WORKDIR /movie_recommendation_system

COPY initial_req.txt .

RUN pip install --no-cache-dir -r initial_req.txt

COPY . . 

# Expose port if you plan to make the app accessible via HTTP (optional)
EXPOSE 8000

# Command to run your script
CMD ["sh" , "-c", "dvc pull && streamlit run app.py"]
