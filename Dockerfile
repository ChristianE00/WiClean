# Use an official Python runtime as a parent image
FROM python:3.8




# Set the working directory in the container
WORKDIR /app

# Copy Jupyter Notebook files and data directory into the container
COPY wiclean.ipynb /app/

# Install required Python packages
RUN pip install jupyter pandas scikit-learn SPARQLWrapper 

# Expose the port for Jupyter Notebook
EXPOSE 8888

CMD ["jupyter", "notebook", "--ip='*'", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/app"]
# Start Jupyter Notebook when the container launches
#CMD ["jupyter", "notebook", "--ip='*'", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/app/notebooks", "--ServerApp.iopub_data_rate_limit=1000000000"]
