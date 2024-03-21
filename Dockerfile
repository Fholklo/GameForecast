FROM tensorflow/tensorflow:2.15.0

# First, pip install dependencies
COPY requirements_prod.txt requirements_prod.txt
RUN pip install -r requirements_prod.txt

# Then only, install taxifare!
COPY package package
COPY setup.py setup.py
COPY model_rating_20240321-102424.h5 model_rating_20240321-102424.h5
RUN pip install -e .

CMD uvicorn package.api.api_file:app --host 0.0.0.0 --port $PORT
