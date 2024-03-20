FROM tensorflow/tensorflow:2.10.0

# First, pip install dependencies
COPY requirements_prod.txt requirements.txt
RUN pip install -r requirements.txt

# Then only, install taxifare!
COPY package package
COPY setup.py setup.py
COPY model_rating_20240320-170248.h5 model_rating_20240320-170248.h5
RUN pip install .

CMD uvicorn package.api.api_file:app --host 0.0.0.0 --port $PORT
