FROM tensorflow/tensorflow:2.10.0

# First, pip install dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Then only, install taxifare!
COPY package package
COPY setup.py setup.py
RUN pip install -e .

COPY Makefile Makefile
RUN make reset_local_files

CMD uvicorn package.api_file:app --host 0.0.0.0 --port $PORT
