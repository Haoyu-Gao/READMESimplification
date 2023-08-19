FROM python:3
RUN mkdir Harvester
ADD Harvester ./Harvester
WORKDIR ./Harvester
RUN echo $(ls)
RUN pip install -r requirements.txt
CMD python3 harvester.py --db=github_raw_md --id=0