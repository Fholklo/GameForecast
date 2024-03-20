.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
run_api:
  uvicorn gameforecast.package.front_end.api_file:app --reload

#########
##MAIN LOCAL
#########
run_preprocess:
  python -c 'from package.main import preprocess; preprocess()'
run_train:
  python -c 'from package.main import train; train()'
#run_pred:
#  python -c 'from package.main import pred; pred()'
#run_evaluate:
#  python -c 'package.main import evaluate; evaluate()'
#run_all: run_preprocess run_train run_pred run_evaluate

#########
##API
#########
run_api_local:
  uvicorn package.api.api_file:app --host 0.0.0.0
#########
##DOCKER LOCAL
#########
build_container_local:
  docker build --tag=$GAR_IMAGE:dev .
run_container_local:
  docker run -it -e PORT=8000 -p 8000:8000 $GAR_IMAGE:dev
#########
##DOCKER DEPLOYMENT
#########
# Step 1 (1 time)
allow_docker_push:
  gcloud auth configure-docker $GCP_REGION-docker.pkg.dev
# Step 2 (1 time)
create_artifacts_repo:
  gcloud artifacts repositories create $ARTIFACTSREPO --repository-format=docker \
  --location=$GCP_REGION -- description="Repository for sotring images"
# Step 3
build_for_production:
  docker build -t $GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$ARTIFACTSREPO/$GAR_IMAGE:prod .
# Step 4
push_image_production:
  docker push $GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$GAR_IMAGE:prod
# Step 5
deploy_to_cloud_run:
  gcloud run deploy --image $GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$ARTIFACTSREPO/$GAR_IMAGE:prod\
   --memory $GAR_MEMORY --region $GCP_REGION
