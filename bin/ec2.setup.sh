#!/bin/bash
# Run from root of project

## Docker
./bin/setup_docker.sh

# get the code
git clone git@github.com:parallellm/parallellm-qa.git
cd parallellm-qa


# TODO: set up env vars in .env file
less .env
# TODO: set up logins.yaml.env file
less config/secret/logins.yaml.env

# build and run the docker image/container
docker build -f docker/Dockerfile -t parallellm-qa .
docker run -d --env-file .env -v $(pwd)/config/secret:/app/config/secret:ro -v $(pwd)/artefacts:/app/artefacts parallellm-qa:latest


