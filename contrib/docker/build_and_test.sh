#/bin/bash

docker run --rm -t -i -v $PWD:/io turbodbc_base /io/contrib/docker/build_and_test_in_docker.sh
