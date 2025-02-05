# espnet-text-to-speech-xla-testing
This light-weight repo contains example configuration and test script content 
for the espnet-text-to-speech-xla project.

The `app` folder has a small set of python scripts to system test the environment 
setup, and to perform some representative text-to-speech (TTS) inference such that:
1. You can determine that the TTS inference is generating valid outputs
2. You can get a sense of the runtime performance of the implementation

The `docker` directory has docker files for establishing a runtime environment
for the app scipts. The docker files are for supporting different versions
of Pytorch (2.1 ... 2.5).

If have tested these scripts and configuration 
on [DCV](https://aws.amazon.com/hpc/dcv/) configured AWS EC2 
[g4dn.xlarge](https://aws.amazon.com/ec2/instance-types/g4/) instance 
type. This has a single GPU. I used the following Github repository to guide the
setup of the DCV EC2 node: 
[aws-deep-learning-ami-ubuntu-dcv-desktop](https://github.com/aws-samples/aws-deep-learning-ami-ubuntu-dcv-desktop). 


For launching the docker environment, the following may prove useful:

```
docker build -t dev/espnet-cuda-xla-25 . 

docker run -t -d -v /home/ubuntu/dev/app:/app --shm-size=16g --net=host --gpus all dev/espnet-cuda-xla-25 sleep infinity

docker_key="KEY-RETURNED-FROM-DOCKER_RUN-COMMAND"

docker exec -it $docker_key /bin/bash
```
