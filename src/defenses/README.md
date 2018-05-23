# ISC_AML_2018

## Defenses

The sample_defense_submission included is based on the fMoW baseline classifier.  
[fMoW Baseline Code](https://github.com/fMoW/baseline)  
[fMoW Weights](https://github.com/fMoW/baseline/releases)  


### Submission Format

In addition to your software implementation, a defense submission must contain a file "metadata.json" that informs the evaluation script how to run your code.  In particular, it tells the evaluation script:

1. which Docker container to use with your code submission and 
2. what toplevel script in your submission to run in order to execute the defense (i.e. the "entry point").  

For the Docker image, any publicly-available docker container (e.g. from the [Google Container Registry](https://cloud.google.com/container-registry/) or [DockerHub](https://hub.docker.com)) should be fine.  Alternately, Neil has created a custom local image which is also available for your use (more on that below).  The format of the "metadata.json" file is as follows:


```
{
	"type": "defense", 
	"container": "YOUR-DOCKER-CONTAINER-NAME-GOES-HERE",   
	"container_gpu": "YOUR-DOCKER-CONTAINER-NAME-GOES-HERE",  
	"entry_point": "YOUR-SCRIPT-NAME-GOES-HERE"  
}
``` 

You should leave the <code>"type"</code> field as-is and modify the other three fields as needed for your code (although you probably are using a GPU and therefore it is the "container_gpu" and "entry_point" fields which are most important).  A concrete example that uses a public image from the Google Container Registry (prefix "gcr.io") is:
```
{
  "type": "defense",
  "container": "gcr.io/tensorflow/tensorflow:1.2.1",
  "container_gpu": "gcr.io/tensorflow/tensorflow:1.2.1-gpu",
  "entry_point": "run_defense.sh"
}
```
If you would like to use your own custom docker image, DockerHub (link above) allows one to publish images for free.   This is the preferred way of providing us with a custom image to use.

The "entry_point" should be the name of a bash shell script that takes two arguments, a directory containing images to classify and an output file where classification outputs will be written.  This script should then launch your classifier with these arguments (and anything else you need).  We recommend just using one of the [examples](./sample_defense/run_defense_noop.sh)  we provide (suitably modified as needed for your code).


### Local Dockerfiles
In addition to all publicly available docker images, Neil has also created a local Docker image that works with the sample baseline and is available for your use as well.  If you would like to use this image, you can build the image locally using his [Dockerfile](./Dockerfile) and then just use "simple_submission" as your Docker image (see the sample defense [metadata.json](./sample_defense/metadata.json)).

If neither of these options (public Dockerfile or Neil's local image) are sufficient, please contact us and we'll make alternate arrangements.
