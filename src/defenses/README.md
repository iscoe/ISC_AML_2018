# ISC_AML_2018

## Defenses

The sample_defense_submission included is based on the fMoW baseline classifier.  
[fMoW Baseline Code](https://github.com/fMoW/baseline)  
[fMoW Weights](https://github.com/fMoW/baseline/releases)  


### Format    

In addition to your software implementation, a defense submission should contain a file "metadata.json" that informs the evaluation script how to run your code.  In particular, it tells the script (a) which Docker container to use with your code and (b) what script in your submission to run in order to execute the evaluation (i.e. the "entry point").  Any publicly-available docker container (e.g. from the Google Container Registry or [DockerHub](https://hub.docker.com)) should be fine.  Alternately, Neil has created a custom local image which is also available for your use (more on that below).  The format of the "metadata.json" file is as follows:


```
{
	"type": "defense", 
	"container": "YOUR-DOCKER-CONTAINER-NAME-GOES-HERE",   
	"container_gpu": "YOUR-DOCKER-CONTAINER-NAME-GOES-HERE",  
	"entry_point": "YOUR-SCRIPT-NAME-GOES-HERE"  
}
``` 

A concrete example that uses a public image from the Google Container Registry (prefix "gcr.io") is
```
{
  "type": "defense",
  "container": "gcr.io/tensorflow/tensorflow:1.2.1",
  "container_gpu": "gcr.io/tensorflow/tensorflow:1.2.1-gpu",
  "entry_point": "run_defense.sh"
}
```

If you would like to publish your own global docker image, DockerHub (link above) allows one to publish images for free.  

### Local Dockerfiles
In addition to all publicly available docker images, Neil has also created a local Docker image that works with the sample basline and is available for your use as well.  If you would like to use this image, you can build the image locally using his [Dockerfile](./Dockerfile) and then just use the sample defense [metadata.json](./sample_defense/metadata.json).  

If neither of these options (public Dockerfile or Neil's example) are sufficient, please contact us and we'll arrange to get your Docker image running locally.  
