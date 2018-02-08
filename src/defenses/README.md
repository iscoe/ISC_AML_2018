# ISC_AML_2018

## Defenses
This is the folder with the challenge team provided defenses.  
The defenses should contain a metadata.json file of the following form that contains the following information  

`{
	"type": "defense", 
	"container": "cpu docker container name",   
	"container_gpu": "gpu docker container name",  
	"entry_point": "run_defense.X"  
}` 

The sample_defense_submission included is based on the fMoW baseline classifier.  
[fMoW Baseline Code](https://github.com/fMoW/baseline)  
[fMoW Weights](https://github.com/fMoW/baseline/releases)  
### Dockerfiles
There is also a Dockerfile provided which is a modified version of the Keras Dockerfile. 