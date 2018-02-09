#!/home/neilf/Installs/anaconda3/bin/python
import json
import os
import subprocess


def main():
    ### Use this to run on the default submissions 
    base_dir = '/home/neilf/Fendley/adversarial/ISC_AML_2018'
    output_dir = os.path.join(base_dir, 'output/')
    sample_defense_dir = os.path.join(base_dir,'src/defenses/sample_defense_submission/')
    sample_offense_dir = os.path.join(base_dir, 'image_sets/test_images')
    
    defense_folders = []
    offense_folders = []
    offense_folders.append(sample_offense_dir) 
    defense_folders.append(sample_defense_dir)
    
    run_submissions(defense_folders,offense_folders,output_dir)


def run_submissions(defense_folders,offense_folders,output_dir):
 
    for defense in defense_folders:
      for offense in offense_folders:
        run_defense(defense, offense, output_dir)

    
def run_defense(submission_dir, offense_dir, output_dir):
    #Load metadata from their submission
    metadata = json.load(open(os.path.join(submission_dir,'metadata.json')))
    outputname = '/output/' + os.path.basename(submission_dir) +'_'+os.path.basename(offense_dir)+ '_predictions.csv'
    #create nvidia docker command to run
    cmd = ['sudo', 'nvidia-docker', 'run',
           '-v', '{0}:/input_images'.format(offense_dir),
           '-v', '{0}:/output'.format(output_dir),
           '-v', '{0}:/code'.format(submission_dir),
           '-w', '/code',
           '--user', 'www-data', metadata['container_gpu'], metadata['entry_point'],
           '/input_images', outputname]
    
    subprocess.call(cmd)



if __name__ == '__main__':
    main()
    