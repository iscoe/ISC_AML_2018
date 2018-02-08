#!/home/neilf/Installs/anaconda3/bin/python
import json
import os
import subprocess

def main():
    # Change this to CLI 
    defense_names = ['sample_defense_submission']
    base_dir = '/home/neilf/Fendley/adversarial/ISC_AML_2018'
    
    defense_dir = os.path.join(base_dir,'src/defenses')
    data_dir = os.path.join(base_dir, 'image_sets/test_images')
    output_dir = os.path.join(base_dir, 'output/')
    for defense in defense_names:
      submission_dir = os.path.join(defense_dir, defense)
      run_defense(submission_dir, data_dir, output_dir)

    
def run_defense(submission_dir, data_dir, output_dir):
    #Load metadata from their submission
    metadata = json.load(open(os.path.join(submission_dir,'metadata.json')))
    outputname = '/output/' + os.path.basename(submission_dir) + '_predictions.csv'
    #create nvidia docker command to run
    cmd = ['sudo', 'nvidia-docker', 'run',
           '-v', '{0}:/input_images'.format(data_dir),
           '-v', '{0}:/output'.format(output_dir),
           '-v', '{0}:/code'.format(submission_dir),
           '-w', '/code',
           '--user', 'www-data', metadata['container_gpu'], metadata['entry_point'],
           '/input_images', outputname]
    
    subprocess.call(cmd)



if __name__ == '__main__':
    main()
    