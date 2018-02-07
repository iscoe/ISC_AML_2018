#!/home/neilf/Installs/anaconda3/bin/python
import json
import os
import subprocess

def main():
    # Change this to CLI 
    submission_name = 'submission_test'
    base_dir = '/home/neilf/Fendley/adversarial/ISC_AML_2018'

    submission_dir = os.path.join(base_dir, 'src/'+submission_name)
    data_dir = os.path.join(base_dir, 'test_images/')
    output_folder = os.path.join(base_dir, 'output/')
    
    #Load metadata from their submission
    metadata = json.load(open(os.path.join(submission_dir,'metadata.json')))
    outputname = '/output/' + submission_name + '_predictions.csv'
    #create nvidia docker command to run
    cmd = ['sudo', 'nvidia-docker', 'run',
           '-v', '{0}:/input_images'.format(data_dir),
           '-v', '{0}:/output'.format(output_folder),
           '-v', '{0}:/code'.format(submission_dir),
           '-w', '/code',
           '--user', 'www-data', metadata['container_gpu'], metadata['entry_point'],
           '/input_images', outputname]
    
    subprocess.call(cmd)



if __name__ == '__main__':
    main()
    