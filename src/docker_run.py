#!/home/neilf/Installs/anaconda3/bin/python
import json
import os
import subprocess

def main():
    # Change this to CLI 
    base_dir = '/home/neilf/Fendley/adversarial/ISC_AML_2018'
    submission = os.path.join(base_dir, 'src/submission_test/')
    data_dir = os.path.join(base_dir, 'test_images/')
    output_folder = os.path.join(base_dir, 'output/')
    
    #Load metadata from their submission
    metadata = json.load(open(os.path.join(submission,'metadata.json')))
    
    #create nvidia docker command to run
    cmd = ['sudo', 'nvidia-docker', 'run',
           '-v', '{0}:/input_images'.format(data_dir),
           '-v', '{0}:/output'.format(output_folder),
           '-v', '{0}:/code'.format(submission),
           '-w', '/code',
           '--user', 'www-data', metadata['container_gpu'], metadata['entry_point'],
           '/input_images', '/output/predictions.npy']
    
    subprocess.call(cmd)



if __name__ == '__main__':
    main()
    