
import time
import subprocess
import os

os.environ['KAGGLE_API_TOKEN'] = 'KGAT_1d71fff1057b6cbeb696e983235beb02'
kernel_slug = 'dosomethingbetter/neural-bates-dml-10d-gpu'

print(f'Waiting for kernel {kernel_slug} to finish...')
while True:
    try:
        result = subprocess.run(['python', '-m', 'kaggle', 'kernels', 'status', kernel_slug], capture_output=True, text=True)
        output = result.stdout.strip()
        print(f'Status check: {output}', flush=True)
        
        if 'complete' in output.lower():
            print('Kernel finished! Downloading output...', flush=True)
            subprocess.run(['python', '-m', 'kaggle', 'kernels', 'output', kernel_slug, '-p', '.'], check=True)
            print('Output downloaded successfully.', flush=True)
            break
        elif 'error' in output.lower() or 'cancel' in output.lower():
            print('Kernel stopped with an error or was canceled.', flush=True)
            break
            
    except Exception as e:
        print(f'Error checking status: {e}', flush=True)
        
    time.sleep(300) # Check every 5 minutes
