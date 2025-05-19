import subprocess
import time
import datetime
import torch
import re

default_fname = 'temp_report'

def get_all_temps():
    res_obj = subprocess.run('nvidia-smi', capture_output=True, timeout=5)
    res_str = str(res_obj.stdout)
    raw_temps = re.findall('..[0-9][0-9]C', res_str)
    temps = [int(s[:-1].strip()) for s in raw_temps]
    return temps

# accepts torch.device object
def get_device_temp(device):
    temps = get_all_temps()
    return temps[device.index]

# run in a training loop once. If the temp is above max_temp, 
# delays until the temperature gets knocked down to min_temp.
# temp is in degrees C
# device is a torch.device object
def monitor_stage(device, min_temp=60, max_temp=75):
    T = get_device_temp(device)
    secs_to_cool = 0
    if T > max_temp:
        while T > min_temp:
            secs_to_cool += 1
            time.sleep(1)
            T = get_device_temp(device)
    return secs_to_cool

# writes a temperature log, until interrupted or until it's written for a week.
def write_temp_report(fname=None, delay=1, max_len=3600*24*7):
    if fname is None:
        fname = default_fname
    temps = get_all_temps()
    num_devices = len(temps)
    f = open(fname, 'w')
    f.write(str(datetime.datetime.now()))
    f.write('\n')
    f.write(f"Writing temperature log, delay is {delay} seconds between rows.\n")
    col_header_str_list = [f"GPU {ind}" for ind in range(num_devices)]
    f.write('\t'.join(col_header_str_list) + '\n')
    f.write('\t'.join([str(t) for t in temps]) + '\n')
    f.close()
    episode = 1
    while episode < max_len:
        time.sleep(delay)
        episode += 1
        temps = get_all_temps()
        f = open(fname, 'a')
        f.write('\t'.join([str(t) for t in temps]) + '\n')
        f.close()
    return None


