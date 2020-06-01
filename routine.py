import h5py
import datetime
import os
import glob
import matplotlib.pyplot as plt
import matplotlib
import multiprocessing as mp
import csv
import numpy as np
from math import pi, log

matplotlib.use('Agg')

# Define necessary paths
process_path = r"/mnt/g/channels/20181101_avhrr_h35_extent"
output_path = r"/mnt/h/input"

# Define calibration coefficient dictionary from the netcdf
calibration_dict = {
    # Scale factor from xarray read data is converted to float automatically
    # '1': [0.01,0.00001191044000,1.43876900000000,140.30000000000000,0.00000000000000,0.00,0.00000000000000],
    # '2': [0.01,0.00001191044000,1.43876900000000,245.00000000000000,0.00000000000000,0.00,0.00000000000000],
    '1': [1, 0.000011910440, 1.43876900, 140.3, 0.0, 0.0, 0.0],
    '2': [1, 0.000011910440, 1.4387690, 245.0, 0.0, 0.0, 0.0],
    # 'a': [0.00001,0.00001191044000,1.43876900000000,13.10000000000000,0.00000000000000,0.00,0.0],
    'a': [1, 0.000011910440, 1.4387690, 13.10, 0.0, 0.0, 0.0],
    '4': [1, 0.000011910440, 1.4387690, 0.0, 1.0013640, -0.504870, 933.630070],
    '5': [1, 0.000011910440, 1.4387690, 0.0, 1.001140, -0.381710, 839.620060]
}


# define method for calibrating the given radiance if it is going to be converted to reflectance
def calibrate_reflectance(channel, dataset):
    # scale_factor is one calibration_dict[channel][0]
    f_sol = calibration_dict[channel][3]
    radiance = dataset.astype('float32')
    radiance_non_zero = dataset[np.nonzero(dataset)]
    # Original formula (radiance_non_zero / 100 * scale_factor * 100 * pi) / float(f_sol)
    reflectance_percentage = (radiance_non_zero * pi) / float(f_sol)
    radiance[np.nonzero(dataset)] = reflectance_percentage
    return radiance


# define method for calibrating the given radiance if it is going to be converted to brightness temperature
def calibrate_brightness_temperature(channel, dataset):
    # scale_factor is one calibration_dict[channel][0]

    # get formula's coefficient from the dictionary
    c1 = calibration_dict[channel][1]
    c2 = calibration_dict[channel][2]
    alpha = calibration_dict[channel][4]
    beta = calibration_dict[channel][5]
    wnc = calibration_dict[channel][6]

    radiance = dataset.astype('float32')
    radiance_non_zero = dataset[np.nonzero(dataset)] / 100.0
    temperature_brightness = ((c2 * wnc) / np.log(1.0 + (c1 * wnc ** 3) / radiance_non_zero) - beta) / alpha
    radiance[np.nonzero(dataset)] = temperature_brightness
    return radiance


# initiate calibration process
def run_calibration(file_):
    calibrated = []
    print(file_)
    channel = file_.split("__")[1].split(".")[0]
    f = h5py.File(os.path.join(process_path, file_), 'r')
    ds = f[channel]
    # reflectance calibration is required only in channels 1 , 2 and a
    if channel in ['a', '1', '2']:
        calibrated = calibrate_reflectance(channel, ds[()])

    # reflectance calibration is required only in channels 4, 5
    if channel in ['4', '5']:
        calibrated = calibrate_brightness_temperature(channel, ds[()])
    f.close()
    os.remove(os.path.join(process_path, file_))
    # export calibrated channel data to an hdf file
    with h5py.File(os.path.join(output_path, file_), 'w') as h5file:
        h5file.create_dataset('/' + str(channel),
                              shape=calibrated.shape,
                              dtype=np.dtype('int16'),
                              data=calibrated * 100)


if __name__ == '__main__':
    # decide files to be processed in parallel
    increment = 8
    remove_files_option = True
    first_date = "20190512"
    if remove_files_option:
        remove_file_flag = ' --remove-files'
    else:
        remove_file_flag = ' '
    input_path = output_path
    start_date = datetime.datetime.strptime(first_date, '%Y%m%d')
    for d_ in range(19):
        try:
            date_ = (start_date + datetime.timedelta(days=d_)).strftime("%Y%m%d")
            start = datetime.datetime.now()
            files = glob.glob1(process_path, "eps_M01_{}_*.hdf".format(date_))

            for f in range(0, len(files), increment):
                files_in = files[f:f + increment]
                N = mp.cpu_count()
                with mp.Pool(processes=N) as p:
                    results = p.map(run_calibration, [file_ for file_ in files_in])
            file_name = "{}_avhrr_h35_extent_not_merged.tar.gz".format(date_)
            cmd = "tar -czvf " + file_name + " eps_M01_{}_*.hdf".format(date_) + remove_file_flag
            os.chdir(input_path)
            print(cmd)
            end = datetime.datetime.now()
            os.system(cmd)
            print(end - start)
        except:
            print(date_, "There is an error")
            with open(datetime.datetime.now().strftime("%Y%m%d") + "_log.csv", 'ab') as csv_file:
                csvwriter = csv.csvwriter(csv_file)
                csvwriter.writerow(
                    ['{} :: {} :: Error'.format(datetime.datetime.now().strftime("%Y%m%d %H:%M:%S"), date_)])
            continue
