'''
Evaluates error between simulation results and patient-specific metrics that we
aim to match.
'''

import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import os
from prediction import *

#from regazzoni2022_mono_pred import Simulation
from regazzoni2022_mono import Simulation


# ----------------------------- MAIN CODE ------------------------------------ #
def gen_results(y_test, case_type, sim_parameters,

                #index is used when mode is one_extra.
                output_size,
                index=13, mode="increment",

                output_path="./"):

    # Create an instance of the Simulation class
    simulation = Simulation(sim_parameters)

    # Set parameters
    # y_pred = [2.47014160e+01, 1.78618673e-02, 2.09048271e+00, 1.24385327e-01,
    #          1.03915676e-01, 1.19570255e-01, 3.10267568e-01, 5.02145253e-02,
    #          1.72481899e+01, 7.64159317e+01, 4.09196510e+01, 4.60950012e+01]
    #
    # y_orig = [2.49059842e+01, 1.41062380e-02, 1.85244397e+00, 1.20549375e-01,
    #          1.24153034e-01, 1.09650453e-01, 3.21833094e-01, 5.11867428e-02,
    #          1.78679284e+01, 7.91088126e+01, 4.17553168e+01, 4.88132920e+01]


    # Get the new parameters from the baseline
    simulation.get_parameters_pred(y_test, index, mode)

    # Run simulation
    results_dict = simulation.integrate()

    # Save input parameters to results_dict
    results_dict['parameters'] = simulation.parameters

    # Extract metrics from simulation
    simulation.compute_clinical_metrics(results_dict)
    sim_metrics = results_dict['clinical_metrics']

    combined_input = simulation.record_input(index=8) # obtain all data : 12+8 = 20
    combined_output = simulation.record_output(sim_metrics)

    # There are all in 1-dimensional
    #print("combined_input.shape = ", combined_input.shape)
    #print("combined_output.shape = ", combined_output.shape)


    # adjust combined_input size, according to the settings at the trainining state
    if mode == "one_extra":

        assert (output_size == 13)

        combined_input_new = combined_input[:output_size]
        combined_input_new[12] = combined_input[index-1] # replace column 13

        combined_input = combined_input_new

    else:
        combined_input = combined_input[:output_size]


    #print("combined_input.shape = ", combined_input.shape)
    #exit(0)

    combined_result = np.concatenate((combined_input, combined_output))

    # Create output directory if it doesn't exist
    output_dir = os.path.join(output_path, f'evaluate_{case_type}')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, 'sim_metrics.txt'), 'w') as file:
        for metric, metric_value in sim_metrics.items():
            file.write(f"{metric}: {metric_value['Value']} {metric_value['Units']} \n")

    simulation.plot_results(results_dict, output_dir, font_size=12)

    # Save simulation results
    # simulation.save_results(results_dict, output_dir)

    # Plot patient-specific metrics and simulation metrics on bar plot
    # simulation.plot_metrics(sim_metrics, os.path.join(output_dir, 'metric_comparison.png'))

    return combined_result

# ---------------------------------------------------------------------------- #
# For parallel execution in differential_evolution, need to use the following
if __name__ == "__main__":

    # Load parameters of model (as dictionary) (all in mmHg/mL units)
    # Specify the name of the file (without .py extension)
    # parameters_file = 'regazzoni2022_parameters'
    # parameters_module = importlib.import_module(parameters_file) # Dynamically import the module
    # Get the path of the parameters file
    # parameters_filepath = parameters_module.__file__
    # Get the parameters dictionary from the module
    # parameters = parameters_module.parameters

    # Get directory of this script
    current_dir = os.path.dirname(os.path.realpath(__file__))
    #print("current_dir = ", current_dir)
    #exit(0)

    # saved result path of model training
    path = "./save_results"


    # General sim parameters
    n_cardiac_cyc = 10
    dt = 0.001 # Here this is a psudo number to determine how many
    save_last_n_cardiac_cycles = 2




    dir_names_l1 = os.listdir(path)

    fullpath_model = []

    print("---------- scanning model files ...")
    for f in dir_names_l1:

        fullpath_model_subdir = []
        fullpath_model_subdir.append(path + "/" + f)

        #print("f = ", f)
        dir_names_l2 = os.listdir(path + "/" + f)

        for f1 in dir_names_l2:
            if f1.endswith(".pt") is False and f1.endswith(".txt") is False:
                #print("f1 = ", f1)
                fullpath_model_subdir.append(path + "/" + f + "/" + f1)

        fullpath_model.append(fullpath_model_subdir)

    #exit(0)

    # print("fullpath_model", fullpath_model)
    # for i in fullpath_model:
    #     print("fullpath_model = ", i)
    # exit(0)


    # select which data is used for prediction, batch size is 1
    # num is the index number.
    num = 10



    for fullpath in fullpath_model:
        fullpath_model_subdir = fullpath

        print("----------- start to process : ", fullpath_model_subdir[0])

        # fullpath_model_subdir[0]: is the basic path containing xxx.pt file
        filtered_output_pt_path = fullpath_model_subdir[0] + "/" + "filtered_output.pt"
        filtered_input_pt_path = fullpath_model_subdir[0] + "/" + "filtered_input.pt"

        '''
        currently, the code only supports batch size = 1 for inference
        '''
        example_input = torch.load(filtered_output_pt_path)[num, :]
        y_orig = torch.load(filtered_input_pt_path).numpy()[num, :]


        # deal with model inference
        for model_path in fullpath_model_subdir[1:]: # ignore [0] element

            print("process model ", model_path)

            # Load model configuration
            with open(os.path.join(model_path, 'model_config.json'), 'r') as f:
                model_config = json.load(f)

            output_size = model_config['output_size']
            mode = model_config['mode']
            index = model_config['index']

            #print("y_orig.shape = ", y_orig.shape)

            if mode == "one_extra":
                assert (output_size == 13)
                assert (index > 12)
                y_orig_new = y_orig[:13]
                y_orig_new[12] = y_orig[index-1]
            else:
                y_orig_new = y_orig[:output_size]


            y_pred = get_prediction(example_input, model_path)

            # y_pred and y_orig are only 1 dimensional allowed
            #print("y_pred.shape = ", y_pred.shape)
            #print("y_orig.shape = ", y_orig.shape)
            #exit(0)

            y_pred_results = gen_results(y_pred, 'pred',
                                         [n_cardiac_cyc, dt, save_last_n_cardiac_cycles],
                                         output_size=output_size,
                                         index=index,
                                         mode=mode,
                                         output_path=model_path
                                         )
            #print(y_pred_results)


            y_orig_results = gen_results(y_orig_new, 'orig',
                                         [n_cardiac_cyc, dt, save_last_n_cardiac_cycles],
                                         output_size=output_size,
                                         index=index,
                                         mode=mode,
                                         output_path=model_path
                                         )
            #print(y_orig_results)
