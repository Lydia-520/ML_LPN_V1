import os
import pandas as pd
import numpy as np
import multiprocessing
from multiprocessing import Pool, Manager
from tqdm import tqdm

from regazzoni2022_mono import Simulation



# global variables
n_cardiac_cyc = 10
dt = 0.001 # Here this is a pseudo number to determine how many
save_last_n_cardiac_cycles = 2

output_index = 0




# ----------------------------- MAIN CODE ------------------------------------ #
def gen_results(sim_parameters, index=0):
    # Create an instance of the Simulation class
    simulation = Simulation(sim_parameters)


    # Get the new parameters from the baseline
    #simulation.get_parameters(start_array, end_array)
    simulation.get_parameters_generate(index=index)



    # Run simulation
    results_dict = simulation.integrate()



    # Save input parameters to results_dict
    results_dict['parameters'] = simulation.parameters

    # Extract metrics from simulation
    simulation.compute_clinical_metrics(results_dict)
    sim_metrics = results_dict['clinical_metrics']



    #combined_input = simulation.record_input(8) # get 12+8 = 20 numbers
    # In one_extra mode, 'index=0' obtains the default 12 numbers.
    combined_input = simulation.record_input_one_extra(index) # get numbers
    combined_output = simulation.record_output(sim_metrics)


    #print("combined_input.shape", combined_input.shape)
    #print("combined_output.shape", combined_output.shape)


    #combined_result = np.concatenate((combined_input, combined_output))
    combined_result = np.concatenate((combined_output, combined_input))

    return combined_result

def process_case(case_index, progress_queue, lock, output_file):

    global output_index

    # n_cardiac_cyc, dt, save_last_n_cardiac_cycles are from the parent thread in the __main__ function
    combined_result = gen_results([n_cardiac_cyc, dt, save_last_n_cardiac_cycles],
                                  index=output_index)

    #print("n_cardiac_cyc, dt, save_last_n_cardiac_cycles = ", n_cardiac_cyc, dt, save_last_n_cardiac_cycles)

    #print("case_index", case_index) # each sub-thread has an case_index
    #print("combined_result.shape", combined_result.shape) # 42 numbers = 30 + 12

    # Save results every 10000 cases
    if case_index % 10000 == 0:
        with lock:
            df = pd.DataFrame([combined_result])
            if os.path.exists(output_file):
                df.to_csv(output_file, mode='a', header=False, index=False) # append to file
            else:
                df.to_csv(output_file, mode='w', header=False, index=False) # write to the new file

    # Notify progress
    progress_queue.put(1)
    return combined_result

def update_progress(progress_queue, num_cases):
    pbar = tqdm(total=num_cases)
    while True:

        progress = progress_queue.get()
        if progress is None:  # Check for termination signal
            break
        pbar.update(progress)
    pbar.close()




def generate_results_task(index=0, # learnable variable control
                          output_path="./"):

    global output_index

    output_index = index

    ## These params in the parent thread are shared to the sub-threads that are created below.
    # General sim parameters
    n_cardiac_cyc = 10
    dt = 0.001 # Here this is a pseudo number to determine how many
    save_last_n_cardiac_cycles = 2

    num_cases = 50000# rows in combined_results.csv are num_cases + 3, like 200+3=203
    num_workers = multiprocessing.cpu_count()  # Use all available CPUs
    num_workers = 1


    # using multicore cpu to concurrently process data is faster
    #print("num_workers = ", num_workers)

    output_file = os.path.join(output_path, 'combined_results.csv')

    # Create a manager for the progress queue and a lock
    manager = Manager()
    progress_queue = manager.Queue()
    lock = manager.Lock()

    # Create a pool of workers
    with Pool(processes=num_workers) as pool:
        # Start a background thread to update the progress bar
        progress_thread = multiprocessing.Process(target=update_progress, args=(progress_queue, num_cases))
        progress_thread.start()

        # Map process_case function to all case indices
        # each thread runs a process_case function
        results = [pool.apply_async(process_case, (i, progress_queue, lock, output_file)) for i in range(num_cases)]

        #print("results_1.len = ", len(results)) # num_cases

        results = [result.get() for result in results]

        #print("results_2.len = ", len(results)) # num_cases


        # Signal the progress thread to terminate
        progress_queue.put(None)
        progress_thread.join()

        # Convert the results to a DataFrame and save to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, mode='a', header=False, index=False)  # Final save to ensure all results are included

    print('All cases processed and saved to combined_results.csv')


# ---------------------------------------------------------------------------- #



if __name__ == "__main__":


    ## These params in the parent thread are shared to the sub-threads that are created below.
    # General sim parameters
    n_cardiac_cyc = 10
    dt = 0.001 # Here this is a pseudo number to determine how many
    save_last_n_cardiac_cycles = 2

    num_cases = 20000 # rows in combined_results.csv are num_cases + 3, like 200+3=203
    num_workers = multiprocessing.cpu_count()  # Use all available CPUs

    # using multicore cpu to concurrently process data is faster
    #print("num_workers = ", num_workers)

    # Get directory of this script
    current_dir = os.path.dirname(os.path.realpath(__file__))
    output_file = os.path.join(current_dir, 'combined_results.csv')

    # Create a manager for the progress queue and a lock
    manager = Manager()
    progress_queue = manager.Queue()
    lock = manager.Lock()

    # Create a pool of workers
    with Pool(processes=num_workers) as pool:
        # Start a background thread to update the progress bar
        progress_thread = multiprocessing.Process(target=update_progress, args=(progress_queue, num_cases))
        progress_thread.start()

        # Map process_case function to all case indices
        # each thread runs a process_case function
        results = [pool.apply_async(process_case, (i, progress_queue, lock, output_file)) for i in range(num_cases)]

        #print("results_1.len = ", len(results)) # num_cases

        results = [result.get() for result in results]

        #print("results_2.len = ", len(results)) # num_cases


        # Signal the progress thread to terminate
        progress_queue.put(None)
        progress_thread.join()

        # Convert the results to a DataFrame and save to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, mode='a', header=False, index=False)  # Final save to ensure all results are included

    print('All cases processed and saved to combined_results.csv')
