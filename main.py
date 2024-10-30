import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import json
from torchsummary import summary
from dnn import *  # Ensure this imports your SimpleNN as definition


from genResults import generate_results_task
from process_data import process_data_task_one_extra



import shutil

import os
import time



# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def create_data(output_index=0,
                batch_size = 64,
                test_size = 0.2,
                path="./",
                output_path="./",
                mode="increment"):

    '''
    parameter:
         mode, "increment" / "one_extra" refers to how we use the data in
               "filtered_input.pt".
               "increment": to include more numbers successively.
               "one_extra": only include a different new number
    '''

    # each element represents the numbers of filtered_input.pt tensor
    output_array_selection = [12, 13, 14, 15, 16, 17, 18, 19, 20]

    in_path = os.path.join(path, 'filtered_output.pt')
    out_path = os.path.join(path, 'filtered_input.pt')

    # Load data
    data_input = torch.load(in_path)
    data_output = torch.load(out_path)

    # Convert to NumPy arrays for easier manipulation
    data_input_np = data_input.numpy()
    data_output_np = data_output.numpy()

    #print("data_input_np.shape = ", data_input_np.shape)
    #print("data_output_np.shape = ", data_output_np.shape)
    #exit(0)

    # Determine the sizes dynamically
    input_size = data_input_np.shape[1]


    #output_size = data_output_np.shape[1]
    # select the needed input_size
    output_size = output_array_selection[output_index]


    # obtain the needed input values
    if mode == "increment":
        data_output_np = data_output_np[:, :output_size]
    elif mode == "one_extra":
        if index > 0:
            output_size = 13

    #print("data_input_np.shape = ", data_input_np.shape)
    #print("data_output_np.shape = ", data_output_np.shape)
    #print("data_output_np[1, :] = ", data_output_np[1, :])
    #exit(0)



    # Split into training + validation and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(data_input_np,
                                                                data_output_np,
                                                                test_size=test_size,
                                                                random_state=42)

    # Split training + validation into training and validation
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                      test_size=test_size, random_state=42)

    # Compute normalization parameters from the training data
    input_min = np.min(X_train, axis=0)
    input_max = np.max(X_train, axis=0)

    output_min = np.min(y_train, axis=0)
    output_max = np.max(y_train, axis=0)


    # Save normalization parameters
    normalization_params = {
        'input_min': input_min.tolist(),
        'input_max': input_max.tolist(),
        'output_min': output_min.tolist(),
        'output_max': output_max.tolist()
    }

    with open(os.path.join(output_path, 'normalization_params.json'), 'w') as f:
        json.dump(normalization_params, f)

        # Apply min-max normalization to training, validation, and test data
        X_train_normalized = (X_train - input_min) / (input_max - input_min)
        X_val_normalized = (X_val - input_min) / (input_max - input_min)
        X_test_normalized = (X_test - input_min) / (input_max - input_min)

        y_train_normalized = (y_train - output_min) / (output_max - output_min)
        y_val_normalized = (y_val - output_min) / (output_max - output_min)
        y_test_normalized = (y_test - output_min) / (output_max - output_min)

        # Convert back to PyTorch tensors
        X_train_tensor = torch.tensor(X_train_normalized, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val_normalized, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test_normalized, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_normalized, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val_normalized, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test_normalized, dtype=torch.float32)



        # Define the device
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Move data to the device
        X_train_tensor = X_train_tensor.to(device)
        X_val_tensor = X_val_tensor.to(device)
        X_test_tensor = X_test_tensor.to(device)
        y_train_tensor = y_train_tensor.to(device)
        y_val_tensor = y_val_tensor.to(device)
        y_test_tensor = y_test_tensor.to(device)

        # Create datasets and DataLoaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader




'''
train_loader, val_loader, test_loader = create_data(output_index=6,
                                                    batch_size=64,
                                                    test_size=0.2,
                                                    path="./",
                                                    mode="one_extra")
exit(0)
'''



def creat_dnn_model(output_index=0,
                    output_array_selection=[12, 13, 14, 15, 16, 17, 18, 19, 20],
                    hidden_sizes=[128, 256, 256, 256, 256, 128],  # Initialize parameters
                    mode="increment",  # "increment", "one_extra"
                    dropout_rate=0.5  # Dropout rate to prevent overfitting
                    ):

    # This is from filtered_output.pt
    input_size = 30
    output_size = output_array_selection[output_index]

    if mode == "one_extra":
        if output_index == 0:
            model = LargerNN(input_size, hidden_sizes, 12, dropout_rate).to(device)
        else:
            model = LargerNN(input_size, hidden_sizes, 13, dropout_rate).to(device)
    else:
        model = LargerNN(input_size, hidden_sizes, output_size, dropout_rate).to(device)

    return model




def prepare_data(output_index=0,
                 path_save="./save_results",
                 mode="one_extra"): # "one_extra"

    output_array_selection = [12, 13, 14, 15, 16, 17, 18, 19, 20]
    sel = output_array_selection[output_index]

    output_size = sel


    if mode == "one_extra":
        output_path = os.path.join(path_save, "output_index_" + str(sel))
    else:
        output_path = os.path.join(path_save, "output_increment_" + str(output_size))


    os.makedirs(output_path)


    generate_results_task(index=output_index, output_path=output_path)
    process_data_task_one_extra(index=output_index, output_path=output_path)


    return output_path # return data path






# save test loss
test_accuracy_records = []


def model_train_test(output_index=0,
                     path="./save_results",
                     epochs=500,
                     mode="increment"): # "increment", "one_extra"

    output_array_selection = [12, 13, 14, 15, 16, 17, 18, 19, 20]
    sel = output_array_selection[output_index]

    input_size = 30


    if mode == "one_extra":
        if output_index == 0:
            output_size = 12
        else:
            output_size = 13 # fixed at 13
    else:
        output_size = sel


    hidden_sizes = [128, 256, 256, 256,  256, 128]

    #print("output_size", output_size)


    output_path = path
    path_data = path


    train_log_file = open(os.path.join(output_path, "train_log.txt"), "w")
    test_log_file = open(os.path.join(output_path, "test_log.txt"), "w")
    #return

    # create data
    train_loader, val_loader, test_loader = create_data(output_index,
                                                        batch_size=64,
                                                        test_size=0.2,
                                                        path=path_data,
                                                        output_path=output_path,
                                                        mode=mode)

    # Define the loss function and optimizer
    #feature_weights = torch.tensor(np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    #                                         1.0, 1.0]), dtype=torch.float32).to(device)

    # 20 numbers as weights
    feature_weights = torch.tensor(np.array([1.0, 1.0, 1.0, 1.0,
                                             1.0, 1.0, 1.0, 1.0,
                                             1.0, 1.0, 1.0, 1.0,

                                             # add 8 numbers
                                             1.0, 1.0, 1.0, 1.0,
                                             1.0, 1.0, 1.0, 1.0,
                                             ]), dtype=torch.float32).to(device)

    # get needed weights according to output_size
    if mode == "one_extra":
        if output_index == 0:
            feature_weights = feature_weights[:12]
        else:
            feature_weights = feature_weights[:13]
    else:
        feature_weights = feature_weights[:output_size]

    #print(feature_weights.shape)


    # creat model
    model = creat_dnn_model(output_index=output_index, mode=mode)

    #print(model)
    #exit(0)


    criterion = WeightedMSELoss(feature_weights).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)




    # Training loop
    for epoch in range(epochs):  # Number of epochs
        model.train()
        train_loss = 0  # Initialize training loss accumulator

        # Training phase
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            # Ensure data is on the same device as the model

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()  # Accumulate training loss

            # Calculate average training loss for the epoch
            avg_train_loss = train_loss / len(train_loader)


        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                # Ensure data is on the same device as the model
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

            # Calculate average validation loss for the epoch
            avg_val_loss = val_loss / len(val_loader)

        # Print training and validation loss
        print(f'Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

        ss = 'Epoch ' + str(epoch + 1) + ', Training Loss: ' + str(avg_train_loss) + \
             ', Validation Loss: ' + str(avg_val_loss)
        train_log_file.writelines([ss, "\r\n"])


    # Testing loop
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            # Ensure data is on the same device as the model
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

        print(f'Test Loss: {test_loss / len(test_loader):.4f}')
        ss = 'Test Loss: ' + str(test_loss / len(test_loader))
        test_log_file.writelines([ss, "\r\n"])

        test_accuracy_records.append(test_loss / len(test_loader))


    # Save model configuration and state dictionary
    model_config = {
        'input_size': input_size,
        'hidden_sizes': hidden_sizes,
        'output_size': output_size,
        'mode': mode,
        'index': sel
    }

    # Save configuration to a JSON file
    with open(os.path.join(output_path,'model_config.json'), 'w') as f:
        json.dump(model_config, f)

    # Save state dictionary
    torch.save(model.state_dict(), os.path.join(output_path, 'model_state.pth'))


    # close file
    train_log_file.close()
    test_log_file.close()







# output_array_selection = [12, 13, 14, 15, 16, 17, 18, 19, 20]
# index                  = [0 -> 8]

#cur_time = time.time() # get timesteps
cur_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
print("current time = ", cur_time)

path_data = "./"
path_save = os.path.join("save_results", str(cur_time))

os.makedirs(path_save)



### mode = one_extra
for index in range(0, 9): # 0,1,2,3, ..., 8

    print("========================== start one_extra data prepare, index = ", index)
    path = prepare_data(output_index=index,
                        path_save=path_save,
                        mode="one_extra")
    print("========================== end one_extra data prepare, index = ", index)
    print()


    print("========================== start one_extra training index = ", index)

    model_train_test(output_index=index,
                     path=path,
                     epochs=500,
                     mode="one_extra")

    print("========================== end one_extra training index = ", index)
    print()




best_log_file = open(os.path.join(path_save, "best_result_log.txt"), "w")


best_loss = 1.0
best_index = 0
# i = 0 : the default 12 numbers
# i = 1 -> 8: one_extra mode, index_13 -> index_20
for i in range(len(test_accuracy_records)):

    print("index = ", i, " : loss = ", test_accuracy_records[i])

    ss = "index = " + str(i) + " : loss = " + str(test_accuracy_records[i])
    best_log_file.writelines([ss, "\r\n"])

    if best_loss > test_accuracy_records[i]:
        best_loss = test_accuracy_records[i]
        best_index = i

print("best index = ", best_index)
print("best loss = ", best_loss)

ss = "best index = " + str(best_index) + " : loss = " + str(best_loss)
best_log_file.writelines([ss, "\r\n"])
best_log_file.close()
