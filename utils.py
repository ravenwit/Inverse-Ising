from .solver import InverseIsing
from .samples import BoltzmanSamples, NESamples
from .machines import BoltzmanMachine, NEMachine
import numpy as np
import matplotlib.pyplot as plt


# Generate true fields and interactions
def generate_true_fields(N):
    rand_matrix = np.random.normal(0, 1/N, size=(N, N))
    symmetric_matrix = (rand_matrix + rand_matrix.T) / 2
    np.fill_diagonal(symmetric_matrix, 0)
    return symmetric_matrix, np.random.normal(0, 1, size=N)/2


def analysis_of_lr(inverse_ising, lrs, iter=300):
    '''
    lrs = [1, 0.5, 0.1, 0.09, 0.05, 0.01, 0.009, 0.005, 0.001]
    '''

    inverse = inverse_ising
    np.save("alr_sample_{}_{}".format(inverse.N, inverse.data.n_samples), inverse.data.data)

    for lr in lrs:
        
        print("Starting... lr={}".format(lr))
        inferMachine, loss = inverse.infer_parameters(atol=0.01, lr=lr, iter=iter)
        np.save("alr_{}_{}_{}_coupling".format(inverse.N, inverse.data.n_samples, lr), inferMachine.couplings)
        np.save("alr_{}_{}_{}_field".format(inverse.N, inverse.data.n_samples, lr), inferMachine.fields)
        np.save("alr_{}_{}_{}_loss".format(inverse.N, inverse.data.n_samples, lr), loss)


def analysis_of_size(true_machine, sizes, lr=0.05):
    '''
    lrs = [1, 0.5, 0.1, 0.09, 0.05, 0.01, 0.009, 0.005, 0.001]
    '''
    np.save("asz_true_coupling_{}".format(true_machine.N), true_machine.couplings)
    np.save("asz_true_field_{}".format(true_machine.N), true_machine.fields)

    for n_sample in sizes:

        training_ensamble = true_machine.generate_ensamble(n_sample)
        inverse = InverseIsing(training_ensamble, non_equilibrium=False, symmetric=True, partition_number=np.abs(true_machine.calc_Z()))
        
        np.save("asz_sample_{}_{}_{}".format(inverse.N, inverse.data.n_samples, lr), inverse.data.data)
        
        print("Starting... size={}".format(n_sample))
        inferMachine, loss = inverse.infer_parameters(atol=0.01, lr=lr)
        np.save("asz_{}_{}_{}_coupling".format(inverse.N, inverse.data.n_samples, lr), inferMachine.couplings)
        np.save("asz_{}_{}_{}_field".format(inverse.N, inverse.data.n_samples, lr), inferMachine.fields)
        np.save("asz_{}_{}_{}_loss".format(inverse.N, inverse.data.n_samples, lr), loss)


def plot_losses(conf):
    N = conf["N"]
    n_samples = conf["n_sample"]
    lrs = conf["lrs"]
    fig, ax = plt.subplots()
    if isinstance(n_samples, int):
        for lr in lrs:
            loss = np.load("../InverseIsingData/version2/alr_{}_{}_{}_loss.npy".format(N, n_samples, lr))
            ax.plot(loss, label="lr={}, sample_size={}".format(lr, n_samples))
            ax.legend()
    else:
        # assert len(lrs) == 1
        for n_sample in n_samples:
            loss = np.load("../InverseIsingData/version2/asz_{}_{}_{}_loss.npy".format(N, n_sample, lrs))
            ax.plot(loss, label="lr={}, sample_size={}".format(lrs, n_sample))
            ax.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.ylim([-30, -50])
    plt.show()


def plot_fields(conf):
    N = conf["N"]
    n_samples = conf["n_sample"]
    lrs = conf["lrs"]
    
    true_fields = np.load("../InverseIsingData/version2/asz_true_field_{}.npy".format(N))
    infer_fields_array = []
    text2 = []
    for n_sample in n_samples:
        infer_fields = np.load("../InverseIsingData/version2/asz_{}_{}_{}_field.npy".format(N, n_sample, lrs))
        infer_fields_array.append(infer_fields)
        text2.append("Inferred Fields for sample_size={}".format(n_sample))

    compare_array(true_fields, infer_fields_array, True, text1="True Fields", text2=text2)
    plt.show()


def plot_couplings(conf):
    N = conf["N"]
    n_samples = conf["n_sample"]
    lrs = conf["lrs"]
    
    true_coupling = np.load("../InverseIsingData/version2/asz_true_coupling_{}.npy".format(N))
    infer_coupling_array = []
    text2 = []
    for n_sample in n_samples:
        infer_coupling = np.load("../InverseIsingData/version2/asz_{}_{}_{}_coupling.npy".format(N, n_sample, lrs))
        infer_coupling_array.append(infer_coupling)
        text2.append("Inferred Couplings for sample_size={}".format(n_sample))

    compare_array(true_coupling, infer_coupling_array, True, text1="True Couplings", text2=text2)
    plt.show()



def open_salamandar_data(path, non_equilibrium=False):
    salamandar_data = []
    with open(path, "r") as afile:
        for line in afile:
            line_array = [int(x) for x in str(line.strip()).split(" ")]
            salamandar_data.append(line_array)
    salamandar_data = np.array(salamandar_data)
    if non_equilibrium:
        return NESamples(salamandar_data.T)
    else:
        print("nobuou")
        return BoltzmanSamples(salamandar_data.T, big=True)


def compare_array(array1, array2, multi=False, text1="True Couplings", text2="Inferred Couplings", linedata=""):
    '''
    array1 is True
    array2 is inferred
    '''

    # Create the plot
    plt.figure(figsize=(10, 6))
    if multi:
        for index, array in enumerate(array2):
            plt.plot(array1.flatten(), array.flatten(), marker='o', markersize=5, linestyle='', lw=0.5, label=text2[index])
    
        plt.ylabel('{}'.format(text2[0]))
        plt.title('Comparison of {} and {}'.format(text1, text2[0]))

    else:
        plt.plot(array1.flatten(), array2.flatten(), marker='o', markersize=5, linestyle='', lw=0.5, label='Point')
        plt.ylabel('{}'.format(text2))
        plt.title('Comparison of {} and {}'.format(text1, text2))
    
    plt.xlabel('{}'.format(text1))
    

    plt.grid(True)

    if linedata:
        xx = [linedata[0, 0], linedata[1, 0]]
        yy = [linedata[0, 1], linedata[1, 1]]
        plt.plot(xx, yy, 'r--', lw=2, label='y = x')
        
    plt.legend()



