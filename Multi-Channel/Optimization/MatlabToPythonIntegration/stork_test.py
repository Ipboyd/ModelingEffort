import os
import sys

sys.path.append("../../")

import numpy as np
import torch

import matplotlib.pyplot as plt
import seaborn as sns

import stork.datasets
from stork.datasets import HDF5Dataset, DatasetView

from stork.models import RecurrentSpikingModel
from stork.nodes import InputGroup, ReadoutGroup, LIFGroup
from stork.connections import Connection
from stork.generators import StandardGenerator
from stork.initializers import FluctuationDrivenCenteredNormalInitializer


def main():

    #I just copied the needed files in
    datadir = os.getcwd()


    nb_inputs = 700
    duration = 0.7
    time_step = dt = 2e-3
    nb_time_steps = int(duration / time_step)
    time_scale = 1
    unit_scale = 1
    validation_split = 0.9


    gen_kwargs = dict(
        nb_steps=nb_time_steps,
        time_scale=time_scale / time_step,
        unit_scale=unit_scale,
        nb_units=nb_inputs,
        preload=True,
        precompute_dense=False,
        unit_permutation=None,
    )

    #print(os.getcwd())
    #print(os.path.join(datadir, "shd_train.h5"))

    train_dataset = HDF5Dataset(os.path.join(datadir, "shd_train.h5"), **gen_kwargs)

    # Split into train and validation set
    mother_dataset = train_dataset
    elements = np.arange(len(mother_dataset))
    np.random.shuffle(elements)
    split = int(validation_split * len(mother_dataset))
    valid_dataset = DatasetView(mother_dataset, elements[split:])
    train_dataset = DatasetView(mother_dataset, elements[:split])

    test_dataset = HDF5Dataset(os.path.join(datadir, "shd_test.h5"), **gen_kwargs)

    # Model Parameters
    batch_size = 30 #400
    beta = 20
    device = torch.device("cuda")
    dtype = torch.float


    nb_hidden_units = 128
    nb_hidden_layers = 3
    nb_classes = 20

    tau_mem = 20e-3
    tau_syn = 10e-3
    tau_readout = duration

    lr = 5e-3
    nb_epochs = 5 #100


    act_fn = stork.activations.SuperSpike
    act_fn.beta = beta
    neuron_group = LIFGroup

    model = RecurrentSpikingModel(batch_size, nb_time_steps, nb_inputs, device, dtype)

    # Define regularizer parameters (set regularizer strenght to 0, if you don't want to use them)
    lowerBoundL2Strength = 100
    lowerBoundL2Threshold = 1e-3
    upperBoundL2Strength = 0.1
    upperBoundL2Threshold = (
        7  # Regularizes spikecount: 7 spikes ~ 10 Hz in 700ms simulation time
    )

    # Define regularizer list
    regs = []

    regLB = stork.regularizers.LowerBoundL2(
        lowerBoundL2Strength, threshold=lowerBoundL2Threshold, dims=False
    )
    regs.append(regLB)
    regUB = stork.regularizers.UpperBoundL2(
        upperBoundL2Strength, threshold=upperBoundL2Threshold, dims=1
    )
    regs.append(regUB)


    sigma_u = 1.0
    nu = 15.8

    initializer = FluctuationDrivenCenteredNormalInitializer(
        sigma_u=sigma_u, nu=nu, timestep=dt
    )


    input_group = model.add_group(InputGroup(nb_inputs))



    upstream_group = input_group
    for l in range(nb_hidden_layers):
        hidden_neurons = model.add_group(
            LIFGroup(nb_hidden_units, tau_mem=tau_mem, tau_syn=tau_syn, activation=act_fn)
        )

        hidden_neurons.regularizers.extend(regs)
        con = model.add_connection(Connection(upstream_group, hidden_neurons))
        con.init_parameters(initializer=initializer)

        upstream_group = hidden_neurons


    readout_group = model.add_group(
        ReadoutGroup(nb_classes, tau_mem=tau_readout, tau_syn=tau_syn, initial_state=-1e-2)
    )

    con_ro = model.add_connection(Connection(upstream_group, readout_group))
    con_ro.init_parameters(initializer=initializer)


    

    for i in range(nb_hidden_layers):
        model.add_monitor(stork.monitors.SpikeCountMonitor(model.groups[1 + i]))

    for i in range(nb_hidden_layers):
        model.add_monitor(stork.monitors.StateMonitor(model.groups[1 + i], "out"))

    loss_stack = stork.loss_stacks.MaxOverTimeCrossEntropy()


    opt = stork.optimizers.SMORMS3
    #generator = StandardGenerator(nb_workers=2)
    generator = StandardGenerator(nb_workers=0,persistent_workers=False)


    model.configure(
        input=input_group,
        output=readout_group,
        loss_stack=loss_stack,
        generator=generator,
        optimizer=opt,
        optimizer_kwargs=dict(lr=lr),
        time_step=dt,
    )

    #Added due to multiprocessing issue?
    model.loader_kwargs = dict(num_workers=0,persistent_workers=False)

    # Monitors spike counts before training
    res = model.monitor(test_dataset)
    total_spikes = [torch.sum(res[i]).item() for i in range(nb_hidden_layers)]

    results = {}
    res = model.monitor(test_dataset)
    total_spikes_per_layer = [torch.sum(res[i]).item() for i in range(nb_hidden_layers)]
    results["avg_layer_freq_init"] = [
        nb / len(test_dataset) / duration / nb_hidden_units for nb in total_spikes_per_layer
    ]

    print(results)

    plt.figure(dpi=150)
    stork.plotting.plot_activity_snapshot(
        model, data=test_dataset, nb_samples=5, point_alpha=0.3
    )
    plt.show()

    history = model.fit_validate(
    train_dataset, valid_dataset, nb_epochs=nb_epochs, verbose=False
    )

    results["train_loss"] = history["loss"].tolist()
    results["train_acc"] = history["acc"].tolist()
    results["valid_loss"] = history["val_loss"].tolist()
    results["valid_acc"] = history["val_acc"].tolist()

    res = model.monitor(test_dataset)
    total_spikes_per_layer = [torch.sum(res[i]).item() for i in range(nb_hidden_layers)]
    results["avg_layer_freq"] = [
        nb / len(test_dataset) / duration / nb_hidden_units for nb in total_spikes_per_layer
    ]

    scores = model.evaluate(test_dataset).tolist()
    results["test_loss"], _, results["test_acc"] = scores

    fig, ax = plt.subplots(2, 2, figsize=(5, 3), dpi=150)

    for i, n in enumerate(["train_loss", "train_acc", "valid_loss", "valid_acc"]):
        if i < 2:
            a = ax[0][i]
        else:
            a = ax[1][i - 2]

        a.plot(results[n], color="black")
        a.set_xlabel("Epochs")
        a.set_ylabel(n)

    ax[0, 1].set_ylim(0, 1)
    ax[1, 1].set_ylim(0, 1)

    sns.despine()
    plt.tight_layout()

    print("Test loss: ", results["test_loss"])
    print("Test acc.: ", results["test_acc"])

    plt.figure(dpi=150)
    stork.plotting.plot_activity_snapshot(
        model, data=test_dataset, nb_samples=5, point_alpha=0.3
    )


if __name__ == "__main__":
    # the guard is ALWAYS required on Windows when anything might spawn
    main()

# import torch
# import stork
# from stork.neurons import LIFneuron
# from stork.layer import DenseLayer
# from stork.initializers import fluctuation_driven_

# net = torch.nn.Sequential(
#     DenseLayer(in_features=1, out_features=32, neuron=LIFNeuron),
#     DenseLayer(in_features=32, out_features =1,neuron=LIFneuron)
# )
# fluctuation_drive_(net)

# x= torch.rand(100,1)
# spikes = net(x)
# loss = spikes.mean()
# loss.backward()
