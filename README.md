# MouseSpatialGrid

This repo does two things
1. It generates simulated neural spikes in response to user-defined stimuli, based on STRFs and spatial tuning curves. This is the "input model." This is based on the work written by Junzi Dong, as described in her [2016 eNeuro paper](https://www.eneuro.org/content/3/1/ENEURO.0086-15.2015).
2. It puts those neural spikes through a spiking neural network, based on the [Aim Network](https://www.biorxiv.org/content/10.1101/2020.12.10.419762v1), which runs on the [DynaSim](https://github.com/DynaSim/DynaSim) Framework.

## 1. The Input Model
The input model is found in `/ICSimStim`. To get started, open `InputGaussianSTRF_main.m`.
There are two sets of "tuning", defined as "bird" and "mouse".
The STRFs for each option were optimized for their respective stimuli.
You can define your own sets of parameters and give it a name.

The model assumes stimuli are located at certain locations along the horizontal plane (`azimuth` in `InputGaussianSTRF_v2`).
The model weighs the stimuli's spectrograms by user-defined tuning curves, comebines them, convolves then with STRFs to generate firing rates, then create spike trains with poisson spiking generator.

Then the model calculates a discriminability measure based on the van Rossum spike distance, as described in [Dong et al., 2016](https://www.eneuro.org/content/3/1/ENEURO.0086-15.2015).

### 1.a Bird Tuning
If you set `tuning = 'bird'`, then the model will use birdsongs (located in `\stimuli`) as the stimuli.
It assumes that tuning curves are Gaussian-shaped, with tuning width being some user-defined value, sigma

### 1.b Mouse Tuning
Here, the input stimuli is white gaussian noise that were modulated with human speech.
The tuning curves were defined to be left sigmoid, right sigmoid, Gaussian centered at 0, and U-shaped (inverted Gaussian). These were based on work by [Ono & Oliver, 2014](https://www.jneurosci.org/content/34/10/3779).

## 2. The spiking Network Model
To get started, use `run_SpikingNetwork_main.m`. Specify the location of the input spike trains (generated using the input model).

Here, the `varies` and `netcons` structure must be specified for the model to run correctly.
The `varies` structure specifies parameters of the synapses and neurons within the network, and `netcons` specifies the connections between populations of neurons.

After the simulation, the post-processing step will calculate the firing rate and discrimination performance for each neuron in the network. The results are then displayed by calling `plotPerformanceGrids_new`.
