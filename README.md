# MouseSpatialGrid

This repo does two things
1. It generates simulated neural spikes in response to user-defined stimuli, based on STRFs and spatial tuning curves. This is the "input model." This is based on the work written by Junzi Dong, as described in her [2016 eNeuro paper](https://www.eneuro.org/content/3/1/ENEURO.0086-15.2015).
2. It puts those neural spikes through a spiking neural network, based on the [Aim Network](https://www.biorxiv.org/content/10.1101/2020.12.10.419762v1), which runs on the [DynaSim](https://github.com/DynaSim/DynaSim) Framework.

## 1. The Input Model
The input model is found in `/ICSimStim`. To get started, open `InputGaussianSTRF_onandoffset.m`.
There are two sets of "tuning", defined as "bird" and "mouse".
The STRFs for each option were optimized for their respective stimuli.
You can define your own sets of parameters and give it a name.

The model assumes stimuli are located at certain locations along the horizontal plane (`azimuth` in `InputGaussianSTRF_v4`).
The model weighs the stimuli's spectrograms by user-defined tuning curves, comebines them, convolves then with STRFs to generate firing rates, then create spike trains with poisson spiking generator.

Onset-sensitive and offset-sensitive input spikes are created; the latter activity is created by flipping the convolution of the STRF and stimulus across the x-axis and adding an offset based on the maximum onset firing rate. The resulting trace is then half-wave rectified. Modeling onset and offset-sensitive inputs is based on findings of parallel processing of onset and offset in mouse auditory cortex [Li et al. 2019](https://www.cell.com/cell-reports/fulltext/S2211-1247(19)30399-7).

Then the model calculates a discriminability measure based on the SPIKE-distance, using the same template-matching procedure as described in [Nocon et al. 2021](https://www.biorxiv.org/content/10.1101/2021.09.11.459906v2.full).

### 1.a Bird Tuning
If you set `tuning = 'bird'`, then the model will use birdsongs (located in `\stimuli`) as the stimuli.
It assumes that tuning curves are Gaussian-shaped, with tuning width being some user-defined value, sigma

### 1.b Mouse Tuning
Here, the input stimuli is white gaussian noise that were modulated with human speech.
The tuning curves were defined to be left sigmoid, right sigmoid, Gaussian centered at 0, and U-shaped (inverted Gaussian). These were based on work by [Ono & Oliver, 2014](https://www.jneurosci.org/content/34/10/3779) and [Panniello et al. 2018](https://pubmed.ncbi.nlm.nih.gov/29136122/). 

## 2. The spiking Network Model
To get started, use `run_SpikingNetwork_column.m`. You can specify the location of the input spike trains (generated using the input model) by un-commenting `uigetdir`.

Here, the `varies` and `netcons` structure must be specified for the model to run correctly.
The `varies` structure specifies parameters of the synapses and neurons within the network, and `netcons` specifies the connections between populations of neurons. The script calls the `columnNetwork` function, which then calls Dynasim to build the model and run simulations.

After the simulation, the post-processing step will calculate the firing rate and discrimination performance for each neuron in the network by calling `postProcessData_new`. `createSimNotes` writes a txtfile with the information on all varied parameters in the `varies` struct in the same folder as the cell rasters. Keep in mind that `createSimNotes` does not track the parameter values in the `columnNetwork` function. The results are then displayed by calling `plotPerformanceGrids_V3`.

IMPORTANT CHANGES TO THIRD-PARTY TOOLBOXES

## 1. strflab_v1.45/preprocessing/preprocSound.m

Line 100-102: stimSampleRate is set to 10000 Hz instead of 1000 Hz

## 2. strflab_v1.45/preprocessing/sound/timefreq.m

Line 65-67: increment is set to fix(sampleRate/10000) instead of fix(0.001*sampleRate)

These two changes result in spectrograms with a sampling frequency of 10000 Hz, which matches the Dynasim simulation dt of 0.1 ms
If these changes aren't made, the spectrograms have a sampling frequency of 1000 Hz, which is too slow for the purposes of our simulations
