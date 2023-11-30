# MouseSpatialGrid

This repo does two things
1. It generates simulated neural spikes in response to user-defined stimuli, based on STRFs and spatial tuning curves. This is the "input model." This is based on the work written by Junzi Dong, as described in her [2016 eNeuro paper](https://www.eneuro.org/content/3/1/ENEURO.0086-15.2015).
2. It puts those neural spikes through a spiking neural network, based on the [Aim Network](https://www.biorxiv.org/content/10.1101/2020.12.10.419762v1), which runs on the [DynaSim](https://github.com/DynaSim/DynaSim) Framework.

## 1. The Input Model
The model for generating inputs is found in `/ICSimStim`. To get started, open `main_STRF_target.m`.

This will convolve the spatial grid stimuli from [Nocon et al. 2023](https://www.nature.com/articles/s42003-023-05126-0) with STRFs based on measurements of mouse MGB from [Lohse et al. 2020](https://www.nature.com/articles/s41467-019-14163-5).

The final output of this script is 'default_STRF_with_offset_200k.mat', which contains firing rate traces for stimulus onsets and offsets (i.e. silent/quieter portions of stimuli). The Poisson-based input cells in the network model will use these firing rate traces to randomly generate spikes.

The offset-based firing rate is created by flipping the convolution of the STRF and stimulus across the x-axis and adding an offset based on the maximum onset firing rate. The resulting trace is then half-wave rectified. Modeling onset and offset-sensitive inputs is based on findings of parallel processing of onset and offset in mouse auditory cortex [Li et al. 2019](https://www.cell.com/cell-reports/fulltext/S2211-1247(19)30399-7).

## 2. The spiking Network Model
To get started, use `run_SpikingNetwork_withOffset.m`.

### 2.a Initializing code and loading parameters
The first section of the code loads in the FR traces from section 1 and defines the `study_dir` directory where the model code and spikes will be stored; `expName` where the results will be stored under the folder `simData`, where ALL simulation results are stored.

The second section will load the parameters (`varies`) and simulation options (`options`) for the model.

The main fields in `options` are:
  `nCells` - the # of columns within the model (`nCells`), which should be equal to the amount of spatially-tuned or frequency-tuned channels you want in the model
  `opto` - whether you want to run multiple simulations with either a control or optogenetic condition (see `params_7.m` and `params_8.m`)
  `locNum` - the spatial grid configurations you want to simulate. If you want to simulate a full spatial grid experiment, `locNum = []`, which will run all 24 configurations (4 target-only, 4 masker-only, and 16 target-masker).
  `time_end` - the length of simulations. For a full spatial grid simulation, `time_end = padToTime * 24`; for a single location, `time_end = 3500` (in ms).

All params files should be .m files with the `varies` struct, which Dynasim uses to vary model parameters. The first entry in `varies` will always be
```
varies(1).conxn = '(On->On,Off->Off');
varies(1).param = 'trial';
varies(1).range = trialInds(:)';
```
This defines the number of simulations you want to run for each target identity. `trialInds` is a vector `1:20`, where `1:10` means that 10 simulations for target 1 will be ran and `11:20` means that 10 simulations for target 2 will be ran.

### 2.a Loading input spikes and spatial tuning

The third section of the code will weigh each of the firing rate traces based on the channel's spatial tuning in `tuningcurve`. `tuningcurve` is defined from -90˚ to +90˚ and should have a number of rows equal to `nCells`. For masked simulations, we add the weighted target and masker firing rate traces and then re-weigh them so that the summed up spatial tuning weights (`t_wt` and `m_wt`) equal 1. 

* Later down the line, we want to add other tuning curves based on work by [Ono & Oliver, 2014](https://www.jneurosci.org/content/34/10/3779) and [Panniello et al. 2018](https://pubmed.ncbi.nlm.nih.gov/29136122/). These studies defined the tuning curves in IC as left sigmoid, right sigmoid, Gaussian centered at 0, and U-shaped (inverted Gaussian). 

### 2.b Running the simulation

After saving the firing rate traces to the `study_dir/solve` directory, the model will be ran using the `columnNetwork_V2` function. This function contains the model populations (`%% neuron populations`) and synaptic connections (`%% connections`) between cell types.

### 2.c Post-processing

After the simulations have finished, `SpikingNetwork_withOffset` will call the script (`postProcessSims`) which runs all the post-processing.

`nVaried` calculates the amount of parameter set variations you ran. Since the first entry in `varies` will always make us run 20 simulations at the least (10 target 1 and 10 target 2 simulations), that doesn't count here. If any of the other entries in `varies.range` has more than one entry, that will be the parameter that's varied (e.g. if `varied(end).range = [0.01 0.02]`) without an opto-condition, we will have run 40 simulations total.

`createSimNotes` writes a txtfile with the information on all varied parameters in the `varies` struct in the `simData/expName` folder . Keep in mind that `createSimNotes` does not track the parameter values in `columnNetwork_V2`. That's why the code saves a `results.mat` file which contains the spikes from the topmost cell (as of writing this, the `R2On` cells) and the `.model` field, which _does_ save all the parameters in the model.

`postProcessData_new` will do a lot of things, such as 1) calculate the SPIKE-distance-based discriminability between both target identities, 2) calculates the average firing rate during stimulus playback, and 3) make a figure showing the raster plot for all units for a given parameter set. 

The last part of the code `%% convert peakDrv to samples (10000 Hz)` is based on analysis from [Penikis and Sanes 2023](https://www.jneurosci.org/content/43/1/93/tab-e-letters). For grids and discriminability, we don't use this, but we've included these analyses in submissions to show how our model can explain the results in that paper.

# Important changes to third-party toolboxes

## 1. strflab_v1.45/preprocessing/preprocSound.m

Line 100-102: stimSampleRate is set to 10000 Hz instead of 1000 Hz

## 2. strflab_v1.45/preprocessing/sound/timefreq.m

Line 65-67: increment is set to fix(sampleRate/10000) instead of fix(0.001*sampleRate)

These two changes result in spectrograms with a sampling frequency of 10000 Hz, which matches the Dynasim simulation dt of 0.1 ms.
If these changes aren't made, the spectrograms have a sampling frequency of 1000 Hz, which is too coarse for the purposes of our simulations.
