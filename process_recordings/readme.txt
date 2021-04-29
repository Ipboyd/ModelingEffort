Given experimental data somewhere...
- recordingDataToModelFormat
	calculate PSTHs from data. Concatenates PSTHs for each song/masker configuration, saves to a format used in DNN simulations.
- genInputFromExpData
	calculate PSTHs from data. Mixes song and noise PSTHs by weighing them with tuning curves.
	Generates spike trains for SNN modeling.
	Must manually select representative responses to song and noise.
- runSNNonExpData
	Sets up and runs SNN using the data generated from genInputFromExpData.
	Calculates and plots performance.