# NBIO-UTAH
Repository aimed to integrate our efforts in neural data analysis

Reminder: we should choose a license

Functions description:

-Extract_LFP:

 This function extracts loads the information in a .Ns6 file into a matlab
matrix in witch each line is the meassurement of an electrode.
 It needs the functions openNEN and openNsx to be in any subfoulder.
 It also desamples the meassurement for a factor 'down'. It takes a
predifined value of 100 if not given any.
 It reordes the electrodes given the file 'Electrodes_Order.mat' and
includes the trigger saved in the .NEV file in the last line.

-Electrodes_Order:

It is a vector containing the order of the electrodesin the Utah array.
