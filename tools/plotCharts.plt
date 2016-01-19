################################################################################
# Plot charts
################################################################################
# Alfredo Canziani, Aug 15
################################################################################

# Define a decent style for plotting (from Torch)
blue_050 = "#1D4599"
green_050 = "#11AD34"
set style line 1  linecolor rgbcolor blue_050  linewidth 2 pt 7
set style line 2  linecolor rgbcolor green_050 linewidth 2 pt 5 linetype 3
set style increment user

# Switch on the grid
set grid

# Plotting on screen
#  + Cross-entropy
set term wxt 0
plot './train.log' u 1 w lines title 'Training cross-entropy', './test.log' u 2 w lines title 'Testing cross-entropy'
set term postscript eps enhanced color
set o 'cross-entropy.eps'
replot

#  + Accuracy
set term wxt 1
plot './train.log' u 2 w lines title 'Training accuracy', './test.log' u 1 w lines title 'Testing accuracy'
set term postscript eps enhanced color
set o 'accuracy.eps'
replot

# Pause 1 second
pause -1
