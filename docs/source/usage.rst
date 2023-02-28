Installation
------------

To run the framework several python modules are required, the easiest way to do it is with the anaconda/miniconda software.

1. Install Anaconda or Miniconda from: https://www.anaconda.com/

2. Create and activate a new environment::

       conda create -n mimo_sim python=3.9
       conda activate mimo_sim

3. Add conda-forge channels::

    conda config --add channels conda-forge

4. Install packages::

       conda install matplotlib, numpy, scipy, numba
       conda install pytorch cudatoolkit=11.6 -c pytorch -c conda-forge

   or::

    conda env update --file requirements.yml

5. Install VS Code or any other development framework and select created environment ``mimo_sim`` as the interpreter.

Usage
-----
Select and run one of the  ``main_...`` scripts or create a custom one from the building blocks and functions in the
parent directory


Plotting and saving the simulation outputs
------------------------------------------

The results of the simulations are stored in the ``figs`` directory and the CSV files containing numerical results
are in ``figs/csv_results``.

The CSV files are mostly organized in the following structure:
 - The first row is the values of the parameters that have been swept.
 - The following rows contain the measured value (e.g. BER) in regard to the swept parameter.
   For multiple simulation parameters, the next rows correspond to the values of the other parameters.

For example, for CNC and MCNC receivers the following rows contain results for the sequence of the iterations.
The structure of the CSV files might be case-dependent for more details refer to the CSV saving routine in
the simulation script and the naming of the saved arrays.

An example of CSV formatting code::

    data_lst = [] #each list entry should be an array of values
    data_lst.append(n_ant_arr)
    for arr1 in bers_per_chan_per_nite_per_n_ant: # flattening the array before saving to csv
        for arr2 in arr1:
            data_lst.append(arr2)
    utilities.save_to_csv(data_lst=data_lst, filename=filename_str)

Depending on the IDE and the path at which the scripts are executed, the path used in the scripts to the
``figs`` and ``figs/csv_results`` might have to be adjusted both in the ``utilities.py`` and at the end
of the simulation script when the figure is saved in ``plt.savefig(...)``

For saving plots the two possible options are::

     plt.savefig("../figs/%s.png" % filename_str, dpi=600, bbox_inches='tight')

or::

     plt.savefig("figs/%s.png" % filename_str, dpi=600, bbox_inches='tight')

For saving the CSV data the two variants are (both ``utilities.save_to_csv`` and ``utilities.read_from_csv``
should be modified)::

    with open("../figs/csv_results/%s.csv" % filename, 'rw', newline='') as csv_file:

or::

    with open("figs/csv_results/%s.csv" % filename, 'rw', newline='') as csv_file:

**Modify and test the saving accordingly, otherwise, your current simulation results and plots might be lost!**
