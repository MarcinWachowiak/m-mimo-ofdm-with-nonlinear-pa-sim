Usage
=====
Select and run one of the  ``main_...`` scripts or create a custom one from the building blocks and functions in the parent directory

Installation
------------

To run the framework several python modules are required, easiest way to do it is with the anaconda/miniconda software.

1. Install Anaconda or Miniconda from: https://www.anaconda.com/

2. Create and activate new environment::

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

