# Simulation of the mMIMO OFDM system with nonlinear power amplifiers 

Installation
------------

To run the framework several python modules are required, the easiest way to do it is with the anaconda/miniconda software.

1. Install Anaconda or Miniconda from: https://www.anaconda.com/

2. Create and activate a new environment:
    ```
    conda create -n mimo_sim python=3.9
    conda activate mimo_sim
    ```
 

3. Add conda-forge channels:
    ```
    conda config --add channels conda-forge
    ```
4. Install packages:
    ```
    conda env update --file requirements.yml
    ```
5. Install VS Code or any other development framework and select created environment ``mimo_sim`` as the interpreter.

Documentation
-----------------
1. To build the documentation go to the docs directory:
    ```
    cd .\docs\
    ```
2. Build the docs:
    ```
    .\make.bat html make
    ```
   
Once built, the documentation can be viewed by opening the ``.\docs\build\html\index.html`` with any web browser.


Usage
-----
Select and run one of the  ``main_...`` scripts or create a custom one from the building blocks and functions in the
parent directory.
