import matlab.engine

matlab = matlab.engine.start_matlab()

matlab.rng(210);  # Set RNG state for repeatability

A = 10000  # Transport block length, positive integer
rate = 449 / 1024  # Target code rate, 0<R<1
rv = 0  # Redundancy version, 0-3
modulation = 'QPSK'  # Modulation scheme, QPSK, 16QAM, 64QAM, 256QAM
nlayers = 1  # Number of layers, 1-4 for a transport block

ldpc_en_cfg = matlab.ldpcEncoderConfig()
ldpc_de_cfg = matlab.ldpcDecoderConfig()
