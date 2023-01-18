clear all;
close all;

qd_channel_env_setup(64 , 4096, 15e3, 3.5e9, 300, '3GPP_3D_UMa_LOS');
channel_mat1 = qd_get_channel_mat(300, 0, 1.5);
channel_mat2 = qd_get_channel_mat(300, 0, 1.5);
