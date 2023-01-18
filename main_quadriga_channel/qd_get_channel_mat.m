function [channel_mat] = qd_get_channel_mat(rx_loc_x, rx_loc_y, rx_loc_z)
    global layout;
    global bandwidth;
    global n_sub_carr;
    layout.rx_position = [rx_loc_x; rx_loc_y; rx_loc_z];
    channel_mat = squeeze(layout.get_channels.fr(bandwidth, n_sub_carr));
end