# mimo_simulation_py

MIMO simulation in python

---

# VM setup - Chrome Remote Desktop:

1. Create VM in Google cloud
    1. Enable display device
2. Connect to it via SSH
3. VM remote desktop setup:
    1. ```
       sudo apt update 
       sudo apt install --assume-yes wget tasksel
       ```
    2. ``` 
       wget https://dl.google.com/linux/direct/chrome-remote-desktop_current_amd64.deb
       sudo apt-get install --assume-yes ./chrome-remote-desktop_current_amd64.deb
       ```
    3. ``` 
       sudo tasksel install ubuntu-desktop
       ```
    4. ```
       sudo DEBIAN_FRONTEND=noninteractive \
       apt install --assume-yes  task-gnome-desktop
       ```
    5. ```
       sudo bash -c 'echo "exec /etc/X11/Xsession /usr/bin/gnome-session" > /etc/chrome-remote-desktop-session'
       ```
    6. ```
       sudo systemctl disable lightdm.service
       ```

4. Go to the Chrome Desktop: Set up via SSH: Set up another computer. Follow the instructions on the site.
5. Check if Remote desktop is online:
   ```
   sudo systemctl status chrome-remote-desktop@$USER
   ```
6. Login to the VM
7. Conda/Miniconda set up
    1. Download installation file from: https://docs.conda.io/en/latest/miniconda.html#linux-installers
    2. Run the file: ```bash Miniconda3-latest-Linux-x86_64.sh```
    3. Create and activate new environment:
       ```
       conda create -n mimo_sim python=3.9
       conda activate mimo_sim
       ```
    4. Add conda-forge channels:
       ```
       conda config --add channels conda-forge
       ```
    6. Install packages:
       ```
       conda install matplotlib, numpy, scipy, pytorch, numba
       ```
8. Install VS Code via snap:
   ```
   sudo apt install snapd
   sudo snap install code --classic
   ```
9. Create and add SSH key to Github:
   ```
   ssh-keygen -t ed25519 -C "your_email@example.com"
   clip < ~/.ssh/id_ed25519.pub or nano ~/.ssh/id_ed25519.pub
   ```
10. Fetch the project:
    ```
    git clone git@github.com:MarcinWachowiak/mimo-simulation-py.git
    ```
11. Open in code and test:
    ```
    code 
    ```

---