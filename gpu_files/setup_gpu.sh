#ssh -i ~/path_to_key/key_name ucloud@x.x.x.x

#LOCAL:
#ssh -i "C:/Users/idaba/OneDrive/Skrivebord/key_file" ucloud@130.225.38.62
#scp -i C:/Users/idaba/OneDrive/Skrivebord/key_file -r gpu_files ucloud@130.225.38.62:/home/ucloud


sudo apt update
sudo apt full-upgrade -y
sudo apt install nvidia-headless-460 nvidia-utils-460 -y

sudo apt install python3-pip
sudo apt install protobuf-compiler
sudo reboot
#ssh -i "C:/Users/idaba/OneDrive/Skrivebord/key_file" ucloud@130.225.38.62


#then from GPU:
#pip install -r requirements.txt

#proto stuff:
# DO NOT: git clone https://github.com/protocolbuffers/protobuf.git # NOPE :(

#maybe just pip install protobuf ????

# /gpu_files/protobuf/python/protobuf-3.12.4 ??????


# ( CHECK SARA TO IDA ON SLACK ???)