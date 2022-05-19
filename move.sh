

#scp -i path/to/key_file -r folder_with_files/ ucloud@x.x.x.x:/path/to/new_folder

#scp -i C:/Users/idaba/OneDrive/Skrivebord/key_file FILE_NAME ucloud@130.225.38.24:/home/ucloud/FOLDER_NAME
scp -i C:/Users/idaba/OneDrive/Skrivebord/key_file data/test1k ucloud@130.225.38.24:/home/ucloud



# FROM GPU TO LOCAL:
scp -i C:/Users/idaba/OneDrive/Skrivebord/key_file -r ucloud@130.225.38.24:/home/ucloud/gpu_files/mt517-180211\* data

scp -i C:/Users/idaba/OneDrive/Skrivebord/key_file ucloud@130.225.38.24:/home/ucloud/gpu_files/mt517-180211_all_abs_preds.npy data