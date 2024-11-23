# pip install -r ../requirements.txt

cd SAM-6D

### Install pointnet2
cd Pose_Estimation_Model/model/pointnet2
python setup.py install
# cd ../../../

# ### Download ISM pretrained model
# cd Instance_Segmentation_Model
# python download_fastsam.py
# python download_dinov2.py
# cd ../

# ### Download PEM pretrained model
# cd Pose_Estimation_Model
# python download_sam6d-pem.py

# cd ../../
