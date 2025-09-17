the original instructions expect cudatoolkit 10.2 but since i have a newer gpu ie rtxa5000 with toolkit 11.7 installed i had to change smthns

for example i used this line for conda installations
conda create -n 3dsps python=3.9
conda activate 3dsps
conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 pytorch-cuda=11.1 -c pytorch -c nvidia

and then for pip installs-
pip install plyfile opencv-python trimesh==2.35.39 tensorboardX easydict tqdm h5py matplotlib pyyaml imageio pandas


and then in the setup.py changed a flag from 
os.environ["TORCH_CUDA_ARCH_LIST"] = "3.7+PTX;5.0;6.0;6.1;6.2;7.0;7.5"
to
os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5;8.0;8.6"

