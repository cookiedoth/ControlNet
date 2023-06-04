wget https://huggingface.co/MexFoundation/map_diffusion-1.3/resolve/main/map_diffusion.ckpt -O models/map_diffusion.ckpt
gcloud auth login --no-launch-browser
gsutil cp gs://tokyo_suburbs/dataset.tar .
tar -xvf dataset.tar
gcloud auth revoke --all
python tool_add_control.py ./models/map_diffusion.ckpt ./models/map_control.ckpt
python fill_cn.py
