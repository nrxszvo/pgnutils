CONDA_VER=latest
OS_TYPE=x86_64
CONDA_DIR=/home/ubuntu/miniconda
PY_VER=3.10

if [ ! -d "$CONDA_DIR" ]; then
	echo "removing local python site-packages..."
	rm -rf /home/ubuntu/.local/lib/python3.10/site-packages
	echo "installing conda..."
	curl -LO "http://repo.continuum.io/miniconda/Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh"
	bash Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh -p /home/ubuntu/miniconda -b
	rm Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh
fi

if ! command -v conda 2>&1 >/dev/null
then
	export PATH=/home/ubuntu/miniconda/bin:${PATH}
fi

conda update -y conda
conda install -c anaconda -y python=${PY_VER}
conda install -y "numpy<2.0" pyyaml pytorch torchvision torchaudio pytorch-cuda=12.4 lightning tensorboard -c pytorch -c nvidia
conda init
pip install fairscale
ln -s ~/mimicChessData/datasets .
