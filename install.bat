@echo off



echo *** %time% *** Deleting TRELLIS.2 directory if it exists
if exist TRELLIS.2\. rd /S /Q TRELLIS.2

echo *** %time% *** Cloning repository
git clone -b feature/local-tools https://github.com/RATATA17/TRELLIS.2_Experimental.git --recursive TRELLIS.2

cd /d TRELLIS.2

echo *** %time% *** Removing inner .git directory
if exist TRELLIS.2\.git rmdir /s /q TRELLIS.2\.git

echo *** %time% *** add app_mod.py
del ..\app_mod.py
copy ..\app_mod.py

echo *** %time% *** Creating venv
python -m venv venv

echo *** %time% *** Activating venv
call venv\scripts\activate.bat

echo *** %time% *** Upgrading pip
python -m pip install --upgrade pip
pip install wheel==0.45.1
pip install setuptools==65.5.0
pip install ninja==1.11.1.4

echo *** %time% *** Installing requirements
pip install imageio
pip install imageio-ffmpeg
pip install tqdm
pip install easydict
pip install opencv-python-headless
pip install trimesh
pip install transformers
pip install gradio==6.0.1
pip install tensorboard
pip install pandas
pip install lpips
pip install zstandard
pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
pip install kornia
pip install timm
pip install https://huggingface.co/datasets/Softology-Pro/VoC/resolve/main/flash_attn-2.7.4.post1-cp310-cp310-win_amd64.whl
pip install triton-windows==3.5.0.post21
pip install hf_xet

echo *** %time% *** Installing GPU torch
pip uninstall -y torch
pip uninstall -y torch
pip install --no-cache-dir --ignore-installed --force-reinstall --no-warn-conflicts torch==2.7.0+cu128 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

echo *** %time% *** Installing nvdiffrast
pip install --no-deps --no-build-isolation https://github.com/Deathdadev/TRELLIS.2-Windows/releases/download/test-wheels/nvdiffrast-0.4.0-cp310-cp310-win_amd64.whl

echo *** %time% *** Installing nvdiffrec
pip install --no-deps --no-build-isolation https://github.com/Deathdadev/TRELLIS.2-Windows/releases/download/test-wheels/nvdiffrec_render-0.0.0-cp310-cp310-win_amd64.whl

echo *** %time% *** Installing pillow-simd
pip install --no-deps --no-build-isolation https://github.com/Deathdadev/TRELLIS.2-Windows/releases/download/test-wheels/pillow_simd-9.5.0.post2-cp310-cp310-win_amd64.whl

echo *** %time% *** Installing cumesh
pip install --no-deps --no-build-isolation https://github.com/Deathdadev/TRELLIS.2-Windows/releases/download/test-wheels/cumesh-0.0.1-cp310-cp310-win_amd64.whl

echo *** %time% *** Installing FlexGEMM
pip install --no-deps --no-build-isolation https://github.com/Deathdadev/TRELLIS.2-Windows/releases/download/test-wheels/flex_gemm-0.0.1-cp310-cp310-win_amd64.whl

echo *** %time% *** Installing o-voxel
pip install --no-deps --no-build-isolation https://github.com/Deathdadev/TRELLIS.2-Windows/releases/download/test-wheels/o_voxel-0.0.1-cp310-cp310-win_amd64.whl

echo *** %time% *** Patching transformers
pip uninstall -y transformers
pip install transformers==4.57.6

pushd %CD%

echo *** %time% *** Downloading dinov3 models
cd /D %HF_HOME%\hub\
if exist models--camenduru--dinov3-vitl16-pretrain-lvd1689m\. rd models--camenduru--dinov3-vitl16-pretrain-lvd1689m /s/q
if exist models--facebook--dinov3-vitl16-pretrain-lvd1689m\. rd models--facebook--dinov3-vitl16-pretrain-lvd1689m /s/q
hf download camenduru/dinov3-vitl16-pretrain-lvd1689m
ren models--camenduru--dinov3-vitl16-pretrain-lvd1689m models--facebook--dinov3-vitl16-pretrain-lvd1689m

echo *** %time% *** Downloading RMBG-2.0 models
cd /D %HF_HOME%\hub\
if exist models--camenduru--RMBG-2.0\. rd models--camenduru--RMBG-2.0 /s/q
if exist models--briaai--RMBG-2.0\. rd models--briaai--RMBG-2.0 /s/q
hf download camenduru/RMBG-2.0
ren models--camenduru--RMBG-2.0 models--briaai--RMBG-2.0

popd

call venv\scripts\deactivate.bat
cd ..
echo *** %time% *** Finished TRELLIS.2 install
echo.
echo Check the stats for any errors.  Do not assume it worked.
pause
