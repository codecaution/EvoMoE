python3 -m pip uninstall fairseq -y
#pip install iopath
pip install hydra.core
pip install omegaconf
pip install bitarray
pip install sacrebleu
pip install iopath
pip install boto3
pip install fairscale
pip install tensorboardX
cp -r /jizhicfs/brendenliu/stablemoe ~
cd ~/stablemoe
#python3 -m pip install --editable ./
pip install --user --editable .
python3 setup.py build_ext --inplace
