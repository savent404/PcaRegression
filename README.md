# prepare
``` bash
sudo apt install python3-tk libxcb-xinerama0
pip3 install -r requirements.txt
```
# usage
``` bash
# train(put config.json in this dir)
python3 main.py --mode train
# predict
python3 main.py --mode predict
```