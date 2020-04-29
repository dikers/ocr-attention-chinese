#   OCR 识别


通过调用trdg，自动生成中文手写体图片， 然后通过crnn+ctc进行文本识别。


## 建立环境

```shell script
conda create -n  ocr-cn python=3.6 pip scipy numpy ##运用conda 创建python环境
source activate ocr-cn
pip install -r requirements.txt -i https://mirrors.163.com/pypi/simple/
```


## 准备数据

```shell script
cd create_data
sh ./generate_fsns_data.sh '../sample_data/test.txt'  0.1  '../output/'

```