# 環境  
Ubuntu18.04
Python3.6  
Anaconda3  

# 必要なパッケージをインストール  
## 仮想環境の作成とアクティベート  

```
conda create -m py36  
conda activate py36  
conda install pip  
```

## 一般的なパッケージをインストール

```
pip install pandas
pip install numpy
pip install tensorflow  
pip install keras  
pip install scikit-learn  
```

## tensorflow-servingをインストール

```
pip install tensorflow-serving-api  
```

## grpcをインストール  
リクエストを送るためのパッケージ  
tensorflow-servingには、rpcを使ってリクエストを送るため  

```
pip install grpcio  
```

# dockerのインストール  
## レポジトリのアップデート

```bash
sudo apt update  
```

## HTTPS経由でrepositoryをやりとり出来るようにするためのパッケージをインストール  

```bash
sudo apt install -y \
     apt-transport-https \
     ca-certificates \
     curl \
     software-properties-common
```

## Dockerの公式GPG keyを追加する  

```bash
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo apt update
```

※curlコマンドで、URLにhttpリクエストを送り、gpgを受け取ってからパイプラインで渡してる。  
※curlコマンドとは・・・任意のサーバーやWebサイトへhttpリクエストを送って、そのレスポンスをチェックすることができるコマンド。  
※-fsSLというオプションは、  
-HTTPリクエストがサーバーのエラーにより失敗した時に22というEXIT CODEを返す。  
-進行の状況を表示せず、エラーメッセージは表示する。  
-アクセスしたページが移動していた場合、もう１度移動先にリクエストを送る。  
というオプション  
 
※apt-key add -   
apt-keyコマンドは、パッケージをインストールするときに、本物かどうかを証明するためのgpgに対するコマンド。  
addで、gpgをubuntuに追加して、パッケージの安全性を保証している。今回はdockerのパッケージを保証している。  
「-」は標準出力から受け取るときに使うオプション  

## インストール

```bash
sudo apt install -y docker-ce
```

dockerのインストールでエラーが出たら以下で対応  

```bash
sudo dpkg --audit
```

```
dpkg: パッケージ dokcer-ce の処理中にエラーが発生しました (--configure):
 パッケージ `dokcer-ce' はインストールされていないので、設定できません
処理中にエラーが発生しました:
 dokcer-ce
```

```bash
sudo dpkg --configure docker-ce
```

## sudoなしで実行させるには  
https://qiita.com/iganari/items/fe4889943f22fd63692a  

# サーバ側の設定  
## step1. Dockerfileをbuild  
buildでDockerfileからimageを作成する  

```bash
docker build -t tf-serving .  
```

※debconf: delaying package configuration, since apt-utils is not installed　というエラーは無視していい。  
※「.」はカレントディレクトリのDockerfileからimageを作成するということ。ここに普通にパスを渡せば、そこにあるDockerfileが実行される。  
※「-t」はimageに任意の名前をつけることができるオプション。今回は、「serving-example」という名前をつけている。  

## step2. Dockerの起動  
imageからコンテナを作成して、コンテナに入る  

```bash
docker run --name tf-serving -v `pwd`:/root/serving-kawase -p 8500:8500 -it tf-serving /bin/bash  
```

※--nameでコンテナに名前をつけている  
※-v [実環境上でコンテナにコピーしたいディレクトリパス]:[コンテナ上のコピー先ディレクトリパス]  
上記の例だとDockerfileに記述されている作業用ディレクトリにコピーしている。存在しないディレクトリだったら作成してコピーを勝手に行ってくれる。  
※-p [実環境上のポート]:[実環境上のポートと同期させたいコンテナ上のポート]  
上記の例だと実環境上の8500番ポートとコンテナ上の8500番ポートは同じという意味にするということでいいと思う。  
※-it はrunするときにとりあえずつけておく  
※最後の方にあるtf-servingは、runしたいbuild時に指定したimageの名前を指定する。  
※/bin/bash はコンテナ起動時にbashを起動してくれる。  

## step3. tensorflow-servingの実行  
tensorflow-servingは、すでにDockerfileにインストールするように記述されている。  

```bash
tensorflow_model_server --model_name='predict' --model_base_path=/root/serving-kawase/tmp/predict_price  
```

※「--model_name」は、リクエストを投げるプログラムで「request.model_spec.name」に記述した内容によって変える。  
※「--model_base_path」は、モデルが保存されているディレクトリを指定  

# クライアント側の設定  
開発環境で実行する場合、コンテナに入ったターミナルとは別のターミナルで以下を実行する  

## step1. 学習  
まずは、為替予測プログラムを実行して、学習を始める。  

```bash
python pred.py  
```

## step2. リクエスト  
tensorflow-servingを実行したサーバに向けてリクエストを行うプログラムを実行する。  

```bash
python pred_cli.py  
```

※step1で「--model_name」に指定した名前をspec.nameのとこに記述するのを忘れずに。parserにしたほうがいいかも  