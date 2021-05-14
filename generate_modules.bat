
1.在docs文件夹（linux）文件夹下产生必要文件(首次运行)
sphinx-quickstart

2.# 产生目录
sphinx-apidoc -f -o ./src ../featurebox (按需要运行)

3.# 产生网页
make html

4.# 发生报错时候清除
make clean

#重复234直到满意