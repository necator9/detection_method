rsync -avzh --delete --exclude '.git' --exclude '.idea' fila@192.168.7.1:/home/fila/PycharmProjects/detection_module /root/
./module_main.py
rsync -avzh --delete ./*.db fila@192.168.7.1:/home/fila/