ps aux|ps -ef | grep multiprocessing| grep -v grep | awk '{print $2}'| xargs kill -9
