import os
import random

data_path = "data\\LP\\ACM"

def static_line():
    link = {}
    link_type = None

    with open(os.path.join(data_path, "link.dat"),"r") as f:
        for line in f:
            th = line.split("\t")
            head_node_id, tail_node_id, link_type_l = int(th[0]), int(th[1]), int(th[2])
            if link_type is not link_type_l:
                link_type = link_type_l
                link[link_type] = 1
            else:
                link[link_type] += 1

def neg_sample():
    num = 10000
    for i in range(num):
        print(str(random.randint(0,2900))+"\t"+str(random.randint(0,2900))+"\t"+"2")

neg_sample()