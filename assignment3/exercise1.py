
import numpy as np
import matplotlib.pyplot as plt
ph1=0.25
ph2=0.55
reward=[]
for i in range(1,100):
    reward.append(0)


while True:
    D=0
    for s in range(1,100):
        v=