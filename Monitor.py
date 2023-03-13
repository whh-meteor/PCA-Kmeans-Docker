import time
 
import psutil
 
#根据进程名查找pid
def getInfo(pid):
    #方法1-使用Psutil第三方模块，此方法接受的pid为int-缺点记录的内存占用为内存峰值（进程所使用的最大物理内存）
    p = psutil.Process(pid)
    #获取当前进程的CPU占用
    #print(p.cpu_percent(interval=1),psutil.cpu_count())
    cpu = round(p.cpu_percent(interval=0.1)/psutil.cpu_count(),2)
    #获取当前进程的内存占用
    full_info = str(p.memory_full_info())
 
    rss = full_info.split("(")[-1].split(",")[0].split("=")[-1]
    uss = full_info.split("(")[-1].split(",")[-1].split("=")[-1].split(")")[0]
    vms = full_info.split("(")[-1].split(",")[1].split("=")[-1]
 
    #print(memory.split("(")[-1].split(",")[0].split("=")[-1])
    times = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    wbdata = times + "," + str(cpu) + "," + str(round(int(rss)/1024/1024,2))+ "," + str(round(int(uss)/1024/1024,2))+ "," + str(round(int(vms)/1024/1024,2))
    print(wbdata)
    return wbdata
 
#定时存进一个csv文档里
def saveinformation(information):
    filename = "savememory.csv"
    with open(filename,'a+') as f:
        "每隔10s去执行一次"
        f.write(information)
        f.write('\n')
 
def work():
    #输入pid和时间间隔
    pid = input("请输入进程的pid：")
    intervalTime = input("请输入性能监控间隔时间（建议0.3s以上）：")
    #先写csv文件头
    initdata = "Time,CPU(%),RSS(M-该进程实际使用物理内存（包含共享库占用的全部内存）),USS(M-进程独立占用的物理内存（不包含共享库占用的内存）),VMS(M-该进程使用的虚拟内存总量)"
    with open('savememory.csv','w') as f:
        f.write(initdata)
        f.write('\n')
    while 1 :
        info = getInfo(int(pid))
        saveinformation(info)
#定时执行
        time.sleep(float(intervalTime))
 