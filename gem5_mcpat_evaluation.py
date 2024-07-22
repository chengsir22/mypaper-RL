import os
import logging
import math
import time
import pandas as pd
import subprocess
from getevaluation import getevaluation

# sample = {
#     "core": 4,  # 1-16
#     "l1i_size": 8,
#     "l1d_size": 8,
#     "l2_size": 10,
#     "l1d_assoc": 2,
#     "l1i_assoc": 2,
#     "l2_assoc": 2,
#     "sys_clock": 2.8,
#     "latency": 0,
#     "area": 0,
#     "power": 0,
# }

logger = logging.getLogger("logger")


def run_simulation(config):
    cmd = f"./gem5/build/X86/gem5.fast -d 'out/simout' -re ./gem5/configs/deprecated/example/fs.py --script=./parsec-image/benchmark_src/canneal_{config['core']}c_simdev.rcS -F 5000000000 \
           --cpu-type=TimingSimpleCPU --num-cpus={config['core']} --sys-clock='{config['sys_clock']}GHz' --caches --l2cache   --l1d_size='{config['l1d_size']}kB' --l1i_size='{config['l1i_size']}kB' --l2_size='{config['l2_size']}kB' --l1d_assoc={config['l1d_assoc']} --l1i_assoc={config['l1i_assoc']} --l2_assoc={config['l2_assoc']} \
           --kernel=./parsec-image/system/binaries/x86_64-vmlinux-2.6.28.4-smp --disk-image=./parsec-image/system/disks/x86root-parsec.img"
    os.system(cmd)

    with open(f"./out/simout/stats.txt") as f:
        contents = f.read().split("---------- Begin Simulation Statistics ----------")
    for i, content in enumerate(contents):
        with open(f"./out/simout/{i}.txt", "w") as f:
            f.write(content if i == 0 else "---------- Begin Simulation Statistics ----------" + content)


def run_mcpat(config):
    try:
        if os.path.exists(f"./out/simout/3.txt"):
            # Gem5ToMcPAT
            subprocess.run(
                ["python3", "./cMcPAT/Scripts/GEM5ToMcPAT.py", f"./out/simout/3.txt", f"./out/simout/config.json",
                 f"./cMcPAT/mcpat/ProcessorDescriptionFiles/x86_AtomicSimpleCPU_template_core_{config['core']}.xml",
                 "-o", f"./out/simout/test.xml"])
            # startMcPAT
            with open(f"./out/simout/test2.log", "w") as file_output:
                subprocess.run(["./cMcPAT/mcpat/mcpat", "-infile", f"./out/simout/test.xml", "-print_level", "5"],
                               stdout=file_output)
            # get metrics
            metrics = getevaluation(f"./out/simout/test2.log", f"./out/simout/3.txt")
            return metrics
        else:
            logger.warning("Error: McPAT input file not found.")
            return None
    except Exception as e:
        logger.warning(f"Error during McPAT execution: {e}")
        return None
    finally:
        # clear
        for path in [f"./out/simout/3.txt", f"./out/simout/test2.log", f"./out/simout/test.xml"]:
            if os.path.exists(path):
                os.remove(path)


def evaluation(status):
    config = get_system_config(status)
    for key, value in config.items():
        logger.info(f"{key}: {value}")

    start_time = time.time()
    alldata, data = read_csv()

    row = find_row(data, alldata, status)
    # 检查结果是否非空，然后添加到data中
    if row is not None:
        data = pd.concat([data, pd.DataFrame([row])], ignore_index=True)
        data.to_csv("./out/data/data.csv", index=False)
        logger.info(f"skip")
        return row[-3:].to_dict()
    else:
        logger.warning("No matching row found. so let's start Gem5 simulator")

        logger.info(f"+++++++++++  start Gem5 simulator +++++++++++")
        run_simulation(config)
        logger.info(f"++++++++++++ end Gem5 simulator ++++++++++++++")

        logger.info(f"+++++++++++++++++++ start McPAT ++++++++++++++++++++++")
        metrics = run_mcpat(config)
        logger.info(f"+++++++++++++++++++  end McPAT ++++++++++++++++++++")

        end_time = time.time()
        logger.info(f"Total execution time: {end_time - start_time}")

        if metrics:
            logger.info(f"Evaluation success")
            logger.info(
                f"latency: {metrics['latency']} sec , area: {metrics['area']} mm^2 , energy: {metrics['energy']} mJ , power: {metrics['power']} W")
            status["latency"] = round(metrics["latency"], 6)
            status["area"] = round(metrics["area"], 3)
            status["power"] = round(metrics["power"], 3)
        else:
            status["latency"] = 0
            status["area"] = 0
            status["power"] = 0
            logger.info(f"Evaluation failed.")

        data = pd.concat([data, pd.DataFrame([status])], ignore_index=True)
        data.to_csv(f"./out/data/data.csv", index=False)
        return metrics


def get_system_config(status):
    config = {
        "core": str(status["core"]),
        "l1i_size": str(int(math.pow(2, int(status["l1i_size"])))),
        "l1d_size": str(int(math.pow(2, int(status["l1d_size"])))),
        "l2_size": str(int(math.pow(2, int(status["l2_size"])))),
        "l1d_assoc": str(int(math.pow(2, status["l1d_assoc"]))),
        "l1i_assoc": str(int(math.pow(2, status["l1i_assoc"]))),
        "l2_assoc": str(int(math.pow(2, status["l2_assoc"]))),
        "sys_clock": str(status["sys_clock"]),
    }
    return config


# -----------------读取csv-----------------------
def read_csv():
    alldata = f"./out/data/alldata.csv"
    data = f"./out/data/data.csv"
    return read_csv_must(alldata), read_csv_must(data)


def read_csv_must(file_path):
    # 判断文件是否存在
    if os.path.exists(file_path):
        # 文件存在，读取文件
        dataframe = pd.read_csv(file_path)
    else:
        # 文件不存在，创建一个空的DataFrame并保存为CSV文件
        columns = ['core', 'l1i_size', 'l1d_size', 'l2_size', 'l1d_assoc', 'l1i_assoc', 'l2_assoc', 'sys_clock',
                   'latency', 'area', 'power']
        dataframe = pd.DataFrame(columns=columns)
        dataframe.to_csv(file_path, index=False)
    return dataframe


def find_row(data, alldata, status):
    # 将status字典转换为DataFrame中行的形式
    status_series = pd.Series(status)

    # 检查data中每一行的前8列
    for index, row in data.iterrows():
        if row[:8].equals(status_series):
            return row  # 找到匹配的行，返回这一行

    # 如果在data中没有找到，检查alldata
    for index, row in alldata.iterrows():
        if row[:8].equals(status_series):
            return row  # 找到匹配的行，返回这一行

    # 如果两个DataFrame都没有找到匹配的行，返回None
    return None
