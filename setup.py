import os

# gem5测试脚本
# os.system("./gem5/build/X86/gem5.opt ./01-gem5-test.py")
# os.system("./gem5/build/X86/gem5.opt ./gem5/configs/tutorial/part1/two_level.py")
# os.system("./gem5/build/X86/gem5.opt ./gem5/configs/tutorial/part2/two_level.py --clk='3GHz' --l1i_size='32kB' --l1d_size='256kB' --cache_block=16 -- l2_size='1024kB'")
# os.system("./gem5/build/X86/gem5.opt ./gem5/configs/deprecated/example/se.py --cmd=./gem5/tests/test-progs/hello/bin/x86/linux/hello --cpu-type=TimingSimpleCPU --num-cpus=4 --l1d_size=64kB --l1i_size=16kB --caches")

# test评估模块 
os.system("python3 ./02-test-evaluation.py")

# random
# os.system("python3 ./03-random.py")

# reinforce
# os.system("python3 ./04_reinforce.py")

# reinforce
# os.system("python ./08-reinforce-ljf.py")


# dqn
# os.system("python ./09-dqn.py")
