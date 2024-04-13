#first import m5 library and all SimObjects that we have compiled
import os
import m5
from m5.objects import *

#the system object will be the father of other objects 
system = System()

#we donnot care about the voltage_domain,so just use the default value
system.clk_domain = SrcClockDomain()
system.clk_domain.clock = '1GHz'
system.clk_domain.voltage_domain = VoltageDomain()

#use timing mode for memory simulation,we will almost always use this
system.mem_mode = 'timing'
system.mem_ranges = [AddrRange('1024MB')]

#use TimingSimpleCPU which excute every instruction in a single cycle
system.cpu = TimingSimpleCPU()

#create the system-wide memory bus
system.membus = SystemXBar()

#connect the cache ports on the CPU to memory bus
#the ports can be on either side of = ande the same connection will be made 
system.cpu.icache_port = system.membus.cpu_side_ports
system.cpu.dcache_port = system.membus.cpu_side_ports

#these four lines are to make sure that our system will function correctly
#connecting the PIO and interrupt ports to the memory bus is an x86-specific requirement
system.cpu.createInterruptController()
system.cpu.interrupts[0].pio = system.membus.mem_side_ports
system.cpu.interrupts[0].int_requestor = system.membus.cpu_side_ports
system.cpu.interrupts[0].int_responder = system.membus.mem_side_ports

system.system_port = system.membus.cpu_side_ports

#we need a memory controller and connect it to the bus,here we use DDR3 controller
#the memory range will be responsible for the entire memory range of our system
system.mem_ctrl = MemCtrl()
system.mem_ctrl.dram = DDR3_1600_8x8()
system.mem_ctrl.dram.range = system.mem_ranges[0]
system.mem_ctrl.port = system.membus.mem_side_ports

#now we have finished instantiating our simulated system

#let us set the CPU workload
# Here we set the X86 "hello world" binary. With other ISAs you must specify
# workloads compiled to those ISAs. Other "hello world" binaries for other ISAs
# can be found in "tests/test-progs/hello".
thispath = os.path.dirname(os.path.realpath(__file__))
binary = os.path.join(
    thispath,
    "gem5",
    "tests/test-progs/hello/bin/x86/linux/hello",
)
system.workload = SEWorkload.init_compatible(binary)

#we have to create the process to run out CPU
process = Process()
process.cmd = [binary]
system.cpu.workload = process
system.cpu.createThreads()

root = Root(full_system = False, system = system)
m5.instantiate()

print('Beginning simulation!')
exit_event = m5.simulate()

print('Exiting @ tick {} because {}'
        .format(m5.curTick(),exit_event.getCause()))