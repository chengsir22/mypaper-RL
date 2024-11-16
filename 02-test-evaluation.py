from utils import *

log_file = f"./out/log/02_test_evaluation.log"

logger = get_logger(log_file=log_file)

checkdict = dict()
checkdict['core'] = 4
checkdict['l1i_size'] = 8
checkdict['l1d_size'] = 8
checkdict['l2_size'] = 8
checkdict['l1d_assoc'] = 2
checkdict['l1i_assoc'] = 2
checkdict['l2_assoc'] = 2
checkdict['sys_clock'] = 2.8

metrics = evaluation(checkdict)

print(metrics)
print("02-test-evaluation success")

# 4,8,8,8,2,2,2,2.8,0.00019,57.049,7.896

# core（核心数）:

# 小型处理器：1-4核。
# 中等型处理器：4-8核。
# 大型处理器：8核以上。
# l1i_size（一级指令缓存大小）:

# 小型处理器：通常在4KB到16KB之间。
# 中等型处理器：通常在16KB到64KB之间。
# 大型处理器：通常在64KB到128KB之间。
# l1d_size（一级数据缓存大小）:

# 小型处理器：通常在4KB到16KB之间。
# 中等型处理器：通常在16KB到64KB之间。
# 大型处理器：通常在64KB到128KB之间。
# l2_size（二级缓存大小）:

# 小型处理器：通常在64KB到256KB之间。
# 中等型处理器：通常在256KB到512KB之间。
# 大型处理器：通常在512KB到2MB之间。
# l1d_assoc（一级数据缓存关联度）和 l1i_assoc（一级指令缓存关联度）:

# 通常在2到8之间，表示缓存的相联度。较高的关联度可以提高缓存命中率，但会增加硬件复杂性。
# l2_assoc（二级缓存关联度）:

# 通常在4到16之间，表示二级缓存的相联度。
# sys_clock（系统时钟速度）:

# 小型处理器：通常在1.0GHz到3.0GHz之间。
# 中等型处理器：通常在3.0GHz到4.5GHz之间。
# 大型处理器：通常在4.5GHz以上。


# 面积 (Area):

# 小型处理器：通常在1平方毫米到10平方毫米之间。
# 中等型处理器：通常在10平方毫米到500平方毫米之间。
# 大型处理器：通常在500平方毫米以上。
# 功耗 (Power):

# 小型处理器：通常在1瓦特（W）到10瓦特之间。
# 中等型处理器：通常在10瓦特到50瓦特之间。
# 大型处理器：通常在50瓦特-200W以上。
