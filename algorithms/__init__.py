from algorithms.genetic import GeneticScheduler
from algorithms.tabu import TabuScheduler
from algorithms.ga_cpsat import GACpsatScheduler

# 算法註冊表：key = 前端選單值，value = 算法實例
# 未來新增算法只需：
#   from algorithms.my_algo import MyScheduler
#   REGISTRY['my_algo'] = MyScheduler()
REGISTRY: dict = {
    "ga_cpsat": GACpsatScheduler(),
    "genetic": GeneticScheduler(),
    "tabu": TabuScheduler(),
}
