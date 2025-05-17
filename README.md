# Genshin_zoo

## Description

本项目旨在解决一个基于网格的拼图放置优化问题，目标是通过选择和放置一系列具有不同形状和属性的拼图块，最大化在固定尺寸网格上的总得分。项目采用进化算法（Genetic Algorithm, GA）结合局部搜索（Local Search, LS）技术来探索解空间。

## Requirements

*   Python 3.x
*   `deap`
*   `numpy`
*   `tqdm`

## Usage

1.  克隆项目仓库：
    ```bash
    git clone https://github.com/xu2243/Genshin_zoo.git
    cd Genshin_zoo
    ```
2.  安装依赖：
    ```bash
    pip install deap numpy tqdm
    ```
3.  运行求解器：
    *   运行基于智能贪婪放置的遗传算法：
        ```bash
        python GA_greedy.py
        ```
    *   运行基于严格顺序放置的模因算法：
        ```bash
        python GA_strict.py
        ```
4.  程序将持续运行并输出进度。按下 `Ctrl+C` 停止程序。
5.  最佳解将被记录在 `best_solution_log.txt` 或 `best_solution_log_memetic_strict.txt` 文件中。

## Project Structure

*   `GA_greedy.py`: 实现基于智能贪婪放置策略的遗传算法，结合周期性局部搜索。
*   `GA_strict.py`: 实现基于严格顺序放置策略的模因算法，结合周期性局部搜索应用于精英个体。
*   `best_solution_log.txt`: `GA_greedy.py` 记录最佳解的日志文件。
*   `best_solution_log_memetic_strict.txt`: `GA_strict.py` 记录最佳解的日志文件。

## Algorithm

本项目实现了两种基于进化计算的求解方法：

1.  **GA_greedy.py:** 采用遗传算法，个体编码仅包含拼图块的选择和旋转。拼图块的实际放置位置在适应度评估时通过一个智能贪婪策略动态确定。算法周期性地对当前最佳个体进行局部搜索，通过移除部分已放置拼图并重新贪婪放置或进行平移操作来尝试改进布局。
2.  **GA_strict.py:** 采用模因算法，个体编码包含拼图块的选择、旋转和**提议的放置位置**。拼图块的放置严格按照个体编码中的顺序进行，冲突的拼图块将被跳过。局部搜索直接作用于个体编码（基因型），通过修改提议的位置、旋转或调整顺序来探索邻域解。该版本利用多核并行计算加速评估和局部搜索过程。

## License

本项目采用 MIT License。
