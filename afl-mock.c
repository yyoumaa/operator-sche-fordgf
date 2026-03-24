#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <semaphore.h>
#include <unistd.h>

// 对应 Python 端 528 字节特征 + 8 字节 Reward
struct afl_bandit_shm {
    double global_ctx[2];
    double regions_ctx[16][4];
    double batch_max_reward;
};

// 对应 Python 端的 struct.pack('<ii', ...)
struct afl_bandit_decision {
    int chosen_region;
    int chosen_family;
};

int main() {
    printf("[C Mock] 初始化 IPC 资源...\n");

    // 1. 创建并打开共享内存
    int fd_c2py = shm_open("/shm_c2py", O_CREAT | O_RDWR, 0666);
    int fd_py2c = shm_open("/shm_py2c", O_CREAT | O_RDWR, 0666);
    ftruncate(fd_c2py, 1024);
    ftruncate(fd_py2c, 128);

    // 映射内存
    struct afl_bandit_shm *c2py_mem = mmap(NULL, 1024, PROT_READ | PROT_WRITE, MAP_SHARED, fd_c2py, 0);
    struct afl_bandit_decision *py2c_mem = mmap(NULL, 128, PROT_READ | PROT_WRITE, MAP_SHARED, fd_py2c, 0);

    // 2. 创建并初始化信号量 (初始值为 0)
    sem_t *sem_c_feat = sem_open("/sem_c_feat", O_CREAT, 0666, 0);
    sem_t *sem_c_batch = sem_open("/sem_c_batch", O_CREAT, 0666, 0);
    sem_t *sem_py_dec = sem_open("/sem_py_dec", O_CREAT, 0666, 0);

    printf("[C Mock] 等待 2 秒，请现在启动 Python 程序...\n");
    sleep(2);

    // 3. 模拟 Fuzzing 循环
    for (int loop = 1; loop <= 5; loop++) {
        printf("\n=== Fuzzing 回合 %d ===\n", loop);

        // [阶段 A] 模拟计算特征
        printf("[C Mock] 1. 计算特征中...\n");
        c2py_mem->global_ctx[0] = 0.5 + (loop * 0.1);
        c2py_mem->global_ctx[1] = 0.8;
        
        for (int i = 0; i < 16; i++) {
            c2py_mem->regions_ctx[i][0] = i * 0.1; // 模拟熵
            c2py_mem->regions_ctx[i][1] = 0.9;     // 模拟可见字符比例
            c2py_mem->regions_ctx[i][2] = 10.0;    // 历史 reward
            c2py_mem->regions_ctx[i][3] = 0.05;    // 历史覆盖率
        }

        // 通知 Python 特征已准备好
        sem_post(sem_c_feat);

        // [阶段 B] 等待 Python 决策
        printf("[C Mock] 2. 等待 Python 决策...\n");
        sem_wait(sem_py_dec);

        // 读取决策
        int target_region = py2c_mem->chosen_region;
        int target_family = py2c_mem->chosen_family;
        printf("[C Mock] 3. 收到决策 -> 变异区域: %d, 算子族: %d\n", target_region, target_family);

        // [阶段 C] 模拟执行变异并计算 Reward
        printf("[C Mock] 4. 执行 256 次定向变异...\n");
        usleep(50000); // 模拟变异耗时 50ms
        
        // 模拟产生一个 Reward（随机给一个正数或负数）
        c2py_mem->batch_max_reward = (target_region == 3) ? 50.0 : -5.0; // 假设选对区域 3 就得高分
        
        // 通知 Python Reward 已准备好
        sem_post(sem_c_batch);
        printf("[C Mock] 5. Reward (%.2f) 已发送。\n", c2py_mem->batch_max_reward);
        
        usleep(500000); // 休息半秒，方便肉眼观察
    }

    // 4. 清理资源（非常重要，否则下次运行会报错）
    printf("\n[C Mock] 测试结束，清理 IPC 资源...\n");
    sem_unlink("/sem_c_feat");
    sem_unlink("/sem_c_batch");
    sem_unlink("/sem_py_dec");
    shm_unlink("/shm_c2py");
    shm_unlink("/shm_py2c");

    return 0;
}