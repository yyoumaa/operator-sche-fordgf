#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <semaphore.h>
#include <unistd.h>
#include <time.h>

// ==================== 共享内存布局 ====================
// c2py: 特征区(536字节) + reward(8字节) + best_region(4字节) + best_family(4字节)
// 偏移: 0~535=特征, 536=reward, 544=best_region, 548=best_family

struct c2py_shm {
    double global_ctx[3];       // 3个全局特征
    double regions_ctx[16][4];  // 16个region，每个4个特征
    double batch_max_reward;    // 偏移536
    int    best_region;         // 偏移544
    int    best_family;         // 偏移548
};

// py2c: P_reg(16个double) + P_fam(16*5=80个double) = 96个double = 768字节
#define NUM_REGIONS 16
#define NUM_FAMILIES 5

struct py2c_shm {
    double P_reg[NUM_REGIONS];               // 16个概率
    double P_fam[NUM_REGIONS][NUM_FAMILIES]; // 16x5个概率
};

// ==================== 轮盘赌采样 ====================
int roulette_select(double *probs, int n) {
    double r = (double)rand() / RAND_MAX;
    double cumsum = 0.0;
    for (int i = 0; i < n; i++) {
        cumsum += probs[i];
        if (r <= cumsum) return i;
    }
    return n - 1; // 兜底
}

int main() {
    srand(time(NULL));
    printf("[C Mock] 初始化 IPC 资源...\n");

    // 1. 创建共享内存
    int fd_c2py = shm_open("/shm_c2py", O_CREAT | O_RDWR, 0666);
    int fd_py2c = shm_open("/shm_py2c", O_CREAT | O_RDWR, 0666);
    ftruncate(fd_c2py, 1024);
    ftruncate(fd_py2c, 1024); // 改成1024，容纳768字节的概率数组

    // 映射内存
    struct c2py_shm *c2py_mem = mmap(NULL, 1024,
        PROT_READ | PROT_WRITE, MAP_SHARED, fd_c2py, 0);
    struct py2c_shm *py2c_mem = mmap(NULL, 1024,
        PROT_READ | PROT_WRITE, MAP_SHARED, fd_py2c, 0);

    // 2. 创建信号量
    sem_t *sem_c_feat  = sem_open("/sem_c_feat",  O_CREAT, 0666, 0);
    sem_t *sem_c_batch = sem_open("/sem_c_batch",  O_CREAT, 0666, 0);
    sem_t *sem_py_dec  = sem_open("/sem_py_dec",   O_CREAT, 0666, 0);

    printf("[C Mock] 等待 2 秒，请现在启动 Python 程序...\n");
    sleep(2);

    // 3. 模拟 Fuzzing 循环
    for (int loop = 1; loop <= 50; loop++) {
        printf("\n=== Fuzzing 回合 %d ===\n", loop);

        // [阶段 A] 写入特征
        printf("[C Mock] 1. 写入特征...\n");

        // 3个全局特征
        c2py_mem->global_ctx[0] = 0.1 * loop;       // 静态进度，随时间增大
        c2py_mem->global_ctx[1] = 0.5 + 0.05 * loop; // 动态地位
        c2py_mem->global_ctx[2] = 0.02 * loop;       // reward动量

        // 16个region特征，故意让region 3和region 7更特别
        for (int i = 0; i < 16; i++) {
            if (i == 3) {
                // region 3：高熵、低可打印比，模拟"特殊区域"
                c2py_mem->regions_ctx[i][0] = 0.95; // 高熵
                c2py_mem->regions_ctx[i][1] = 0.1;  // 低可打印比
                c2py_mem->regions_ctx[i][2] = 0.5;
                c2py_mem->regions_ctx[i][3] = 0.05;
            } else if (i >= 8) {
                // region 8-15：超出文件长度，全0（无效region）
                c2py_mem->regions_ctx[i][0] = 0.0;
                c2py_mem->regions_ctx[i][1] = 0.0;
                c2py_mem->regions_ctx[i][2] = 0.0;
                c2py_mem->regions_ctx[i][3] = 0.0;
            } else {
                // 其他region：普通特征
                c2py_mem->regions_ctx[i][0] = 0.1 * i;
                c2py_mem->regions_ctx[i][1] = 0.8;
                c2py_mem->regions_ctx[i][2] = 0.5;
                c2py_mem->regions_ctx[i][3] = 0.05;
            }
        }

        // 通知 Python 特征已准备好
        sem_post(sem_c_feat);
        printf("[C Mock] 2. 等待 Python 概率分布...\n");

        // [阶段 B] 等待 Python 的概率分布
        sem_wait(sem_py_dec);

        // 打印收到的概率（只打有效region）
        printf("[C Mock] 3. 收到概率分布:\n");
        printf("  P_reg: ");
        for (int i = 0; i < 8; i++) {  // 只打前8个（后8个是无效region）
            printf("r%d=%.3f ", i, py2c_mem->P_reg[i]);
        }
        printf("\n");

        // [阶段 C] 模拟256次havoc，按概率采样
        printf("[C Mock] 4. 执行 256 次轮盘赌变异...\n");

        double best_reward = -1e9;
        int best_region = 0;
        int best_family = 0;

        for (int step = 0; step < 256; step++) {
            // 按P_reg轮盘赌选region
            int r = roulette_select(py2c_mem->P_reg, NUM_REGIONS);

            // 按P_fam[r]轮盘赌选family
            int f = roulette_select(py2c_mem->P_fam[r], NUM_FAMILIES);

            // 模拟reward：region==3 且 family==2 时得高分
            double step_reward = 0.0;
            if (r == 3 && f == 2) {
                step_reward = 40.0 + (rand() % 20);  // 40~60分
            } else if (r == 3) {
                step_reward = 10.0 + (rand() % 10);  // 10~20分
            } else {
                step_reward = (rand() % 5) - 2;      // -2~3分，噪声
            }

            // 记录最佳
            if (step_reward > best_reward) {
                best_reward  = step_reward;
                best_region  = r;
                best_family  = f;
            }
        }

        // [阶段 D] 写回reward和最佳动作
        c2py_mem->batch_max_reward = best_reward;
        c2py_mem->best_region      = best_region;
        c2py_mem->best_family      = best_family;

        printf("[C Mock] 5. batch结束 -> best_reward=%.2f "
               "best_region=%d best_family=%d\n",
               best_reward, best_region, best_family);

        // 通知 Python 可以读reward了
        sem_post(sem_c_batch);

        usleep(300000); // 休息300ms，方便观察日志
    }

    // 4. 清理
    printf("\n[C Mock] 测试结束，清理资源...\n");
    munmap(c2py_mem, 1024);
    munmap(py2c_mem, 1024);
    sem_close(sem_c_feat);
    sem_close(sem_c_batch);
    sem_close(sem_py_dec);
    sem_unlink("/sem_c_feat");
    sem_unlink("/sem_c_batch");
    sem_unlink("/sem_py_dec");
    shm_unlink("/shm_c2py");
    shm_unlink("/shm_py2c");

    return 0;
}