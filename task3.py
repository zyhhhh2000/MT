import pandas as pd
import numpy as np
import time
import pyopencl as cl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import multiprocessing as mp
import pyopencl as cl
from scipy.integrate import odeint, solve_ivp
from verlet import verlet_cython

G = 6.67430e-11  # 引力常数
au = 1.496e11  # 天文单位，米
day = 86400  # 一天的秒数
RE = 1.48e11  # 参考距离
ME = 5.965e24  # 参考质量


def gravitation(t, y, *args):
    n = len(args[0])
    x, y, vx, vy = y[:n], y[n:2 * n], y[2 * n:3 * n], y[3 * n:]
    dxdt = vx
    dydt = vy
    dvxdt = np.zeros_like(vx)
    dvydt = np.zeros_like(vy)

    for i in range(n):
        for j in range(n):
            if i != j:
                r = np.sqrt((x[j] - x[i]) ** 2 + (y[j] - y[i]) ** 2)
                dvxdt[i] += G * args[0][j] * (x[j] - x[i]) / r ** 3
                dvydt[i] += G * args[0][j] * (y[j] - y[i]) / r ** 3

    return np.concatenate([dxdt, dydt, dvxdt, dvydt])

def verlet_python(N, dt, m, x, y, v_x, v_y):
    ts = np.arange(0, N * dt, dt)
    xs, ys = [], []

    # 初始化加速度数组
    accx_1 = np.zeros(len(m))
    accy_1 = np.zeros(len(m))

    for _ in ts:
        # 计算粒子之间的距离和相对位置
        x_ij = x - x.reshape(len(m), 1)
        y_ij = y - y.reshape(len(m), 1)
        r_ij = np.sqrt(x_ij ** 2 + y_ij ** 2)

        # 计算粒子之间的引力
        accx = np.zeros(len(m))
        accy = np.zeros(len(m))
        for i in range(len(m)):
            for j in range(len(m)):
                if i != j:
                    accx[i] += m[j] * x_ij[i, j] / r_ij[i, j] ** 3
                    accy[i] += m[j] * y_ij[i, j] / r_ij[i, j] ** 3

        # 更新速度和位置
        v_x += 0.5 * (accx + accx_1) * dt
        v_y += 0.5 * (accy + accy_1) * dt
        x += v_x * dt + 0.5 * accx * dt ** 2
        y += v_y * dt + 0.5 * accy * dt ** 2

        # 保存位置数据
        xs.append(x.tolist())
        ys.append(y.tolist())

        # 更新加速度
        accx_1 = accx.copy()
        accy_1 = accy.copy()

    return np.array(xs), np.array(ys)

def compute_acceleration(args):
    m, x, y, acc_x, acc_y, i = args
    for j in range(len(m)):
        if i != j:
            x_diff = x[j] - x[i]
            y_diff = y[j] - y[i]
            r = np.sqrt(x_diff ** 2 + y_diff ** 2)
            force = G * m[j] / r ** 3
            acc_x[i] += force * x_diff
            acc_y[i] += force * y_diff

def compute_velocity(args):
    v_x, v_y, acc_x, acc_y, dt = args
    v_x += acc_x * dt
    v_y += acc_y * dt

def compute_position(args):
    x, y, v_x, v_y, acc_x, acc_y, dt = args
    x += v_x * dt + 0.5 * acc_x * dt ** 2
    y += v_y * dt + 0.5 * acc_y * dt ** 2

def verlet_multi(N, dt, m, x, y, v_x, v_y):
    ts = np.arange(0, N * dt, dt)
    xs, ys = [], []

    # Initialize acceleration arrays
    acc_x = np.zeros_like(x)
    acc_y = np.zeros_like(y)

    # Create pool of threads
    pool = mp.Pool()

    for t in ts:
        # Create arguments for each task
        args_acceleration = [(m, x, y, acc_x, acc_y, i) for i in range(len(m))]
        args_velocity = [(v_x, v_y, acc_x, acc_y, dt)] * len(m)
        args_position = [(x, y, v_x, v_y, acc_x, acc_y, dt)] * len(m)

        # Compute acceleration, velocity, and position using the thread pool
        pool.map(compute_acceleration, args_acceleration)
        pool.map(compute_velocity, args_velocity)
        pool.map(compute_position, args_position)

        xs.append(x.tolist())
        ys.append(y.tolist())

        # Reset acceleration arrays
        acc_x.fill(0)
        acc_y.fill(0)

    xs = np.array(xs)
    ys = np.array(ys)

    pool.close()
    pool.join()

    return xs, ys

def verlet_opencl(N, dt, m, x, y, v_x, v_y):
    # 初始化 OpenCL
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context)

    # 创建内存缓冲区
    m_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=m)
    x_buf = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=x)
    y_buf = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=y)
    v_x_buf = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=v_x)
    v_y_buf = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=v_y)

    # 创建内核
    kernel_code = """
    __kernel void verlet_opencl(__global const double* m,
                                __global double* x,
                                __global double* y,
                                __global double* v_x,
                                __global double* v_y,
                                const double dt,
                                const int N) {
        int gid = get_global_id(0);

        double accx_1 = 0.0;
        double accy_1 = 0.0;

        for (int step = 0; step < N; step++) {
            double accx = 0.0;
            double accy = 0.0;
            for (int i = 0; i < N; i++) {
                if (gid != i) {
                    double x_ij = x[gid] - x[i];
                    double y_ij = y[gid] - y[i];
                    double r_ij = sqrt(x_ij * x_ij + y_ij * y_ij);
                    accx += m[i] * x_ij / (r_ij * r_ij * r_ij);
                    accy += m[i] * y_ij / (r_ij * r_ij * r_ij);
                }
            }

            v_x[gid] += 0.5 * (accx + accx_1) * dt;
            v_y[gid] += 0.5 * (accy + accy_1) * dt;
            x[gid] += v_x[gid] * dt + 0.5 * accx * dt * dt;
            y[gid] += v_y[gid] * dt + 0.5 * accy * dt * dt;

            accx_1 = accx;
            accy_1 = accy;
        }
    }
    """
    program = cl.Program(context, kernel_code).build()

    # 执行内核
    program.verlet_opencl(queue, (len(m),), None, m_buf, x_buf, y_buf, v_x_buf, v_y_buf, np.float64(dt), np.int32(N))

    # 从 GPU 获取结果
    cl.enqueue_copy(queue, x, x_buf).wait()
    cl.enqueue_copy(queue, y, y_buf).wait()

    return x, y




def main():
    # 设置模拟参数
    N = 100 #1000 100
    dt = 3600*24 #0.1 3600*24
    nn = [9] #100,200,400 #10
    times = len(nn)
    ttimes = 1 #1 #3
    t_py = np.zeros(times)
    t_mu = np.zeros(times)
    t_cy = np.zeros(times)
    t_op = np.zeros(times)

    for i in range(times):
        for j in range(ttimes):
            # 创建粒子数据-N
            # m_0 = np.random.rand(nn[i]) * 1000 * 5.965e24 * 6.67e-11
            # r_0 = np.random.rand(nn[i]) * 10 * RE
            # theta_0 = np.random.rand(nn[i]) * np.pi * 2
            # v_0 = np.random.rand(nn[i]) * 100 * 1000
            # x_0 = r_0 * np.cos(theta_0)
            # y_0 = r_0 * np.sin(theta_0)
            # v_x_0 = -v_0 * np.sin(theta_0)
            # v_y_0 = v_0 * np.cos(theta_0)

            # 太阳系
            m_0 = np.array([3.32e5, 0.055, 0.815, 1,
                            0.107, 317.8, 95.16, 14.54, 17.14]) * ME * 6.67e-11
            r_0 = np.array([0, 0.387, 0.723, 1, 1.524, 5.203,
                          9.537, 19.19, 30.7]) * RE
            theta_0 = [0.90579977, 4.76568695, 1.34869972, 6.02969388, 2.24714959, 3.45095948,
                     3.41281759, 4.32174632, 2.33019222]
            x_0 = r_0 * np.cos(theta_0)
            y_0 = r_0 * np.sin(theta_0)
            v_0 = np.array([0, 47.89, 35.03, 29.79,
                          24.13, 13.06, 9.64, 6.81, 5.43]) * 1000
            v_x_0 = -v_0 * np.sin(theta_0)
            v_y_0 = v_0 * np.cos(theta_0)
            data = {
                'm': m_0,
                'r': r_0,
                'theta': theta_0,
                'v': v_0,
                'x': x_0,
                'y': y_0,
                'v_x': v_x_0,
                'v_y': v_y_0
            }
            df = pd.DataFrame(data)

            # 将 DataFrame 写入 CSV 文件
            df.to_csv('particle_data.csv', index=False)
            df = pd.read_csv('particle_data.csv')


            if nn[i] < 50:
                # 测试原始 ode 方法的运行时间
                y0 = np.concatenate([df['x'], df['y'], df['v_x'], df['v_y']])

                t_span = (0, N * dt)
                t_eval = np.linspace(t_span[0], t_span[1], N)

                start = time.time()
                sol = solve_ivp(gravitation, t_span, y0, args=(df['m'],), t_eval=t_eval)
                end = time.time()
                print("time ode:", end - start)
                x_results = sol.y[:nn[i]]
                y_results = sol.y[nn[i]:2 * nn[i]]




            # 测试原始 Python 方法的运行时间
            x_py = df['x'].values
            y_py = df['y'].values
            v_x_py = df['v_x'].values
            v_y_py = df['v_y'].values

            start = time.time()
            xsp,ysp = verlet_python(N, dt, m_0, x_py, y_py, v_x_py, v_y_py)
            end = time.time()
            t_py[i] = t_py[i] + end - start
            print("time Python:", end - start)
            # print(xsp)

            # 测试原始 multi 方法的运行时间
            x_mu = df['x'].values
            y_mu = df['y'].values
            v_x_mu = df['v_x'].values
            v_y_mu = df['v_y'].values

            start = time.time()
            xsm, ysm = verlet_multi(N, dt, m_0, x_mu, y_mu, v_x_mu, v_y_mu)
            end = time.time()
            t_mu[i] = t_mu[i] + end - start
            print("time multi:", end - start)
            # print(xsp)


            # 测试 Cython 方法的运行时间
            # 初始化粒子位置数组
            xsc = np.zeros((N, nn[i]))
            ysc = np.zeros((N, nn[i]))

            x_cy = df['x'].values
            y_cy = df['y'].values
            v_x_cy = df['v_x'].values
            v_y_cy = df['v_y'].values

            start = time.time()
            verlet_cython(N, dt, m_0, x_cy, y_cy, v_x_cy, v_y_cy, xsc, ysc)
            end = time.time()
            t_cy[i] = t_cy[i] + end - start
            print("time Cython:", end - start)
            # print(xsc)

            # 测试 Opencl 方法的运行时间
            x_op = df['x'].values
            y_op = df['y'].values
            v_x_op = df['v_x'].values
            v_y_op = df['v_y'].values

            start = time.time()
            x_opencl, y_opencl = verlet_opencl(N, dt, m_0, x_op, y_op, v_x_op, v_y_op)
            end = time.time()
            t_op[i] = t_op[i] + end - start
            print("time Opencl:", end - start)

    ### 50 1 1000 36*2400
    #动态图
    def animate(n):
        for i in range(nn[0]):
            traces[i].set_data(xsp[:n, i], ysp[:n, i])
            pts[i].set_data([xsp[n, i]], [ysp[n, i]])
        # k_text.set_text(textTemplate % (ts[n]/3600/24))
        return traces + pts

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(xlim=(-20 * RE, 20 * RE), ylim=(-20 * RE, 20 * RE))
    ax.grid()
    traces = [ax.plot([], [], '-', lw=0.5)[0] for _ in range(nn[0])]
    pts = [ax.plot([], [], marker='o')[0] for _ in range(nn[0])]
    # textTemplate = 't = %.3f days\n'
    ani = FuncAnimation(fig, animate,
                        N, interval=100, blit=True)
    plt.show()
    ani.save("python.gif",writer='pillow')

    # ### 10 1
    # # 绘制对比结果
    # plt.figure(figsize=(8, 6))
    # plt.scatter(x_results[:,-1], y_results[:,-1], label='odeint', alpha=0.5)
    # plt.scatter(xsp[-1,:], ysp[-1,:], label='Python', alpha=0.5)
    # plt.scatter(xsc[-1,:], ysc[-1,:], label='Cython', alpha=0.5)
    # plt.scatter(xsm[-1,:], ysm[-1,:], label='multi', alpha=0.5)
    # plt.scatter(x_opencl, y_opencl, label='opencl', alpha=0.5)
    # plt.xlabel('X Position')
    # plt.ylabel('Y Position')
    # plt.title('Final Positions of Planets')
    # plt.legend()
    # plt.show()
    # print(x_results[:,-1])
    # print(xsp[-1, :])
    # print(xsc[-1,:])
    # print(xsm[-1, :])
    # print(x_opencl)
    # # 误差图
    # error_py = 0
    # error_mu = 0
    # error_cy = 0
    # error_op = 0
    # for k in range(len(x_results[:,-1])):
    #     error_py = error_py + np.sqrt((x_results[k,-1]-xsp[-1,k]) **2 + (y_results[k,-1]-ysp[-1,k]) **2)
    #     error_mu = error_mu + np.sqrt((x_results[k,-1]-xsm[-1,k]) **2 + (y_results[k,-1]-ysm[-1,k]) **2)
    #     error_cy = error_cy + np.sqrt((x_results[k, -1] - xsc[-1, k]) ** 2 + (y_results[k, -1] - ysc[-1, k]) ** 2)
    #     error_op = error_op + np.sqrt((x_results[k, -1] - x_opencl[k]) ** 2 + (y_results[k, -1] - y_opencl[k]) ** 2)
    # plt.figure(figsize=(8, 6))
    # labels = ['python', 'multi', 'cython', 'opencl']
    # # 画柱状图
    # plt.bar(labels, [error_py/nn[0],error_mu/nn[0],error_cy/nn[0],error_op/nn[0]], color=['blue', 'red', 'green', 'yellow'])
    # # 添加标题和标签
    # plt.title('графики погрешностей')
    # plt.xlabel('метод')
    # plt.ylabel('погрешности')
    # # 显示图形
    # plt.show()
    # print("погрешностей:",error_py/nn[0],error_mu/nn[0],error_cy/nn[0],error_op/nn[0])


    #### 3 100 200 400
    #计算时间图
    plt.figure(figsize=(8, 6))
    # 绘制点
    plt.scatter(nn, t_py / ttimes, label='python', color='blue')
    plt.scatter(nn, t_mu / ttimes, label='multi', color='red')
    plt.scatter(nn, t_cy / ttimes, label='cython', color='green')
    plt.scatter(nn, t_op / ttimes, label='opencl', color='yellow')
    # 绘制线
    plt.plot(nn, t_py / ttimes, label='python', color='blue')
    plt.plot(nn, t_mu / ttimes, label='multi', color='red')
    plt.plot(nn, t_cy / ttimes, label='cython', color='green')
    plt.plot(nn, t_op / ttimes, label='opencl', color='yellow')
    # 添加标签和标题
    plt.xlabel('N')
    plt.ylabel('time')
    plt.title('времени работы методов')
    # 添加图例
    plt.legend()
    # 显示图形
    plt.show()

    # 加速度图
    plt.figure(figsize=(8, 6))
    # 绘制点
    plt.scatter(nn, t_py / t_mu, label='multi', color='red')
    plt.scatter(nn, t_py / t_cy, label='cython', color='green')
    # plt.scatter(nn, t_py / t_op, label='opencl', color='yellow')
    # 绘制线
    plt.plot(nn, t_py / t_mu, label='multi', color='red')
    plt.plot(nn, t_py / t_cy, label='cython', color='green')
    # plt.plot(nn, t_py / t_op, label='opencl', color='yellow')
    # 添加标签和标题
    plt.xlabel('N')
    plt.ylabel('t_py/t_i')
    plt.title('ускорения по сравнению с последовательной версией метода Верле')
    # 添加图例
    plt.legend()
    # 显示图形
    plt.show()




if __name__ == '__main__':
    main()

#python setup.py build_ext

