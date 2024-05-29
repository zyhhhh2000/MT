from mshr import *
from fenics import *
import csv
import imageio.v2 as imageio
import numpy as np
import matplotlib
import matplotlib.tri as tri
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase


def calculate(V, exact_u, num_a, f, h, g, u_n):
    def is_left_boundary(x, on_boundary):
        return on_boundary and x[0] < 0

    bc = DirichletBC(V, h, is_left_boundary)
    u = TrialFunction(V)
    v = TestFunction(V)
    a = (dot(grad(u), grad(v)) + (1 / (dt * num_a)) * u * v) * dx
    L = (u_n / (dt * num_a) + f / num_a) * v * dx + g * v * ds
    u = Function(V)
    solve(a == L, u, bc)
    error_L2 = errornorm(exact_u, u, 'L2')
    vertex_values_exact_u = exact_u.compute_vertex_values(mesh)
    vertex_values_u = u.compute_vertex_values(mesh)
    error_max = np.max(np.abs(vertex_values_exact_u - vertex_values_u))

    # print('норма L2  =', error_L2)
    # print('максимум-норма =', error_max)
    return u, error_L2, error_max


def findrange(mesh, u_list):
    zfaces = [u(cell.midpoint()) for cell in cells(mesh)]
    mn = min(np.min(zfaces) for zfaces in zfaces)
    mx = max(np.max(zfaces) for zfaces in zfaces)
    return mn, mx


def visualization(mesh, u_list1, u_list2, min, max, mn, mx, filename):
    images = []
    for i, (u1, u2) in enumerate(zip(u_list1, u_list2)):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

        ax1.set_title("численное решение")
        ax2.set_title("аналитическое решение")

        n = mesh.num_vertices()
        d = mesh.geometry().dim()
        mesh_coordinates = mesh.coordinates().reshape((n, d))
        triangles = np.asarray([cell.entities(0) for cell in cells(mesh)])
        triangulation = tri.Triangulation(mesh_coordinates[:, 0], mesh_coordinates[:, 1], triangles)

        zfaces1 = np.asarray([u1(cell.midpoint()) for cell in cells(mesh)])
        zfaces2 = np.asarray([u2(cell.midpoint()) for cell in cells(mesh)])
        ax1.tripcolor(triangulation, facecolors=zfaces1, edgecolors='k',
                      norm=matplotlib.colors.Normalize(vmin=min, vmax=max))
        ax2.tripcolor(triangulation, facecolors=zfaces2, edgecolors='k',
                      norm=matplotlib.colors.Normalize(vmin=min, vmax=max))
        bounds = np.linspace(min, max, 6)
        cax = fig.add_axes([0.92, 0.122, 0.02, 0.756])
        ColorbarBase(cax, cmap=plt.cm.viridis, ticks=bounds, boundaries=bounds,
                     norm=matplotlib.colors.Normalize(vmin=min, vmax=max))
        plt.grid()

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)

        if i == len(u_list1) - 1:
            cax.remove()
            ax1.tripcolor(triangulation, facecolors=zfaces1, edgecolors='k',
                          norm=matplotlib.colors.Normalize(vmin=mn, vmax=mx))
            ax2.tripcolor(triangulation, facecolors=zfaces2, edgecolors='k',
                          norm=matplotlib.colors.Normalize(vmin=mn, vmax=mx))
            bounds_2 = np.linspace(mn, mx, 6)
            cax = fig.add_axes([0.92, 0.122, 0.02, 0.756])
            ColorbarBase(cax, cmap=plt.cm.viridis, ticks=bounds_2, boundaries=bounds_2,
                         norm=matplotlib.colors.Normalize(vmin=mn, vmax=mx))
            fig.savefig("2_solutions_final.png", bbox_inches='tight')

        plt.close()
    imageio.mimsave(filename, images, fps=3)


def boundary(x, on_boundary):
    return on_boundary and x[0] < 0


def paint_error(T, num_steps, error_L2_list, error_max_list):
    time_values = np.linspace(0, T, num_steps)
    plt.figure()
    plt.plot(time_values, error_L2_list, label='норма L2')
    plt.plot(time_values, error_max_list, label='максимум-норма')
    plt.xlabel('Время')
    plt.ylabel('погрешности')
    plt.title('норма L2 и максимум-норма')
    plt.legend()
    plt.grid()
    plt.savefig("2_погрешности")


T = 5
num_steps = 20
dt = T / num_steps
t = 0
r = Expression("sqrt(x[0] * x[0] + x[1] * x[1])", degree=2)
phi = Expression("atan2(x[1], x[0])", degree=2)
R = 1
a = 1
domain = Circle(Point(0, 0), R)
mesh = generate_mesh(domain, 32)
V = FunctionSpace(mesh, 'P', 2)

# test 1  u = r*r*sin(2*phi)+r*r+2*t
exact_u = Expression("r*r*sin(2*phi)+r*r+2*t", t=t, r=r, phi=phi, degree=2)
f = Expression("-2", t=t, r=r, phi=phi, degree=2)
g = Expression("2*r*sin(2*phi)+2*r", t=t, r=r, phi=phi, degree=2)
h = Expression("sin(2*phi)+2*t+1", t=t, r=r, phi=phi, degree=2)

# test 2  u = r*r*sin(phi)+t+2
# exact_u = Expression("r*r*sin(phi)+t+2", t=t, r=r, phi=phi, degree=2)
# f = Expression("1-3*sin(phi)", t=t, r=r, phi=phi, degree=2)
# g = Expression("2*r*sin(phi)", t=t, r=r, phi=phi, degree=2)
# h = Expression("sin(phi)+t+2", t=t, r=r, phi=phi, degree=2)

# test 3  u = r*r+2*r*r*sin(phi)*sin(phi)+4*t
# exact_u = Expression("r*r+2*r*r*sin(phi)*sin(phi)+4*t", t=t, r=r, phi=phi, degree=2)
# f = Expression("-4", t=t, r=r, phi=phi, degree=2)
# g = Expression("2*r+4*r*sin(phi)*sin(phi)", t=t, r=r, phi=phi, degree=2)
# h = Expression("2*sin(phi)*sin(phi)+4*t+1", t=t, r=r, phi=phi, degree=2)

u_n = interpolate(h, V)
u_list_fem = []
u_list_exact = []
error_L2_list = []
error_max_list = []
min_u = []
max_u = []

for n in range(num_steps):
    t += dt
    exact_u.t = t
    f.t = t
    h.t = t
    g.t = t
    u, error_L2, error_max = calculate(V, exact_u, a, f, h, g, u_n)
    error_L2_list.append(error_L2)
    error_max_list.append(error_max)
    u_n.assign(u)
    u_list_fem.append(u_n.copy())
    u_list_exact.append(
        Expression("r*r*sin(phi)+t+2", t=t, r=r, phi=phi, degree=2))
    mn, mx = findrange(mesh, u_list_fem)
    min_u.append(mn)
    max_u.append(mx)

visualization(mesh, u_list_fem, u_list_exact, min(min_u), max(max_u), mn, mx, "2_solutions.gif")
paint_error(T, num_steps, error_L2_list, error_max_list)
