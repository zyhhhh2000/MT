from mshr import *
from fenics import *
import imageio
import numpy as np
import matplotlib
import matplotlib.tri as tri
import matplotlib.pyplot as plt


def calculate(V, exact_u, alpha, f, h, g):
    def is_left_boundary(x, on_boundary):
        return on_boundary and x[0] < 0

    bc = DirichletBC(V, h, is_left_boundary)

    u = TrialFunction(V)
    v = TestFunction(V)
    a = (dot(grad(u), grad(v)) + alpha * u * v) * dx
    L = f * v * dx + g * v * ds

    u = Function(V)
    solve(a == L, u, bc)

    error_L2 = errornorm(exact_u, u, 'L2')

    vertex_values_exact_u = exact_u.compute_vertex_values(mesh)
    vertex_values_u = u.compute_vertex_values(mesh)
    error_max = np.max(np.abs(vertex_values_exact_u - vertex_values_u))

    print('норма L2  =', error_L2)
    print('максимум-норма =', error_max)
    return u


def visualize(mesh, u, exact_u, filename):
    n = mesh.num_vertices()
    d = mesh.geometry().dim()
    mesh_coordinates = mesh.coordinates().reshape((n, d))
    triangles = np.asarray([cell.entities(0) for cell in cells(mesh)])
    triangulation = tri.Triangulation(mesh_coordinates[:, 0], mesh_coordinates[:, 1], triangles)

    plt.subplot(1, 2, 1)
    zfaces = np.asarray([u(cell.midpoint()) for cell in cells(mesh)])
    plt.tripcolor(triangulation, facecolors=zfaces, edgecolors='k')
    plt.grid()
    plt.colorbar()
    plt.title("численное решение")

    plt.subplot(1, 2, 2)
    zfaces = np.asarray([exact_u(cell.midpoint()) for cell in cells(mesh)])
    plt.tripcolor(triangulation, facecolors=zfaces, edgecolors='k')
    plt.grid()
    plt.colorbar()
    plt.title("аналитическое решение")

    plt.tight_layout()
    plt.gcf().set_size_inches(24, 8)
    plt.savefig(filename)
    plt.close()


r = Expression("sqrt(x[0] * x[0] + x[1] * x[1])", degree=2)
phi = Expression("atan2(x[1], x[0])", degree=2)
R = 1
alpha = 1

# test 1  u = r ** 2 * sin(2*phi)
exact_u = Expression('r * r * sin(2*phi)', r=r, phi=phi, degree=2)
f = Expression("alpha * r * r * sin(2*phi)", alpha=alpha, r=r, phi=phi, degree=2)
g = Expression("2 * r * sin(2*phi)", r=r, phi=phi, degree=2)
h = Expression("sin(2*phi)", phi = phi, degree=2)

# test 2  u = r * sin(phi)
# exact_u = Expression("r * sin(phi)", r=r, phi=phi, degree=2)
# f = Expression("alpha * r * sin(phi)", alpha=alpha, r=r, phi=phi, degree=2)
# g = Expression("sin(phi)", phi=phi, degree=2)
# h = Expression("sin(phi)", phi=phi, degree=2)

# test 3  u = r ** 2 * sin(phi) + r ** 2
# exact_u = Expression('r * r * sin(phi) + r * r', r=r, phi=phi, degree=2)
# f = Expression("alpha * r * r * sin(phi) + alpha * r * r - 4 - 3 * sin(phi)", alpha=alpha, r=r, phi=phi, degree=2)
# g = Expression("2 * r * sin(phi) + 2 * r", r=r, phi=phi, degree=2)
# h = Expression("1 + sin(phi)", phi=phi, degree=2)

domain = Circle(Point(0, 0), R)
mesh = generate_mesh(domain, 10)
V = FunctionSpace(mesh, 'P', 2)
u = calculate(V, exact_u, alpha, f, h, g)

visualize(mesh, u, exact_u, "1_solution.png")
