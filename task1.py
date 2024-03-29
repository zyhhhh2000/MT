import numpy as np

def load_obj(file_path):
    vertices = [[] for i in range(9)]
    verticess = []
    faces = [[] for i in range(9)]
    roles = 0
    number = 1

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('g '):
                roles = roles + 1
            elif line.startswith('# object'):
                number = [int(num) for num in line[-2:-1][0].split()]
            elif line.startswith('v '):
                vertex = list(map(float, line[2:].split()))
                vertices[number[0]-1].append((vertex[0], vertex[1], vertex[2]))
                verticess.append((vertex[0], vertex[1], vertex[2]))
            elif line.startswith('f '):
                indices = list(map(int, line[2:].split()))
                faces[number[0]-1].append((indices[0], indices[1], indices[2]))
                # faces[roles-1].append(((vertices[roles-1][indices[0]],vertices[roles-1][indices[1]],vertices[roles-1][indices[2]])))
    # print(vertices)
    return vertices, verticess, faces, roles

def count(point1, point2, point3):
    point1 = np.array(point1)
    point2 = np.array(point2)
    point3 = np.array(point3)
    area = 0.5 * np.linalg.norm(np.cross(point2 - point1, point3 - point1))
    return area

vertices, verticess, faces, roles = load_obj('model3.obj')  # 对obj文件进行读取操作
S_i = np.zeros(9)
S_ij = np.zeros((9, 9))

for i in range(9):
    x = []
    y = []
    z = []
    for element in (vertices[i]):
        max = 0
        min = 0
        x.append(element[0])
        z.append(element[1])
        y.append(element[2])
    for element in (faces[i]):
        if np.min([verticess[element[0] - 1][1],verticess[element[1] - 1][1],verticess[element[2] - 1][1]]) == np.max(z):
            max = max + count(verticess[element[0] - 1], verticess[element[1] - 1], verticess[element[2] - 1])
        elif np.max([verticess[element[0] - 1][1],verticess[element[1] - 1][1],verticess[element[2] - 1][1]]) == np.min(z):
            min = min + count(verticess[element[0] - 1], verticess[element[1] - 1], verticess[element[2] - 1])
    # print(min,max,i,np.min(z),np.max(z))
    if i == 0:
        S_ij[i][i+1] = S_ij[i+1][i] = max
    elif i == 8:
        S_ij[i-1][i] = S_ij[i][i-1] = min
    elif i != 4:
        S_ij[i][i+1] = S_ij[i+1][i] = max
        S_ij[i-1][i] = S_ij[i][i-1] = min
for i in range(9):
    for element in (faces[i]):
        S_i[i] = S_i[i] + count(verticess[element[0] - 1], verticess[element[1] - 1], verticess[element[2] - 1])
    # print(S_i[i])
    if i == 0:
        S_i[i] = S_i[i] - S_ij[i][i+1]
    elif i == 8:
        S_i[i] = S_i[i] - S_ij[i-1][i]
    else:
        S_i[i] = S_i[i] - S_ij[i - 1][i] - S_ij[i][i+1]

print(S_i)
print('/')
print(S_ij)