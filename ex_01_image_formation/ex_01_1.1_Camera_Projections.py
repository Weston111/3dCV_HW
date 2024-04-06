import numpy as np
import matplotlib.pyplot as plt

# 定义两个矩阵
K = np.array([[100,0,25,0],
              [0,100,25,0],
              [0, 0 ,1 ,0],
              [0, 0 ,0 ,1]]
             )
R = np.array([[1,0,0,1 ],
              [0,0,-1,0],
              [0,1,0,2 ],
              [0,0,0,1 ]]
             )

# a)
P = np.dot(K, R)
print("\n矩阵K与矩阵R的乘积:")
print(P)

# b)
P_inv = np.linalg.inv(P)
print("\n矩阵P的逆:")
print(P_inv)
X_s=np.array([25,50,1,0.25])
X_w=np.dot(P_inv,X_s)
print("\nX_w:")
print(X_w)


# c)
def calculate_cube_vertices(center, side_length):
    x, y, z = center
    half_side = side_length / 2
    return [
        (x - half_side, y - half_side, z - half_side), #0
        (x - half_side, y + half_side, z - half_side), #1
        (x + half_side, y - half_side, z - half_side), #2
        (x + half_side, y + half_side, z - half_side), #3
        (x - half_side, y - half_side, z + half_side), #4
        (x - half_side, y + half_side, z + half_side), #5
        (x + half_side, y - half_side, z + half_side), #6
        (x + half_side, y + half_side, z + half_side), #7
    ]

edges=[[0,1],[0,2],[1,3],[2,3],[4,6],[4,5],[5,7],[6,7],[0,4],[2,6],[3,7],[1,5]]

def camera_projection_c(c_center,slide,fx,fy,cx,cy):
    vertices=calculate_cube_vertices(c_center, slide)
    K_0=np.array([[fx, 0 ,cx,0],
                [ 0  ,fy ,cy,0],
                [ 0  , 0 , 1,0]])
    vertices = np.array(vertices)
    #增广矩阵
    vertices = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
    vertices_T=vertices.T
    after_K_0 = np.dot(K_0,vertices_T)
    normalized=[]
    for i in range(8):
        normalized.append([after_K_0[0][i]/after_K_0[2][i],after_K_0[1][i]/after_K_0[2][i],1])
    normalized=np.array(normalized)
    normalized = np.round(normalized, 2)
    normalized = normalized[:, :-1]
    x_coords=normalized[:,0]
    y_coords=normalized[:,1]
    for edge in edges:
        # 获取边的两个顶点的坐标
        x1, y1 = x_coords[edge[0]],y_coords[edge[0]]
        x2, y2 = x_coords[edge[1]],y_coords[edge[1]]
        # 绘制这条边
        plt.plot([x1, x2], [y1, y2], 'b--')  # 'b-' 表示蓝色实线
    # 标记顶点
    plt.plot(x_coords, y_coords, 'o')  # 'o' 是标记点的样式

    # 设置图表标题和坐标轴标签
    plt.title('camera')
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    
    # 显示图表
    plt.show()
    return normalized

# i)
# result1=camera_projection_c(c_center=[0,0,15],slide=20,fx=5,fy=5,cx=10,cy=10)
# print(result1)

# ii)
# result2=camera_projection_c(c_center=[0,0,20],slide=20,fx=10,fy=10,cx=10,cy=10)
# print(result2)

# iii)
result3=camera_projection_c(c_center=[0,0,100],slide=20,fx=90,fy=90,cx=10,cy=10)
print(result3)
