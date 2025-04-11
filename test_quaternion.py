import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

def kabsch_algorithm(P, Q):
    """
    Implementierung des Kabsch-Algorithmus zur Berechnung der optimalen Rotationsmatrix
    zwischen zwei Punktwolken P und Q.
    P: Nx3 Matrix mit Ausgangspunkten
    Q: Nx3 Matrix mit Zielpunkten
    """
    # Mittelwerte berechnen
    P_mean = np.mean(P, axis=0)
    Q_mean = np.mean(Q, axis=0)
    
    # Zentrierte Punktmengen
    P_centered = P - P_mean
    Q_centered = Q - Q_mean
    
    # Kovarianzmatrix berechnen
    H = P_centered.T @ Q_centered
    
    # SVD durchführen
    U, S, Vt = np.linalg.svd(H)
    
    # Rotationsmatrix berechnen
    R_opt = Vt.T @ U.T
    
    # Sicherstellen, dass es sich um eine echte Rotationsmatrix handelt
    if np.linalg.det(R_opt) < 0:
        Vt[-1, :] *= -1
        R_opt = Vt.T @ U.T
    
    # Translation berechnen
    t_opt = Q_mean - R_opt @ P_mean
    
    return R_opt, t_opt

# Definiere einen Quader mit 5 Eckpunkten
P = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [5, 5, 1]
])

# Berechnung des Mittelpunkts der viereckigen Grundfläche
ground_center = np.mean(P[:4], axis=0)

# Benutzerdefinierte Quaternionen eingeben
quaternions = [
    [0, 0, np.sin(np.pi/8), np.cos(np.pi/8)],  # 45°-Drehung um Z
    [np.sin(np.pi/8), 0, 0, np.cos(np.pi/8)],  # 45°-Drehung um X
    [0, np.sin(np.pi/8), 0, np.cos(np.pi/8)],  # 45°-Drehung um Y
    [np.sin(np.pi/6), np.sin(np.pi/6), np.sin(np.pi/6), np.cos(np.pi/6)] # Kombinierte Rotation
]

colors = ['b', 'g', 'c', 'm']
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(P[:,0], P[:,1], P[:,2], color='r', label='Originale Marker')

for quat, color in zip(quaternions, colors):
    rotation_matrix = R.from_quat(quat).as_matrix()
    
    # Punkte relativ zum Mittelpunkt der Grundfläche verschieben
    P_centered = P - ground_center
    
    # Rotation anwenden
    P_rotated = (rotation_matrix @ P_centered.T).T
    
    # Punkte zurückverschieben
    P_rotated += ground_center
    
    ax.scatter(P_rotated[:,0], P_rotated[:,1], P_rotated[:,2], color=color, label=f'Rotation {color}')
    
    for edge in [(0,1), (1,2), (2,3), (3,0), (0,4), (1,4), (2,4), (3,4)]:
        ax.plot([P_rotated[edge[0], 0], P_rotated[edge[1], 0]],
                [P_rotated[edge[0], 1], P_rotated[edge[1], 1]],
                [P_rotated[edge[0], 2], P_rotated[edge[1], 2]], color+'-')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()