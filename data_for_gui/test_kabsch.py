import yaml
import numpy as np

def kabsch(filePath, bone):
    """Calculate the optimal rigid transformation matrix from Q -> P using Kabsch algorithm"""

    with open(filePath, "r") as file:
        content = yaml.safe_load(file)


    def readYaml(marker):
        array = np.array([])
        for i in range(5):
            array = np.append(array, [content[marker][i]["x"], content[marker][i]["y"], content[marker][i]["z"]])
        array = array.reshape([5,3])
        return array

    bone_ref = readYaml(bone+"_ref")
    bone_slicer = readYaml(bone+"_slicer")

    q = bone_ref
    p = bone_slicer

    centroid_p = np.mean(p, axis=0)
    centroid_q = np.mean(q, axis=0)

    p_centered = p - centroid_p
    q_centered = q - centroid_q

    H = np.dot(p_centered.T, q_centered)

    U, _,  vt = np.linalg.svd(H)

    R = np.dot(vt.T, U.T)

    if np.linalg.det(R) < 0:
        vt[-1, :] *= -1
        R = np.dot(vt.T, U.T)

    t = centroid_q - np.dot(centroid_p, R.T)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return t, R


if __name__ == "__main__":

    filePath = "data_for_gui/marker_coordinates.yaml"
    bone = "femur"
    translation, rotation = kabsch(filePath, bone)
    print(f"Translation: \n{translation}\nRotation: \n{rotation}")