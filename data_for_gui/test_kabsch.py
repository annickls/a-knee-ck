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

    femur_ursprung = np.array([-125.37961147396625,-90.4217324187824,1277.6917913483626])
    tibia_ursprung = np.array([-108.3848216194612,-90.25476224637612,1557.4634567569026])
    femur_kontakt = np.array([76.27677559858019,-105.80262508762264,1395.5233240191535])
    tibia_kontakt = np.array([75.72417768963511,-105.71990981913535,1403.4341806040643])
    distanz_femur = femur_kontakt-femur_ursprung
    distanz_tibia = tibia_kontakt-tibia_ursprung

    filePath = "data_for_gui/marker_coordinates.yaml"
    bone = "femur"
    translation, rotation = kabsch(filePath, bone)
    distanz_femur_rot = rotation@distanz_femur

    bone = "tibia"
    translation, rotation = kabsch(filePath, bone)
    distanz_tibia_rot = rotation@distanz_tibia
    
    print(f"Distanz von Femur Tracker zu kontaktpunkt: {np.round(distanz_femur_rot, 3)}")
    print(f"Distanz von Tibia Tracker zu kontaktpunkt: {np.round(distanz_tibia_rot, 3)}")