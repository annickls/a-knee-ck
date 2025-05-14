import numpy as np

class KneeJointAnalyzer:
    def __init__(self, femur_landmarks, tibia_landmarks):
        """Initialize with anatomical landmarks"""
        # Store landmarks
        self.femur_landmarks = {
            'name': 'femur',
            'proximal': np.array(femur_landmarks['proximal']),
            'distal': np.array(femur_landmarks['distal']),
            'lateral': np.array(femur_landmarks['lateral']),
            'medial': np.array(femur_landmarks['medial'])
        }
        
        self.tibia_landmarks = {
            'name': 'tibia',
            'proximal': np.array(tibia_landmarks['proximal']),
            'distal': np.array(tibia_landmarks['distal']),
            'lateral': np.array(tibia_landmarks['lateral']),
            'medial': np.array(tibia_landmarks['medial'])
        }
        
        # Create initial anatomical axes
        self.femur_axes, self.femur_transform = self.create_anatomical_axes(self.femur_landmarks)
        self.tibia_axes, self.tibia_transform = self.create_anatomical_axes(self.tibia_landmarks)
        
        # Store as initial state
        self.initial_femur_axes = np.copy(self.femur_axes)
        self.initial_tibia_axes = np.copy(self.tibia_axes)
        
        # Initialize coordinate frames for visualization
        self.coordinate_frames = []

    @staticmethod
    def normalize(v):
        """Normalize a vector to unit length"""
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def create_anatomical_axes(self, points):
        """
        Create anatomical coordinate system based on landmarks
        Returns a rotation matrix where columns are the anatomical axes
        """
        proximal = points['proximal']
        distal = points['distal']
        lateral = points['lateral']
        medial = points['medial']
        
        # Calculate midpoint between medial and lateral points
        mid_point = (lateral + medial) / 2
        
        # 1. Define proximal-distal axis (y-axis)
        # For femur: from distal to proximal
        # For tibia: from proximal to distal
        if points['name'] == 'femur':
            y_axis = self.normalize(proximal - distal)
        else:  # tibia
            y_axis = self.normalize(distal - proximal)
        
        # 2. Define medial-lateral axis (z-axis)
        # From medial to lateral
        z_axis_temp = self.normalize(lateral - medial)
        
        # 3. Define anterior-posterior axis (x-axis) using cross product
        # Ensuring orthogonality
        x_axis = self.normalize(np.cross(y_axis, z_axis_temp))
        
        # 4. Re-compute z-axis to ensure perfect orthogonality
        z_axis = self.normalize(np.cross(x_axis, y_axis))
        
        # Create rotation matrix (columns are the anatomical axes)
        rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
        
        # Create translation vector (origin is typically at the joint center)
        if points['name'] == 'femur':
            origin = distal  # For femur, origin at distal end (knee center)
        else:
            origin = proximal  # For tibia, origin at proximal end (knee center)
        
        # Create transformation matrix (4x4)
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = origin
        
        return rotation_matrix, transform

    def calculate_grood_suntay_angles(self, femur_frame, tibia_frame):
        """
        Calculate knee joint angles using Grood & Suntay's JCS approach
        """
        # Extract axis from frames
        e1 = femur_frame[:, 2]  # Femoral ML axis (fixed in femur)
        e3 = tibia_frame[:, 1]  # Tibial PD axis (fixed in tibia)
        
        # Calculate floating axis
        e2 = self.normalize(np.cross(e3, e1))
        
        # Extract remaining axes for calculations
        e1_fem = femur_frame[:, 0]  # Femoral AP axis
        e3_fem = femur_frame[:, 1]  # Femoral PD axis
        
        e1_tib = tibia_frame[:, 0]  # Tibial AP axis
        e2_tib = tibia_frame[:, 2]  # Tibial ML axis
        
        # Calculate joint angles
        # Flexion (+) / extension (-)
        cos_flexion = np.clip(np.dot(e3_fem, e3), -1.0, 1.0)
        sin_flexion = np.clip(np.dot(np.cross(e1, e3), e3_fem), -1.0, 1.0)
        flexion = np.degrees(np.arctan2(sin_flexion, cos_flexion))
        
        # Abduction (-) / adduction (+) (valgus/varus)
        abd_add = np.degrees(np.arcsin(np.clip(np.dot(e1, e2), -1.0, 1.0)))
        
        # Internal (+) / external (-) rotation
        cos_rotation = np.clip(np.dot(e2, e1_tib), -1.0, 1.0)
        sin_rotation = np.clip(np.dot(np.cross(e2, e1_tib), e3), -1.0, 1.0)
        rotation = np.degrees(np.arctan2(sin_rotation, cos_rotation))
        
        return flexion, abd_add, rotation
    
    def calculate_transformation(self, source_points, target_points):
        """
        Calculate rigid transformation from source to target points
        using the Kabsch algorithm
        """
        # Center the point sets
        source_centroid = np.mean(source_points, axis=0)
        target_centroid = np.mean(target_points, axis=0)
        
        source_centered = source_points - source_centroid
        target_centered = target_points - target_centroid
        
        # Calculate covariance matrix
        H = source_centered.T @ target_centered
        
        # SVD decomposition
        U, S, Vt = np.linalg.svd(H)
        
        # Calculate rotation matrix
        R = Vt.T @ U.T
        
        # Handle reflection case
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Calculate translation
        t = target_centroid - R @ source_centroid
        
        return R, t
    
    def update_transformations(self, femur_markers, tibia_markers):
        """
        Update transformations based on new marker positions
        
        Parameters:
        - femur_markers: Dictionary with keys 'proximal', 'distal', 'lateral', 'medial'
        - tibia_markers: Dictionary with keys 'proximal', 'distal', 'lateral', 'medial'
        
        Returns:
        - Dictionary of joint angles
        """
        # Convert to numpy arrays if needed
        femur_markers = {k: np.array(v) if not isinstance(v, np.ndarray) else v 
                         for k, v in femur_markers.items() if k != 'name'}
        tibia_markers = {k: np.array(v) if not isinstance(v, np.ndarray) else v 
                         for k, v in tibia_markers.items() if k != 'name'}
        
        # Calculate transformations from initial marker positions to current
        femur_R, femur_t = self.calculate_transformation(
            np.vstack([self.femur_landmarks[k] for k in ['proximal', 'distal', 'lateral', 'medial']]),
            np.vstack([femur_markers[k] for k in ['proximal', 'distal', 'lateral', 'medial']])
        )
        
        tibia_R, tibia_t = self.calculate_transformation(
            np.vstack([self.tibia_landmarks[k] for k in ['proximal', 'distal', 'lateral', 'medial']]),
            np.vstack([tibia_markers[k] for k in ['proximal', 'distal', 'lateral', 'medial']])
        )
        
        # Apply transformations to anatomical axes
        current_femur_axes = femur_R @ self.initial_femur_axes
        current_tibia_axes = tibia_R @ self.initial_tibia_axes
        
        # Calculate joint angles
        flexion, varus_valgus, rotation = self.calculate_grood_suntay_angles(
            current_femur_axes, current_tibia_axes)
        
        return {
            'flexion': flexion,
            'varus_valgus': varus_valgus,
            'rotation': rotation
        }