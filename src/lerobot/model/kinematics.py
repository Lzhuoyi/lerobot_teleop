# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from typing import Optional, Tuple, Dict, List
import warnings


class RobotKinematics:
    """Enhanced Robot kinematics using placo library for forward/inverse kinematics, Jacobian, and Hessian computations."""

    def __init__(
        self,
        urdf_path: str,
        target_frame_name: str = "gripper_frame_link",
        joint_names: list[str] = None,
    ):
        """
        Initialize placo-based kinematics solver.

        Args:
            urdf_path: Path to the robot URDF file
            target_frame_name: Name of the end-effector frame in the URDF
            joint_names: List of joint names to use for the kinematics solver
        """
        try:
            import placo
        except ImportError as e:
            raise ImportError(
                "placo is required for RobotKinematics. "
                "Please install the optional dependencies of `kinematics` in the package."
            ) from e

        self.placo = placo
        self.robot = placo.RobotWrapper(urdf_path)
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)  # Fix the base

        self.target_frame_name = target_frame_name

        # Set joint names
        self.joint_names = list(self.robot.joint_names()) if joint_names is None else joint_names
        self.n_joints = len(self.joint_names)

        # Initialize frame task for IK
        self.tip_frame = self.solver.add_frame_task(self.target_frame_name, np.eye(4))

        # Cache for efficiency
        self._last_joint_pos = None
        self._jacobian_cache = None
        self._hessian_cache = None

    def forward_kinematics(self, joint_pos_deg: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics for given joint configuration.

        Args:
            joint_pos_deg: Joint positions in degrees (numpy array)

        Returns:
            4x4 transformation matrix of the end-effector pose
        """
        # Convert degrees to radians
        joint_pos_rad = np.deg2rad(joint_pos_deg)
        for i, joint_name in enumerate(self.joint_names[:len(joint_pos_rad)]):
            self.robot.set_joint(joint_name, float(joint_pos_rad[i]))

        # Update kinematics
        self.robot.update_kinematics()

        # Update cache
        self._last_joint_pos = joint_pos_deg.copy()
        self._jacobian_cache = None
        self._hessian_cache = None

        # Get the transformation matrix
        return self.robot.get_T_world_frame(self.target_frame_name)

    def inverse_kinematics(
        self, 
        current_joint_pos: np.ndarray, 
        desired_ee_pose: np.ndarray, 
        position_weight: float = 1.0, 
        orientation_weight: float = 0.01
    ) -> np.ndarray:
        """
        Compute inverse kinematics using placo solver.

        Args:
            current_joint_pos: Current joint positions in degrees (used as initial guess)
            desired_ee_pose: Target end-effector pose as a 4x4 transformation matrix
            position_weight: Weight for position constraint in IK
            orientation_weight: Weight for orientation constraint in IK, set to 0.0 to only constrain position

        Returns:
            Joint positions in degrees that achieve the desired end-effector pose
        """
        # Convert current joint positions to radians for initial guess
        current_joint_rad = np.deg2rad(current_joint_pos[: len(self.joint_names)])

        # Set current joint positions as initial guess
        for i, joint_name in enumerate(self.joint_names[:len(current_joint_rad)]):
            self.robot.set_joint(joint_name, float(current_joint_rad[i]))

        # Update the target pose for the frame task
        self.tip_frame.T_world_frame = desired_ee_pose

        # Configure the task based on position_only flag
        self.tip_frame.configure(self.target_frame_name, "soft", position_weight, orientation_weight)

        # Solve IK
        self.solver.solve(True)
        self.robot.update_kinematics()

        # Extract joint positions
        joint_pos_rad = []
        for joint_name in self.joint_names:
            joint = self.robot.get_joint(joint_name)
            joint_pos_rad.append(joint)

        # Convert back to degrees
        joint_pos_deg = np.rad2deg(joint_pos_rad)

        # Update cache
        self._last_joint_pos = joint_pos_deg.copy()
        self._jacobian_cache = None
        self._hessian_cache = None

        # Preserve gripper position if present in current_joint_pos
        if len(current_joint_pos) > len(self.joint_names):
            result = np.zeros_like(current_joint_pos)
            result[: len(self.joint_names)] = joint_pos_deg
            result[len(self.joint_names) :] = current_joint_pos[len(self.joint_names) :]
            return result
        else:
            return joint_pos_deg

    def jacobian(self, joint_pos_deg: Optional[np.ndarray] = None, analytical: bool = True) -> np.ndarray:
        """
        Compute the Jacobian matrix of the end-effector with respect to joint positions.

        Args:
            joint_pos_deg: Joint positions in degrees. If None, uses last computed position.
            analytical: If True, computes analytical Jacobian. If False, uses numerical differentiation.

        Returns:
            6xN Jacobian matrix where N is the number of joints.
            First 3 rows are linear velocity, last 3 rows are angular velocity.
        """
        if joint_pos_deg is not None:
            # Update robot configuration
            self.forward_kinematics(joint_pos_deg)
        elif self._last_joint_pos is None:
            raise ValueError("No joint configuration provided and no previous configuration available")

        # Check cache
        if self._jacobian_cache is not None and analytical:
            return self._jacobian_cache

        if analytical:
            # Use placo's built-in Jacobian computation if available
            try:
                jacobian = self.robot.get_frame_jacobian(self.target_frame_name)
                self._jacobian_cache = jacobian
                return jacobian
            except AttributeError:
                # Fallback to numerical differentiation
                warnings.warn("Analytical Jacobian not available in placo, using numerical differentiation")
                analytical = False

        if not analytical:
            # Numerical differentiation
            epsilon = 1e-6
            current_pose = self.robot.get_T_world_frame(self.target_frame_name)
            jacobian = np.zeros((6, self.n_joints))

            for i in range(self.n_joints):
                # Positive perturbation
                joint_pos_plus = np.deg2rad(self._last_joint_pos[: self.n_joints].copy())
                joint_pos_plus[i] += epsilon
                
                for j, joint_name in enumerate(self.joint_names):
                    self.robot.set_joint(joint_name, joint_pos_plus[j])
                self.robot.update_kinematics()
                pose_plus = self.robot.get_T_world_frame(self.target_frame_name)

                # Negative perturbation
                joint_pos_minus = np.deg2rad(self._last_joint_pos[: self.n_joints].copy())
                joint_pos_minus[i] -= epsilon
                
                for j, joint_name in enumerate(self.joint_names):
                    self.robot.set_joint(joint_name, joint_pos_minus[j])
                self.robot.update_kinematics()
                pose_minus = self.robot.get_T_world_frame(self.target_frame_name)

                # Position derivative
                jacobian[:3, i] = (pose_plus[:3, 3] - pose_minus[:3, 3]) / (2 * epsilon)
                
                # Orientation derivative (using rotation matrix difference)
                R_plus = pose_plus[:3, :3]
                R_minus = pose_minus[:3, :3]
                dR = (R_plus - R_minus) / (2 * epsilon)
                
                # Convert to angular velocity (skew-symmetric part)
                current_R = current_pose[:3, :3]
                omega_skew = dR @ current_R.T
                jacobian[3:, i] = np.array([omega_skew[2, 1], omega_skew[0, 2], omega_skew[1, 0]])

            # Restore original configuration
            joint_pos_rad = np.deg2rad(self._last_joint_pos[: self.n_joints])
            for i, joint_name in enumerate(self.joint_names):
                self.robot.set_joint(joint_name, joint_pos_rad[i])
            self.robot.update_kinematics()

            return jacobian

    def hessian(self, joint_pos_deg: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute the Hessian tensor of the end-effector position with respect to joint positions.

        Args:
            joint_pos_deg: Joint positions in degrees. If None, uses last computed position.

        Returns:
            3xNxN Hessian tensor where N is the number of joints.
            hessian[i, j, k] represents the second derivative of position component i
            with respect to joints j and k.
        """
        if joint_pos_deg is not None:
            self.forward_kinematics(joint_pos_deg)
        elif self._last_joint_pos is None:
            raise ValueError("No joint configuration provided and no previous configuration available")

        # Check cache
        if self._hessian_cache is not None:
            return self._hessian_cache

        # Numerical computation of Hessian
        epsilon = 1e-6
        hessian = np.zeros((3, self.n_joints, self.n_joints))

        for i in range(self.n_joints):
            for j in range(i, self.n_joints):
                # Four-point stencil for second derivative
                joint_pos_base = np.deg2rad(self._last_joint_pos[: self.n_joints].copy())
                
                # f(x+h, y+h)
                joint_pos_pp = joint_pos_base.copy()
                joint_pos_pp[i] += epsilon
                joint_pos_pp[j] += epsilon
                pose_pp = self._evaluate_pose(joint_pos_pp)
                
                # f(x+h, y-h)
                joint_pos_pm = joint_pos_base.copy()
                joint_pos_pm[i] += epsilon
                joint_pos_pm[j] -= epsilon
                pose_pm = self._evaluate_pose(joint_pos_pm)
                
                # f(x-h, y+h)
                joint_pos_mp = joint_pos_base.copy()
                joint_pos_mp[i] -= epsilon
                joint_pos_mp[j] += epsilon
                pose_mp = self._evaluate_pose(joint_pos_mp)
                
                # f(x-h, y-h)
                joint_pos_mm = joint_pos_base.copy()
                joint_pos_mm[i] -= epsilon
                joint_pos_mm[j] -= epsilon
                pose_mm = self._evaluate_pose(joint_pos_mm)

                # Second derivative using finite differences
                second_deriv = (pose_pp[:3, 3] - pose_pm[:3, 3] - pose_mp[:3, 3] + pose_mm[:3, 3]) / (4 * epsilon**2)
                
                hessian[:, i, j] = second_deriv
                hessian[:, j, i] = second_deriv  # Symmetry

        # Restore original configuration
        joint_pos_rad = np.deg2rad(self._last_joint_pos[: self.n_joints])
        for i, joint_name in enumerate(self.joint_names):
            self.robot.set_joint(joint_name, joint_pos_rad[i])
        self.robot.update_kinematics()

        self._hessian_cache = hessian
        return hessian

    def _evaluate_pose(self, joint_pos_rad: np.ndarray) -> np.ndarray:
        """Helper function to evaluate pose for given joint positions in radians."""
        for i, joint_name in enumerate(self.joint_names):
            self.robot.set_joint(joint_name, joint_pos_rad[i])
        self.robot.update_kinematics()
        return self.robot.get_T_world_frame(self.target_frame_name)

    def manipulability(self, joint_pos_deg: Optional[np.ndarray] = None) -> float:
        """
        Compute the manipulability index (Yoshikawa's measure).

        Args:
            joint_pos_deg: Joint positions in degrees. If None, uses last computed position.

        Returns:
            Manipulability index (scalar value)
        """
        J = self.jacobian(joint_pos_deg)
        # Use only position part of Jacobian for manipulability
        J_pos = J[:3, :]
        return np.sqrt(np.linalg.det(J_pos @ J_pos.T))

    def condition_number(self, joint_pos_deg: Optional[np.ndarray] = None) -> float:
        """
        Compute the condition number of the Jacobian matrix.

        Args:
            joint_pos_deg: Joint positions in degrees. If None, uses last computed position.

        Returns:
            Condition number of the Jacobian
        """
        J = self.jacobian(joint_pos_deg)
        return np.linalg.cond(J)

    def singularity_check(self, joint_pos_deg: Optional[np.ndarray] = None, threshold: float = 1e-6) -> bool:
        """
        Check if the robot is near a singularity.

        Args:
            joint_pos_deg: Joint positions in degrees. If None, uses last computed position.
            threshold: Threshold for singularity detection

        Returns:
            True if robot is near singularity, False otherwise
        """
        manipulability = self.manipulability(joint_pos_deg)
        return manipulability < threshold

    def velocity_kinematics(self, joint_pos_deg: np.ndarray, joint_vel_deg: np.ndarray) -> np.ndarray:
        """
        Compute end-effector velocity from joint velocities.

        Args:
            joint_pos_deg: Joint positions in degrees
            joint_vel_deg: Joint velocities in degrees per second

        Returns:
            6x1 end-effector velocity vector (linear and angular)
        """
        J = self.jacobian(joint_pos_deg)
        joint_vel_rad = np.deg2rad(joint_vel_deg[: self.n_joints])
        return J @ joint_vel_rad

    def acceleration_kinematics(self, joint_pos_deg: np.ndarray, joint_vel_deg: np.ndarray, 
                               joint_acc_deg: np.ndarray) -> np.ndarray:
        """
        Compute end-effector acceleration from joint positions, velocities, and accelerations.

        Args:
            joint_pos_deg: Joint positions in degrees
            joint_vel_deg: Joint velocities in degrees per second
            joint_acc_deg: Joint accelerations in degrees per second squared

        Returns:
            6x1 end-effector acceleration vector (linear and angular)
        """
        J = self.jacobian(joint_pos_deg)
        H = self.hessian(joint_pos_deg)
        
        joint_vel_rad = np.deg2rad(joint_vel_deg[: self.n_joints])
        joint_acc_rad = np.deg2rad(joint_acc_deg[: self.n_joints])
        
        # Linear acceleration
        linear_acc = J[:3, :] @ joint_acc_rad
        for i in range(3):
            linear_acc[i] += joint_vel_rad.T @ H[i, :, :] @ joint_vel_rad
        
        # Angular acceleration (simplified - would need full implementation for complete accuracy)
        angular_acc = J[3:, :] @ joint_acc_rad
        
        return np.concatenate([linear_acc, angular_acc])

    def get_joint_limits(self) -> Dict[str, Tuple[float, float]]:
        """
        Get joint limits for all joints.

        Returns:
            Dictionary mapping joint names to (min_limit, max_limit) tuples in degrees
        """
        limits = {}
        for joint_name in self.joint_names:
            try:
                joint = self.robot.get_joint_info(joint_name)
                # Convert from radians to degrees
                limits[joint_name] = (np.rad2deg(joint.min_pos), np.rad2deg(joint.max_pos))
            except (AttributeError, KeyError):
                # If joint limits are not available, use default values
                limits[joint_name] = (-180.0, 180.0)
        return limits

    def check_joint_limits(self, joint_pos_deg: np.ndarray) -> List[bool]:
        """
        Check if joint positions are within limits.

        Args:
            joint_pos_deg: Joint positions in degrees

        Returns:
            List of boolean values indicating if each joint is within limits
        """
        limits = self.get_joint_limits()
        within_limits = []
        
        for i, joint_name in enumerate(self.joint_names):
            if i < len(joint_pos_deg):
                min_limit, max_limit = limits[joint_name]
                within_limits.append(min_limit <= joint_pos_deg[i] <= max_limit)
            else:
                within_limits.append(True)
        
        return within_limits

    def distance_to_joint_limits(self, joint_pos_deg: np.ndarray) -> np.ndarray:
        """
        Compute distance to joint limits for each joint.

        Args:
            joint_pos_deg: Joint positions in degrees

        Returns:
            Array of distances to closest joint limit for each joint
        """
        limits = self.get_joint_limits()
        distances = []
        
        for i, joint_name in enumerate(self.joint_names):
            if i < len(joint_pos_deg):
                min_limit, max_limit = limits[joint_name]
                dist_to_min = joint_pos_deg[i] - min_limit
                dist_to_max = max_limit - joint_pos_deg[i]
                distances.append(min(dist_to_min, dist_to_max))
            else:
                distances.append(float('inf'))
        
        return np.array(distances)

    def rodrigues_rotation_matrix(self, rotation_vector: np.ndarray) -> np.ndarray:
        """
        Convert rotation vector to rotation matrix using Rodrigues' rotation formula.
        
        Rodrigues' formula: R = I + sin(θ)[k]_× + (1-cos(θ))[k]_×²
        where θ is the rotation angle, k is the unit axis, and [k]_× is the skew-symmetric matrix.

        Args:
            rotation_vector: 3D rotation vector where the direction is the rotation axis
                           and the magnitude is the rotation angle in radians

        Returns:
            3x3 rotation matrix
        """
        # Handle zero rotation case
        angle = np.linalg.norm(rotation_vector)
        if angle < 1e-12:
            return np.eye(3)
        
        # Normalize to get unit axis
        axis = rotation_vector / angle
        
        # Create skew-symmetric matrix
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        # Rodrigues' formula
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
        
        return R

    def rotation_matrix_to_rodrigues(self, rotation_matrix: np.ndarray) -> np.ndarray:
        """
        Convert rotation matrix to rotation vector using inverse Rodrigues' formula.

        Args:
            rotation_matrix: 3x3 rotation matrix

        Returns:
            3D rotation vector (axis-angle representation)
        """
        # Ensure it's a proper rotation matrix
        if not np.allclose(np.linalg.det(rotation_matrix), 1.0, atol=1e-6):
            raise ValueError("Input matrix is not a proper rotation matrix (determinant != 1)")
        
        if not np.allclose(rotation_matrix @ rotation_matrix.T, np.eye(3), atol=1e-6):
            raise ValueError("Input matrix is not orthogonal")
        
        # Calculate rotation angle
        trace = np.trace(rotation_matrix)
        angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        
        # Handle identity matrix case
        if angle < 1e-12:
            return np.zeros(3)
        
        # Handle 180-degree rotation case
        if abs(angle - np.pi) < 1e-6:
            # Find the eigenvector corresponding to eigenvalue 1
            eigenvals, eigenvecs = np.linalg.eig(rotation_matrix)
            axis_idx = np.argmin(np.abs(eigenvals - 1))
            axis = np.real(eigenvecs[:, axis_idx])
            axis = axis / np.linalg.norm(axis)
            return angle * axis
        
        # General case
        axis = (1 / (2 * np.sin(angle))) * np.array([
            rotation_matrix[2, 1] - rotation_matrix[1, 2],
            rotation_matrix[0, 2] - rotation_matrix[2, 0],
            rotation_matrix[1, 0] - rotation_matrix[0, 1]
        ])
        
        return angle * axis

    def pose_to_rodrigues(self, pose_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract position and Rodrigues rotation vector from a 4x4 pose matrix.

        Args:
            pose_matrix: 4x4 transformation matrix

        Returns:
            Tuple of (position_vector, rotation_vector) where:
            - position_vector: 3D position vector
            - rotation_vector: 3D rotation vector (Rodrigues representation)
        """
        position = pose_matrix[:3, 3]
        rotation_matrix = pose_matrix[:3, :3]
        rotation_vector = self.rotation_matrix_to_rodrigues(rotation_matrix)
        
        return position, rotation_vector

    def rodrigues_to_pose(self, position: np.ndarray, rotation_vector: np.ndarray) -> np.ndarray:
        """
        Create a 4x4 pose matrix from position and Rodrigues rotation vector.

        Args:
            position: 3D position vector
            rotation_vector: 3D rotation vector (Rodrigues representation)

        Returns:
            4x4 transformation matrix
        """
        rotation_matrix = self.rodrigues_rotation_matrix(rotation_vector)
        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = rotation_matrix
        pose_matrix[:3, 3] = position
        
        return pose_matrix

    def rodrigues_jacobian(self, rotation_vector: np.ndarray) -> np.ndarray:
        """
        Compute the Jacobian of the Rodrigues rotation formula.
        This relates changes in rotation vector to changes in rotation matrix.

        Args:
            rotation_vector: 3D rotation vector

        Returns:
            9x3 Jacobian matrix where each column represents the derivative of
            the vectorized rotation matrix with respect to one component of the rotation vector
        """
        angle = np.linalg.norm(rotation_vector)
        
        if angle < 1e-12:
            # For small angles, use linear approximation
            J = np.zeros((9, 3))
            # Identity matrix entries remain unchanged
            # Only off-diagonal elements change
            J[1, 2] = -1  # R[0,1] w.r.t. rotation_vector[2]
            J[2, 1] = 1   # R[0,2] w.r.t. rotation_vector[1]
            J[3, 2] = 1   # R[1,0] w.r.t. rotation_vector[2]
            J[5, 0] = -1  # R[1,2] w.r.t. rotation_vector[0]
            J[6, 1] = -1  # R[2,0] w.r.t. rotation_vector[1]
            J[7, 0] = 1   # R[2,1] w.r.t. rotation_vector[0]
            return J
        
        # For general case, use numerical differentiation
        epsilon = 1e-8
        J = np.zeros((9, 3))
        R_base = self.rodrigues_rotation_matrix(rotation_vector)
        
        for i in range(3):
            # Positive perturbation
            rv_plus = rotation_vector.copy()
            rv_plus[i] += epsilon
            R_plus = self.rodrigues_rotation_matrix(rv_plus)
            
            # Negative perturbation
            rv_minus = rotation_vector.copy()
            rv_minus[i] -= epsilon
            R_minus = self.rodrigues_rotation_matrix(rv_minus)
            
            # Numerical derivative
            dR_dri = (R_plus - R_minus) / (2 * epsilon)
            J[:, i] = dR_dri.flatten()
        
        return J

    def slerp_rodrigues(self, rv1: np.ndarray, rv2: np.ndarray, t: float) -> np.ndarray:
        """
        Spherical linear interpolation between two rotation vectors using Rodrigues representation.

        Args:
            rv1: First rotation vector
            rv2: Second rotation vector
            t: Interpolation parameter [0, 1]

        Returns:
            Interpolated rotation vector
        """
        if not (0 <= t <= 1):
            raise ValueError("Interpolation parameter t must be in [0, 1]")
        
        # Convert to rotation matrices
        R1 = self.rodrigues_rotation_matrix(rv1)
        R2 = self.rodrigues_rotation_matrix(rv2)
        
        # Compute relative rotation
        R_rel = R2 @ R1.T
        
        # Convert to rotation vector
        rv_rel = self.rotation_matrix_to_rodrigues(R_rel)
        
        # Scale by interpolation parameter
        rv_interp_rel = t * rv_rel
        
        # Convert back to rotation matrix
        R_interp_rel = self.rodrigues_rotation_matrix(rv_interp_rel)
        
        # Apply to first rotation
        R_interp = R_interp_rel @ R1
        
        # Convert back to rotation vector
        return self.rotation_matrix_to_rodrigues(R_interp)

    def rodrigues_distance(self, rv1: np.ndarray, rv2: np.ndarray) -> float:
        """
        Compute the angular distance between two rotation vectors.

        Args:
            rv1: First rotation vector
            rv2: Second rotation vector

        Returns:
            Angular distance in radians
        """
        R1 = self.rodrigues_rotation_matrix(rv1)
        R2 = self.rodrigues_rotation_matrix(rv2)
        
        # Relative rotation
        R_rel = R2 @ R1.T
        
        # Extract angle from relative rotation
        trace = np.trace(R_rel)
        angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        
        return angle

    def clear_cache(self):
        """Clear all cached computations."""
        self._jacobian_cache = None
        self._hessian_cache = None
        self._last_joint_pos = None