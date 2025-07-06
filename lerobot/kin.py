import math
import numpy as np
import random

class ArmKinematics:
    """
    A class to represent the kinematics of a robot arm.
    All public methods use degrees for input/output.
    """

    def __init__(self, l1=0.1159, l2=0.1350):
        self.l1 = l1  # Length of the first link
        self.l2 = l2  # Length of the second link

    def deg2rad(self, deg):
        return deg * math.pi / 180.0

    def rad2deg(self, rad):
        return rad * 180.0 / math.pi

    def inverse_kinematics(self, x, y, current=None):
        theta1_offset = -math.atan2(0.028, 0.11257)
        theta2_offset = -math.atan2(0.0052, 0.1349) + theta1_offset

        r = math.sqrt(x**2 + y**2)
        r_max, r_min = self.l1 + self.l2, abs(self.l1 - self.l2)
        if r > r_max or r < r_min:
            print("Target position out of reach:", (x, y))
            if current is not None:
                return current
            else:
                scale = r_max / r if r > r_max else r_min / r
                x *= scale
                y *= scale
                r = math.sqrt(x**2 + y**2)

        # CORRECTED: Remove negative sign
        cos_theta2 = (r**2 - self.l1**2 - self.l2**2) / (2 * self.l1 * self.l2)
        cos_theta2 = max(-1.0, min(1.0, cos_theta2))

        # CORRECTED: Two proper solutions
        theta2a = math.acos(cos_theta2)   # elbow-down
        theta2b = -theta2a                # elbow-up

        beta = math.atan2(y, x)
        
        # CORRECTED: Use subtraction formula
        def calc_theta1(theta2):
            denom = self.l1 + self.l2 * math.cos(theta2)
            num = self.l2 * math.sin(theta2)
            gamma = math.atan2(num, denom)
            return beta - gamma  # CORRECTED: subtraction

        theta1a = calc_theta1(theta2a)
        theta1b = calc_theta1(theta2b)

        # Apply offsets and convert
        solutions = [
            (-self.rad2deg(theta1a - theta1_offset), -self.rad2deg(theta2a - theta2_offset)),
            (-self.rad2deg(theta1b - theta1_offset), -self.rad2deg(theta2b - theta2_offset))
        ]

        # Define joint limits in degrees
        joint2_min = self.rad2deg(-math.pi/2)  # ≈ -5.73°
        joint2_max = self.rad2deg(3.45)  # ≈ 197.7°
        joint3_min = self.rad2deg(-math.pi/2)  # ≈ -11.46°
        joint3_max = self.rad2deg(math.pi)  # ≈ 180°

        # Filter solutions to only include those within joint limits
        valid_solutions = []
        for sol in solutions:
            j2, j3 = sol
            if (joint2_min <= j2 <= joint2_max and 
                joint3_min <= j3 <= joint3_max):
                valid_solutions.append(sol)

        print("All IK solutions (before filtering):", solutions)
        
        # If no valid solutions, fall back to current position
        if not valid_solutions:
            if current is not None:
                print("\nNo valid IK solutions found, returning current position.")
                return current
            else:
                return (0, 0)  # Default fallback

        # Select solution near current position using circular distance
        if current is not None:
            distances = []
            for sol in valid_solutions:
                # Circular distance for joint2
                diff_j2 = abs(sol[0] - current[0])
                circular_diff_j2 = min(diff_j2, 360 - diff_j2)
                
                # Absolute distance for joint3
                diff_j3 = abs(sol[1] - current[1])
                
                distances.append(circular_diff_j2 + diff_j3)
            
            chosen_idx = np.argmin(distances)
            chosen = valid_solutions[chosen_idx]
        else:
            chosen = valid_solutions[0]  # Default to first valid solution

        return chosen  # No clamping needed since solutions are pre-filtered

    def forward_kinematics(self, joint2_deg, joint3_deg):
        """
        2-link planar arm forward kinematics.
        joint2_deg: shoulder angle (degrees)
        joint3_deg: elbow angle (degrees)
        Returns: (x, y) position of the end effector
        """
        # Convert to radians
        joint2 = self.deg2rad(-joint2_deg)
        joint3 = self.deg2rad(-joint3_deg)

        theta1_offset = -math.atan2(0.028, 0.11257)
        theta2_offset = -math.atan2(0.0052, 0.1349) + theta1_offset

        theta1 = joint2 + theta1_offset
        theta2 = joint3 + theta2_offset

        x = self.l1 * math.cos(theta1) + self.l2 * math.cos(theta1 + theta2)
        y = self.l1 * math.sin(theta1) + self.l2 * math.sin(theta1 + theta2)
        return x, y
    
def validate():
    """
    Validate ArmKinematics FK/IK consistency:
    1. Generate random joint2, joint3 in degree (within limits).
    2. Compute (x, y) using FK.
    3. Compute joint2', joint3' using IK from (x, y).
    4. Print and compare the original and retrieved joint values.
    """
    kin = ArmKinematics()
    # Joint limits in degrees (converted from radians used in class)
    joint2_min, joint2_max = kin.rad2deg(-math.pi/2), kin.rad2deg(3.45)
    joint3_min, joint3_max = kin.rad2deg(-math.pi/2), kin.rad2deg(math.pi)

    # Generate random joint values within limits
    joint2_deg = random.uniform(joint2_min, joint2_max)
    joint3_deg = random.uniform(joint3_min, joint3_max)

    print(f"Original joint2: {joint2_deg:.2f} deg, joint3: {joint3_deg:.2f} deg")

    # Forward kinematics
    x, y = kin.forward_kinematics(joint2_deg, joint3_deg)
    print(f"FK result: x={x:.4f}, y={y:.4f}")

    # Inverse kinematics
    joint2_deg_rec, joint3_deg_rec = kin.inverse_kinematics(x, y, current=(joint2_deg, joint3_deg))
    print(f"IK result: joint2: {joint2_deg_rec:.2f} deg, joint3: {joint3_deg_rec:.2f} deg")

    # Difference
    diff2 = abs(joint2_deg - joint2_deg_rec)
    diff3 = abs(joint3_deg - joint3_deg_rec)
    print(f"Difference: joint2: {diff2:.4f} deg, joint3: {diff3:.4f} deg")

    if diff2 < 1e-2 and diff3 < 1e-2:
        print("Validation PASSED.")
    else:
        print("Validation WARNING: Large difference detected.")

# Example usage:
if __name__ == "__main__":
    validate()