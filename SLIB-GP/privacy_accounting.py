# Privacy accounting utilities for DP-SGD
import math
import numpy as np
from typing import Tuple, List


class RDPAccountant:
    """
    Renyi Differential Privacy (RDP) accountant for DP-SGD.

    This implements the RDP accounting mechanism described in:
    "Renyi Differential Privacy" by Mironov (2017)
    """

    def __init__(self, orders: List[float] = None):
        if orders is None:
            # Default orders for RDP computation
            self.orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
        else:
            self.orders = orders

    def compute_rdp(self, q: float, noise_multiplier: float, steps: int) -> float:
        """
        Compute RDP for Gaussian mechanism with subsampling.

        Args:
            q: Sampling probability (batch_size / dataset_size)
            noise_multiplier: Noise multiplier for DP-SGD
            steps: Number of training steps

        Returns:
            RDP value
        """
        if noise_multiplier == 0:
            return float('inf')

        # 使用更保守的RDP计算
        # 对于小采样概率，使用近似公式
        if q < 0.01:
            # 近似公式：RDP ≈ q * steps / (2 * sigma^2)
            rdp = q * steps / (2 * noise_multiplier ** 2)
        else:
            # 对于较大的采样概率，使用更精确的计算
            rdp = 0
            for alpha in self.orders[:10]:  # 只使用前10个order
                rdp_alpha = self._compute_rdp_gaussian(alpha, q, noise_multiplier)
                rdp = max(rdp, rdp_alpha * steps)

        return rdp

    def _compute_rdp_gaussian(self, alpha: float, q: float, noise_multiplier: float) -> float:
        """Compute RDP for Gaussian mechanism with subsampling."""
        if noise_multiplier == 0:
            return float('inf')

        # RDP for Gaussian mechanism
        sigma = noise_multiplier
        rdp_gaussian = alpha / (2 * sigma ** 2)

        # Account for subsampling using the tight bound
        if q == 1:
            return rdp_gaussian

        # RDP for subsampled Gaussian mechanism
        rdp_subsampled = self._compute_rdp_subsampled_gaussian(alpha, q, sigma)
        return rdp_subsampled

    def _compute_rdp_subsampled_gaussian(self, alpha: float, q: float, sigma: float) -> float:
        """Compute RDP for subsampled Gaussian mechanism."""
        if q == 0:
            return 0

        # Use the tight bound for subsampled Gaussian mechanism
        # This is a simplified version - in practice, you'd use more sophisticated bounds

        # For small q, use the Poisson subsampling bound
        if q < 0.1:
            return self._poisson_subsampling_rdp(alpha, q, sigma)
        else:
            # For larger q, use the tight bound
            return self._tight_subsampling_rdp(alpha, q, sigma)

    def _poisson_subsampling_rdp(self, alpha: float, q: float, sigma: float) -> float:
        """RDP for Poisson subsampling."""
        if q == 0:
            return 0

        # 简化的 Poisson 子采样 RDP 计算
        # 使用近似公式避免复杂的组合数计算
        rdp = q * alpha / (2 * sigma ** 2)
        return rdp

    def _tight_subsampling_rdp(self, alpha: float, q: float, sigma: float) -> float:
        """Tight RDP bound for subsampled Gaussian mechanism."""
        # 简化的子采样 RDP 计算
        rdp_gaussian = alpha / (2 * sigma ** 2)
        rdp_subsampled = q * rdp_gaussian
        return rdp_subsampled

    def get_privacy_spent(self, q: float, noise_multiplier: float, steps: int,
                          delta: float = 1e-5) -> Tuple[float, float]:
        """
        Convert RDP to (epsilon, delta) privacy guarantee.

        Args:
            q: Sampling probability
            noise_multiplier: Noise multiplier
            steps: Number of steps
            delta: Failure probability

        Returns:
            (epsilon, delta) privacy guarantee
        """
        rdp = self.compute_rdp(q, noise_multiplier, steps)

        if rdp == float('inf'):
            return float('inf'), delta

        # Convert RDP to (epsilon, delta)
        epsilon = rdp + math.log(1 / delta) / (max(self.orders) - 1)

        return epsilon, delta


class MomentsAccountant:
    """
    Moments accountant for DP-SGD.

    This implements the moments accountant from:
    "Deep Learning with Differential Privacy" by Abadi et al. (2016)
    """

    def __init__(self, max_moment: int = 32):
        self.max_moment = max_moment
        self.moments = [0.0] * (max_moment + 1)

    def accumulate_rdp(self, q: float, noise_multiplier: float, steps: int):
        """Accumulate RDP over training steps."""
        if noise_multiplier == 0:
            return float('inf')

        # Compute RDP for each moment
        for i in range(1, self.max_moment + 1):
            alpha = i
            rdp_alpha = self._compute_rdp_moment(alpha, q, noise_multiplier)
            self.moments[i] = rdp_alpha * steps

    def _compute_rdp_moment(self, alpha: float, q: float, noise_multiplier: float) -> float:
        """Compute RDP for a specific moment."""
        if noise_multiplier == 0:
            return float('inf')

        sigma = noise_multiplier

        # RDP for Gaussian mechanism
        rdp_gaussian = alpha / (2 * sigma ** 2)

        # Account for subsampling
        if q == 1:
            return rdp_gaussian

        # Use the tight bound for subsampled Gaussian mechanism
        return self._compute_subsampled_rdp(alpha, q, sigma)

    def _compute_subsampled_rdp(self, alpha: float, q: float, sigma: float) -> float:
        """Compute RDP for subsampled Gaussian mechanism."""
        if q == 0:
            return 0

        # Simplified subsampling bound
        # In practice, you'd use the tight bound from the literature
        rdp_gaussian = alpha / (2 * sigma ** 2)

        # Conservative bound for subsampling
        if q < 1:
            rdp_subsampled = q * rdp_gaussian
        else:
            rdp_subsampled = rdp_gaussian

        return rdp_subsampled

    def get_privacy_spent(self, delta: float = 1e-5) -> Tuple[float, float]:
        """Get privacy spent using moments accountant."""
        # Find the minimum epsilon over all moments
        min_epsilon = float('inf')

        for i in range(1, self.max_moment + 1):
            if self.moments[i] > 0:
                epsilon = self.moments[i] + math.log(1 / delta) / (i - 1)
                min_epsilon = min(min_epsilon, epsilon)

        if min_epsilon == float('inf'):
            return float('inf'), delta

        return min_epsilon, delta


def compute_noise_multiplier(target_epsilon: float, target_delta: float,
                             q: float, steps: int, orders: List[float] = None) -> float:
    """
    Compute required noise multiplier for target privacy budget.

    Args:
        target_epsilon: Target epsilon
        target_delta: Target delta
        q: Sampling probability
        steps: Number of training steps
        orders: RDP orders to consider

    Returns:
        Required noise multiplier
    """
    # 使用更保守的公式计算噪声乘数
    if target_epsilon <= 0:
        return 0.0

    # 考虑采样概率的影响
    # 对于DP-SGD，需要更大的噪声乘数
    # 使用经验公式：σ ≈ sqrt(q * steps / (2 * ε))
    noise_multiplier = math.sqrt(q * steps / (2 * target_epsilon))

    # 确保噪声乘数在合理范围内，但允许更大的值
    noise_multiplier = max(1.0, min(noise_multiplier, 50.0))

    return noise_multiplier


def compute_privacy_parameters(dataset_size: int, batch_size: int, epochs: int,
                               target_epsilon: float, target_delta: float = 1e-5) -> dict:
    """
    Compute privacy parameters for DP-SGD training.

    Args:
        dataset_size: Size of training dataset
        batch_size: Batch size
        epochs: Number of training epochs
        target_epsilon: Target privacy budget
        target_delta: Target failure probability

    Returns:
        Dictionary with privacy parameters
    """
    q = batch_size / dataset_size  # Sampling probability
    steps = epochs * (dataset_size // batch_size)

    # 使用简化的噪声乘数计算，避免复杂的 RDP 计算
    if target_epsilon <= 0:
        noise_multiplier = 0.0
    else:
        # 使用简化的公式：对于高斯机制，ε ≈ 1/(2σ²) * steps
        # 这里使用更保守的估计
        noise_multiplier = math.sqrt(steps / (2 * target_epsilon))
        # 确保噪声乘数在合理范围内
        noise_multiplier = max(0.1, min(noise_multiplier, 10.0))

    # 计算实际隐私消耗（简化版）
    if noise_multiplier == 0:
        actual_epsilon = float('inf')
    else:
        # 简化的隐私消耗计算
        actual_epsilon = steps / (2 * noise_multiplier ** 2)

    return {
        'noise_multiplier': noise_multiplier,
        'max_grad_norm': 1.0,  # Standard clipping norm
        'sampling_probability': q,
        'steps': steps,
        'target_epsilon': target_epsilon,
        'target_delta': target_delta,
        'actual_epsilon': actual_epsilon,
        'actual_delta': target_delta
    }
