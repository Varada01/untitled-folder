"""
Quantum Dot Input Generator for Photonic Neural Network
Simulates quantum dot arrays that convert CIFAR-10 features into optical intensities
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List
import pickle

# ============================================================
# Quantum Dot Physical Parameters
# ============================================================
@dataclass
class QuantumDotParams:
    """Physical parameters for InAs/GaAs quantum dots"""
    # Emission properties
    wavelength: float = 1.55e-6  # Emission wavelength (m) - telecom band
    linewidth: float = 0.1e-9    # Spectral linewidth (m) - ~0.1 nm
    lifetime: float = 1e-9       # Radiative lifetime (s) - ~1 ns
    
    # Quantum efficiency
    internal_qe: float = 0.85    # Internal quantum efficiency
    extraction_eff: float = 0.6  # Light extraction efficiency
    total_qe: float = 0.51       # Total QE = internal × extraction
    
    # Saturation properties
    I_sat: float = 1.0           # Saturation intensity (normalized)
    pump_threshold: float = 0.1  # Minimum pump to trigger emission
    
    # Spatial properties
    dot_diameter: float = 20e-9  # Quantum dot size (m) - ~20 nm
    array_pitch: float = 10e-6   # Spacing between dots in array (m) - 10 μm
    
    # Temporal response
    rise_time: float = 0.5e-9    # Turn-on time (s)
    fall_time: float = 1.5e-9    # Turn-off time (s)

# ============================================================
# Quantum Dot Emission Model
# ============================================================
class QuantumDot:
    """Single quantum dot emitter model"""
    
    def __init__(self, params: QuantumDotParams, position: Tuple[float, float]):
        self.params = params
        self.position = position  # (x, y) coordinates
        self.state = 0.0  # Current emission intensity
        
    def emission_response(self, pump_current: float) -> float:
        """
        Calculate emission intensity from pump current
        
        Model: I_out = QE × I_pump × [1 / (1 + I_pump/I_sat)]
        - Linear at low pump (below saturation)
        - Saturates at high pump (carrier depletion)
        
        Args:
            pump_current: Normalized pump current (0 to 1 scale)
            
        Returns:
            Emission intensity (normalized)
        """
        if pump_current < self.params.pump_threshold:
            return 0.0
        
        # Effective pump above threshold
        effective_pump = pump_current - self.params.pump_threshold
        
        # Saturable emission model
        emission = (self.params.total_qe * effective_pump * 
                   (1.0 / (1.0 + effective_pump / self.params.I_sat)))
        
        return emission
    
    def spectral_profile(self, wavelengths: np.ndarray) -> np.ndarray:
        """
        Lorentzian spectral profile of emission
        
        Args:
            wavelengths: Array of wavelengths (m)
            
        Returns:
            Normalized spectral intensity
        """
        lambda0 = self.params.wavelength
        gamma = self.params.linewidth / 2
        
        # Lorentzian profile
        spectrum = (gamma**2) / ((wavelengths - lambda0)**2 + gamma**2)
        spectrum /= np.max(spectrum)  # Normalize
        
        return spectrum
    
    def temporal_response(self, time: np.ndarray, pump_signal: np.ndarray) -> np.ndarray:
        """
        Temporal response to time-varying pump signal
        
        Args:
            time: Time array (s)
            pump_signal: Pump current vs time
            
        Returns:
            Emission intensity vs time
        """
        emission = np.zeros_like(time)
        state = 0.0
        
        dt = time[1] - time[0] if len(time) > 1 else 1e-12
        
        for i, pump in enumerate(pump_signal):
            target_emission = self.emission_response(pump)
            
            # Exponential rise/fall dynamics
            if target_emission > state:
                tau = self.params.rise_time
            else:
                tau = self.params.fall_time
            
            # First-order dynamics: dI/dt = (target - I) / tau
            state += (target_emission - state) * (dt / tau)
            emission[i] = state
        
        return emission

# ============================================================
# Quantum Dot Array for Multi-Input Encoding
# ============================================================
class QuantumDotArray:
    """Array of quantum dots for encoding neural network inputs"""
    
    def __init__(self, n_inputs: int, params: QuantumDotParams = None):
        """
        Initialize quantum dot array
        
        Args:
            n_inputs: Number of input channels (e.g., 128 for CIFAR-10)
            params: Quantum dot parameters (uses defaults if None)
        """
        self.n_inputs = n_inputs
        self.params = params or QuantumDotParams()
        
        # Create quantum dots in linear array
        self.dots = []
        for i in range(n_inputs):
            y_pos = i * self.params.array_pitch
            position = (0.0, y_pos)  # All at x=0 (input edge)
            self.dots.append(QuantumDot(self.params, position))
        
        print(f"Initialized quantum dot array:")
        print(f"  - Number of dots: {n_inputs}")
        print(f"  - Array length: {(n_inputs-1) * self.params.array_pitch * 1e6:.1f} μm")
        print(f"  - Emission wavelength: {self.params.wavelength * 1e9:.1f} nm")
        print(f"  - Quantum efficiency: {self.params.total_qe * 100:.1f}%")
    
    def encode_features(self, features: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Encode feature vector as optical intensities via quantum dots
        
        Args:
            features: Feature vector (length n_inputs)
            normalize: Whether to normalize features to [0, 1] range
            
        Returns:
            Array of emission intensities from each quantum dot
        """
        if len(features) != self.n_inputs:
            raise ValueError(f"Expected {self.n_inputs} features, got {len(features)}")
        
        # Normalize features if requested
        if normalize:
            f_min, f_max = features.min(), features.max()
            if f_max > f_min:
                features = (features - f_min) / (f_max - f_min)
            else:
                features = np.zeros_like(features)
        
        # Convert features to pump currents (with offset to stay above threshold)
        pump_currents = 0.2 + 0.8 * features  # Map to [0.2, 1.0] range
        
        # Generate emissions from each quantum dot
        emissions = np.array([
            dot.emission_response(pump) 
            for dot, pump in zip(self.dots, pump_currents)
        ])
        
        return emissions
    
    def encode_cifar10_image(self, image_features: np.ndarray, 
                            pca_transformer=None) -> np.ndarray:
        """
        Encode CIFAR-10 image (after PCA reduction) as quantum dot emissions
        
        Args:
            image_features: PCA-reduced feature vector (128-dim for CIFAR-10)
            pca_transformer: Optional PCA transformer to apply if features are raw
            
        Returns:
            Quantum dot emission intensities
        """
        # Apply PCA if transformer provided and features are raw (3072-dim)
        if pca_transformer is not None and len(image_features) == 3072:
            image_features = pca_transformer.transform(image_features.reshape(1, -1))[0]
        
        # Encode via quantum dots
        emissions = self.encode_features(image_features, normalize=True)
        
        return emissions
    
    def visualize_emission_pattern(self, features: np.ndarray, 
                                   title: str = "Quantum Dot Emission Pattern"):
        """
        Visualize the spatial emission pattern for given features
        
        Args:
            features: Input feature vector
            title: Plot title
        """
        emissions = self.encode_features(features)
        positions = np.array([dot.position[1] for dot in self.dots]) * 1e6  # Convert to μm
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Bar chart of emissions
        axes[0].bar(range(self.n_inputs), emissions, color='red', alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Quantum Dot Index', fontweight='bold')
        axes[0].set_ylabel('Emission Intensity (normalized)', fontweight='bold')
        axes[0].set_title('Emission Intensities', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim([-1, min(50, self.n_inputs)])  # Show first 50 dots
        
        # Plot 2: Spatial emission map
        axes[1].scatter(np.zeros(self.n_inputs), positions, 
                       s=emissions*200, c=emissions, cmap='hot', 
                       alpha=0.8, edgecolor='black', linewidth=0.5)
        axes[1].set_xlabel('X Position (μm)', fontweight='bold')
        axes[1].set_ylabel('Y Position (μm)', fontweight='bold')
        axes[1].set_title('Spatial Emission Map', fontweight='bold')
        axes[1].set_xlim([-50, 50])
        cbar = plt.colorbar(axes[1].collections[0], ax=axes[1])
        cbar.set_label('Emission Intensity', fontweight='bold')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig

# ============================================================
# CIFAR-10 Integration
# ============================================================
def load_cifar10_sample():
    """Load a sample CIFAR-10 image for testing"""
    import torchvision
    import torchvision.transforms as transforms
    
    # CIFAR-10 normalization stats
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2470, 0.2435, 0.2616))
    ])
    
    dataset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                           download=True, transform=transform)
    
    # Get first test image
    image, label = dataset[0]
    features = image.numpy().flatten()  # 3072 features
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    return features, label, class_names[label]

def load_pca_transformer():
    """Load pre-trained PCA transformer"""
    try:
        with open('pca_transformer_cifar10.pkl', 'rb') as f:
            pca = pickle.load(f)
        print(f"Loaded PCA transformer: {pca.n_components_} components")
        return pca
    except FileNotFoundError:
        print("Warning: PCA transformer not found. Will use raw features.")
        return None

# ============================================================
# Main Demo
# ============================================================
if __name__ == "__main__":
    print("="*70)
    print(" QUANTUM DOT INPUT GENERATOR FOR PHOTONIC NEURAL NETWORK")
    print("="*70)
    print()
    
    # Initialize quantum dot array for CIFAR-10 (128 inputs after PCA)
    n_inputs = 128
    qd_array = QuantumDotArray(n_inputs=n_inputs)
    print()
    
    # Test 1: Encode random features
    print("TEST 1: Encoding Random Features")
    print("-" * 70)
    random_features = np.random.randn(n_inputs)
    emissions = qd_array.encode_features(random_features)
    print(f"Input features: min={random_features.min():.3f}, max={random_features.max():.3f}")
    print(f"QD emissions:   min={emissions.min():.3f}, max={emissions.max():.3f}, mean={emissions.mean():.3f}")
    print(f"Total optical power: {emissions.sum():.2f} (normalized units)")
    print()
    
    # Visualize
    fig1 = qd_array.visualize_emission_pattern(random_features, 
                                               "Quantum Dot Emission: Random Features")
    plt.savefig('qd_emission_random.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: qd_emission_random.png")
    print()
    
    # Test 2: Load and encode actual CIFAR-10 image
    print("TEST 2: Encoding CIFAR-10 Image")
    print("-" * 70)
    try:
        features_raw, label, class_name = load_cifar10_sample()
        pca = load_pca_transformer()
        
        if pca is not None:
            # Apply PCA reduction
            features_pca = pca.transform(features_raw.reshape(1, -1))[0]
            emissions_cifar = qd_array.encode_cifar10_image(features_pca)
        else:
            # Use first 128 raw features if no PCA
            features_pca = features_raw[:n_inputs]
            emissions_cifar = qd_array.encode_features(features_pca)
        
        print(f"Image class: {class_name} (label {label})")
        print(f"PCA features: {len(features_pca)} dimensions")
        print(f"QD emissions: min={emissions_cifar.min():.3f}, max={emissions_cifar.max():.3f}")
        print(f"Total optical power: {emissions_cifar.sum():.2f}")
        print()
        
        # Visualize
        fig2 = qd_array.visualize_emission_pattern(features_pca,
                                                   f"Quantum Dot Emission: {class_name.upper()}")
        plt.savefig('qd_emission_cifar10.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: qd_emission_cifar10.png")
        
    except Exception as e:
        print(f"Could not load CIFAR-10: {e}")
        print("Skipping CIFAR-10 test...")
    print()
    
    # Test 3: Spectral and temporal characteristics
    print("TEST 3: Quantum Dot Physical Characteristics")
    print("-" * 70)
    
    # Single dot for detailed analysis
    test_dot = qd_array.dots[0]
    
    # Spectral profile
    wavelengths = np.linspace(1.54e-6, 1.56e-6, 500)  # Around 1.55 μm
    spectrum = test_dot.spectral_profile(wavelengths)
    
    fig3, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Spectral profile
    axes[0, 0].plot(wavelengths * 1e9, spectrum, 'b-', linewidth=2)
    axes[0, 0].axvline(test_dot.params.wavelength * 1e9, color='r', 
                       linestyle='--', label='Center wavelength')
    axes[0, 0].set_xlabel('Wavelength (nm)', fontweight='bold')
    axes[0, 0].set_ylabel('Normalized Intensity', fontweight='bold')
    axes[0, 0].set_title('Emission Spectrum', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Pump-emission curve (saturation)
    pump_range = np.linspace(0, 2.0, 100)
    emission_curve = [test_dot.emission_response(p) for p in pump_range]
    axes[0, 1].plot(pump_range, emission_curve, 'r-', linewidth=2)
    axes[0, 1].axvline(test_dot.params.I_sat, color='orange', 
                       linestyle='--', label='Saturation intensity')
    axes[0, 1].axhline(test_dot.params.total_qe, color='green', 
                       linestyle='--', alpha=0.5, label='Max QE')
    axes[0, 1].set_xlabel('Pump Current (normalized)', fontweight='bold')
    axes[0, 1].set_ylabel('Emission Intensity', fontweight='bold')
    axes[0, 1].set_title('Saturation Curve', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Temporal response (step input)
    time = np.linspace(0, 10e-9, 1000)  # 10 ns
    pump_signal = np.ones_like(time)
    pump_signal[time < 2e-9] = 0.0  # Turn on at 2 ns
    pump_signal[time > 7e-9] = 0.0  # Turn off at 7 ns
    pump_signal *= 0.5  # 50% pump level
    
    temporal_emission = test_dot.temporal_response(time, pump_signal)
    axes[1, 0].plot(time * 1e9, pump_signal, 'b--', linewidth=2, label='Pump signal')
    axes[1, 0].plot(time * 1e9, temporal_emission, 'r-', linewidth=2, label='Emission')
    axes[1, 0].set_xlabel('Time (ns)', fontweight='bold')
    axes[1, 0].set_ylabel('Intensity (normalized)', fontweight='bold')
    axes[1, 0].set_title('Temporal Response', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Array encoding efficiency
    test_features = np.linspace(0, 1, n_inputs)
    test_emissions = qd_array.encode_features(test_features, normalize=False)
    axes[1, 1].plot(test_features, test_emissions, 'go', markersize=4, alpha=0.6)
    axes[1, 1].plot([0, 1], [0, test_dot.params.total_qe], 'r--', 
                    linewidth=2, label='Ideal linear')
    axes[1, 1].set_xlabel('Input Feature (normalized)', fontweight='bold')
    axes[1, 1].set_ylabel('Emission Intensity', fontweight='bold')
    axes[1, 1].set_title('Encoding Transfer Function', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Quantum Dot Physical Characteristics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('qd_characteristics.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: qd_characteristics.png")
    print()
    
    # Summary
    print("="*70)
    print(" SUMMARY")
    print("="*70)
    print(f"Quantum Dot Array: {n_inputs} emitters")
    print(f"Wavelength: {test_dot.params.wavelength * 1e9:.1f} nm (telecom band)")
    print(f"Quantum Efficiency: {test_dot.params.total_qe * 100:.1f}%")
    print(f"Response Time: {test_dot.params.rise_time * 1e9:.2f} ns (rise)")
    print(f"Array Pitch: {test_dot.params.array_pitch * 1e6:.1f} μm")
    print(f"Total Array Length: {(n_inputs-1) * test_dot.params.array_pitch * 1e6:.1f} μm")
    print()
    print("Generated Files:")
    print("  • qd_emission_random.png - Random feature encoding")
    print("  • qd_emission_cifar10.png - CIFAR-10 image encoding")
    print("  • qd_characteristics.png - Physical properties")
    print()
    print("="*70)
    
    plt.show()
