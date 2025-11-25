"""
Quantum Circuit to Photonic Pump Pattern Mapping
=================================================

This module maps quantum circuit operations to equivalent photonic pumping patterns
and demonstrates the correspondence between quantum gates and optical nonlinearities.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import json
import os

try:
    import pennylane as qml
    import torch
    HAS_QUANTUM = True
except ImportError:
    HAS_QUANTUM = False
    print("Note: pennylane/torch not available, using numpy only")

# ============================================================
# QUANTUM CIRCUIT STRUCTURE (from capstone_code3.py)
# ============================================================

class QuantumCircuitAnalyzer:
    """
    Analyzes the quantum circuit and extracts operation patterns.
    
    Quantum Circuit Structure:
    1. Input encoding: RY(features[i]) for each qubit
    2. For each layer (3 layers):
       a. RY rotations: RY(qparams[idx]) 
       b. CNOT entanglement: Conditional CNOTs
       c. RZ rotations: RZ(qparams[idx])
       d. RX rotations: RX(qparams[idx])
    3. Measurement: Probability measurement on all qubits
    """
    
    def __init__(self, n_qubits=4, n_layers=3):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.total_params = n_layers * 3 * n_qubits  # RY + RZ + RX per layer
        
    def get_circuit_operations(self) -> List[Dict]:
        """Extract all operations in the quantum circuit."""
        operations = []
        
        # 1. Input encoding layer
        for i in range(self.n_qubits):
            operations.append({
                'type': 'RY',
                'qubit': i,
                'param_type': 'feature',
                'param_index': i,
                'layer': 0,
                'purpose': 'Feature encoding',
                'angle_range': '[-π, π]'
            })
        
        # 2. Parameterized layers
        param_idx = 0
        for layer in range(self.n_layers):
            # a) RY rotations
            for i in range(self.n_qubits):
                operations.append({
                    'type': 'RY',
                    'qubit': i,
                    'param_type': 'trainable',
                    'param_index': param_idx,
                    'layer': layer + 1,
                    'purpose': 'Trainable rotation Y',
                    'angle_range': '[-2π, 2π]'
                })
                param_idx += 1
            
            # b) CNOT entanglement
            for i in range(self.n_qubits):
                for j in range(i+1, self.n_qubits):
                    if (i + j) % 2 == 0:
                        operations.append({
                            'type': 'CNOT',
                            'control': i,
                            'target': j,
                            'param_type': 'none',
                            'param_index': None,
                            'layer': layer + 1,
                            'purpose': 'Qubit entanglement',
                            'angle_range': 'N/A'
                        })
            
            # c) RZ rotations
            for i in range(self.n_qubits):
                operations.append({
                    'type': 'RZ',
                    'qubit': i,
                    'param_type': 'trainable',
                    'param_index': param_idx,
                    'layer': layer + 1,
                    'purpose': 'Trainable rotation Z',
                    'angle_range': '[-2π, 2π]'
                })
                param_idx += 1
            
            # d) RX rotations
            for i in range(self.n_qubits):
                operations.append({
                    'type': 'RX',
                    'qubit': i,
                    'param_type': 'trainable',
                    'param_index': param_idx,
                    'layer': layer + 1,
                    'purpose': 'Trainable rotation X',
                    'angle_range': '[-2π, 2π]'
                })
                param_idx += 1
        
        # 3. Measurement
        operations.append({
            'type': 'MEASURE',
            'qubits': list(range(self.n_qubits)),
            'param_type': 'none',
            'param_index': None,
            'layer': self.n_layers + 1,
            'purpose': 'Probability measurement',
            'angle_range': 'N/A'
        })
        
        return operations


# ============================================================
# PHOTONIC PUMP PATTERN OPERATIONS
# ============================================================

class PhotonicPumpMapper:
    """
    Maps quantum operations to photonic pump pattern operations.
    
    Photonic System Components:
    1. Elliptical field map (F) - Acts like quantum superposition
    2. Segmented mask (S) - Acts like quantum gates/entanglement
    3. Polynomial modulation (P) - Acts like parameterized rotations
    4. Nonlinear propagation - Acts like quantum interference
    """
    
    def __init__(self):
        self.mapping = {}
        self._build_mapping()
    
    def _build_mapping(self):
        """Build the quantum-to-photonic mapping."""
        
        # Rotation gates → Pump intensity modulation
        self.mapping['RY'] = {
            'photonic_operation': 'Y-axis pump modulation',
            'implementation': 'Vertical segmentation pattern',
            'parameters': 'Duty cycle (β) controls rotation angle',
            'physical_analog': 'Phase shift via refractive index modulation',
            'equation': 'β ∝ θ/2π where θ is rotation angle',
            'pump_pattern': 'Vertical stripes with β duty cycle'
        }
        
        self.mapping['RZ'] = {
            'photonic_operation': 'Z-axis pump modulation',
            'implementation': 'Phase-shift pump pattern',
            'parameters': 'Pump intensity (PI) controls rotation angle',
            'physical_analog': 'Optical phase accumulation',
            'equation': 'PI ∝ θ/π where θ is rotation angle',
            'pump_pattern': 'Uniform intensity modulation'
        }
        
        self.mapping['RX'] = {
            'photonic_operation': 'X-axis pump modulation',
            'implementation': 'Horizontal segmentation pattern',
            'parameters': 'Combined β and PI control rotation',
            'physical_analog': 'Polarization rotation',
            'equation': 'β·PI ∝ θ/2π',
            'pump_pattern': 'Horizontal stripes with intensity'
        }
        
        # CNOT gate → Directional coupling
        self.mapping['CNOT'] = {
            'photonic_operation': 'Directional waveguide coupling',
            'implementation': 'Overlapping elliptical field regions',
            'parameters': 'Coupling strength via overlap integral',
            'physical_analog': 'Evanescent field coupling between waveguides',
            'equation': 'Coupling ∝ ∫F_i(r)·F_j(r)·S(r)dr',
            'pump_pattern': 'Bridge pattern connecting two spatial modes'
        }
        
        # Feature encoding → Input intensity
        self.mapping['FEATURE_ENCODING'] = {
            'photonic_operation': 'Input light intensity',
            'implementation': 'Initial field amplitude at input port',
            'parameters': 'Normalized input intensity I_in[j]',
            'physical_analog': 'Optical power injection',
            'equation': 'I_in ∝ tanh(feature) for bounded input',
            'pump_pattern': 'Input port coupling efficiency'
        }
        
        # Measurement → Output detection
        self.mapping['MEASURE'] = {
            'photonic_operation': 'Output intensity measurement',
            'implementation': 'Photodetector at output ports',
            'parameters': 'Transmitted power at each output',
            'physical_analog': 'Optical power detection',
            'equation': 'P_out[i] ∝ |E_out[i]|²',
            'pump_pattern': 'Output port collection efficiency'
        }
        
        # Quantum superposition → Optical field distribution
        self.mapping['SUPERPOSITION'] = {
            'photonic_operation': 'Elliptical field distribution',
            'implementation': 'Elliptical field map F(r)',
            'parameters': 'R0 (ellipse radius), apodization power',
            'physical_analog': 'Spatial mode distribution',
            'equation': 'F(r) = cos(k_eff·R_ji) for R_ji < R0',
            'pump_pattern': 'Elliptical envelope of pump pattern'
        }
        
        # Quantum interference → Nonlinear propagation
        self.mapping['INTERFERENCE'] = {
            'photonic_operation': 'Nonlinear wave mixing',
            'implementation': 'Saturable absorption + optical gain',
            'parameters': 'α (absorption), g (gain), k_nl (nonlinearity)',
            'physical_analog': 'Intensity-dependent propagation',
            'equation': 'dI/dz = (g·pump - α)·I + k_nl·pump·I²',
            'pump_pattern': 'Segmented pattern creates interference'
        }


# ============================================================
# MAPPING QUANTUM CIRCUIT TO PUMP PATTERNS
# ============================================================

class QuantumToPumpMapper:
    """
    Complete mapping from quantum circuit to pump patterns.
    """
    
    def __init__(self, n_qubits=4, n_layers=3, M=10, N=128):
        self.quantum = QuantumCircuitAnalyzer(n_qubits, n_layers)
        self.photonic = PhotonicPumpMapper()
        self.M = M  # Number of output classes
        self.N = N  # Number of input features
        
    def map_quantum_params_to_pump_params(self, q_params: np.ndarray) -> Dict:
        """
        Map quantum parameters to pump pattern parameters.
        
        Args:
            q_params: Quantum parameters [total_q_params]
        
        Returns:
            Dictionary with β and PI values for each connection
        """
        n_connections = self.M * self.N
        param_idx = 0
        
        betas = np.zeros((self.M, self.N))
        PIs = np.zeros((self.M, self.N))
        
        # Map quantum rotation angles to pump parameters
        for i in range(self.M):
            for j in range(self.N):
                # Use cyclic mapping of quantum params to connections
                q_idx = param_idx % len(q_params)
                
                # RY rotation → β (duty cycle)
                # Normalize from [-2π, 2π] to [0.3, 0.95]
                theta_y = q_params[q_idx]
                betas[i, j] = 0.3 + 0.65 * (np.tanh(theta_y / np.pi) + 1) / 2
                
                # RZ rotation → PI (pump intensity)
                # Normalize from [-2π, 2π] to [0.2, 2.0]
                q_idx_z = (param_idx + self.quantum.n_qubits) % len(q_params)
                theta_z = q_params[q_idx_z]
                PIs[i, j] = 0.2 + 1.8 * (np.tanh(theta_z / np.pi) + 1) / 2
                
                param_idx += 1
        
        return {
            'betas': betas,
            'PIs': PIs,
            'mapping_info': {
                'quantum_params_used': len(q_params),
                'connections_generated': n_connections,
                'mapping_strategy': 'Cyclic with nonlinear transformation'
            }
        }
    
    def generate_operation_correspondence_table(self) -> str:
        """Generate a detailed correspondence table."""
        operations = self.quantum.get_circuit_operations()
        
        table = "="*100 + "\n"
        table += "QUANTUM CIRCUIT TO PHOTONIC PUMP PATTERN MAPPING\n"
        table += "="*100 + "\n\n"
        
        # Group by operation type
        op_types = {}
        for op in operations:
            op_type = op['type']
            if op_type not in op_types:
                op_types[op_type] = []
            op_types[op_type].append(op)
        
        for op_type, ops in op_types.items():
            table += f"\n{'='*100}\n"
            table += f"QUANTUM GATE: {op_type}\n"
            table += f"{'='*100}\n"
            table += f"Number of occurrences: {len(ops)}\n\n"
            
            if op_type in self.photonic.mapping:
                mapping = self.photonic.mapping[op_type]
                table += f"PHOTONIC EQUIVALENT:\n"
                table += f"  Operation:     {mapping['photonic_operation']}\n"
                table += f"  Implementation: {mapping['implementation']}\n"
                table += f"  Parameters:    {mapping['parameters']}\n"
                table += f"  Physical:      {mapping['physical_analog']}\n"
                table += f"  Equation:      {mapping['equation']}\n"
                table += f"  Pattern:       {mapping['pump_pattern']}\n"
            else:
                table += f"PHOTONIC EQUIVALENT: [Composite operation]\n"
            
            # Show first few instances
            table += f"\nInstances:\n"
            for idx, op in enumerate(ops[:3]):
                table += f"  {idx+1}. Layer {op['layer']}: {op['purpose']}"
                if 'qubit' in op:
                    table += f" (qubit {op['qubit']})"
                elif 'control' in op:
                    table += f" (control={op['control']}, target={op['target']})"
                table += "\n"
            
            if len(ops) > 3:
                table += f"  ... and {len(ops)-3} more\n"
        
        return table
    
    def visualize_mapping(self, q_params: np.ndarray = None):
        """Visualize the quantum-to-photonic mapping."""
        if q_params is None:
            q_params = np.random.randn(self.quantum.total_params) * 0.5
        
        # Get pump parameters
        pump_params = self.map_quantum_params_to_pump_params(q_params)
        betas = pump_params['betas']
        PIs = pump_params['PIs']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Quantum parameter distribution
        ax = axes[0, 0]
        ax.hist(q_params, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax.set_xlabel('Quantum Parameter Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Quantum Circuit Parameters\n(RY, RZ, RX angles)')
        ax.grid(True, alpha=0.3)
        
        # 2. Beta (duty cycle) distribution
        ax = axes[0, 1]
        im1 = ax.imshow(betas, cmap='viridis', aspect='auto')
        ax.set_xlabel('Input Features (N=128)')
        ax.set_ylabel('Output Classes (M=10)')
        ax.set_title('Pump Pattern β (Duty Cycle)\nMapped from RY rotations')
        plt.colorbar(im1, ax=ax, label='β value')
        
        # 3. PI (pump intensity) distribution
        ax = axes[0, 2]
        im2 = ax.imshow(PIs, cmap='plasma', aspect='auto')
        ax.set_xlabel('Input Features (N=128)')
        ax.set_ylabel('Output Classes (M=10)')
        ax.set_title('Pump Pattern PI (Intensity)\nMapped from RZ rotations')
        plt.colorbar(im2, ax=ax, label='PI value')
        
        # 4. Circuit structure
        ax = axes[1, 0]
        operations = self.quantum.get_circuit_operations()
        op_counts = {}
        for op in operations:
            t = op['type']
            op_counts[t] = op_counts.get(t, 0) + 1
        
        colors = {'RY': 'red', 'RZ': 'blue', 'RX': 'green', 'CNOT': 'orange', 'MEASURE': 'purple'}
        ax.bar(op_counts.keys(), op_counts.values(), 
               color=[colors.get(k, 'gray') for k in op_counts.keys()])
        ax.set_ylabel('Count')
        ax.set_title('Quantum Circuit Operations')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 5. Beta histogram
        ax = axes[1, 1]
        ax.hist(betas.flatten(), bins=30, alpha=0.7, color='green', edgecolor='black')
        ax.set_xlabel('β (Duty Cycle)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of β Values\n(Segmentation Duty Cycle)')
        ax.grid(True, alpha=0.3)
        
        # 6. PI histogram
        ax = axes[1, 2]
        ax.hist(PIs.flatten(), bins=30, alpha=0.7, color='red', edgecolor='black')
        ax.set_xlabel('PI (Pump Intensity)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of PI Values\n(Normalized Pump Power)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('quantum_pump_mapping_visualization.png', dpi=150, bbox_inches='tight')
        print("Visualization saved to: quantum_pump_mapping_visualization.png")
        plt.show()
        
        return pump_params


# ============================================================
# INTEGRATION WITH EXISTING PUMP PATTERN CODE
# ============================================================

def integrate_quantum_params_into_pump_generation(q_mean_path: str, 
                                                  output_dir: str = "./pump_patterns_cifar10"):
    """
    Load quantum parameters from capstone code and generate pump patterns.
    
    Args:
        q_mean_path: Path to saved quantum parameters (q_mean from checkpoint)
        output_dir: Directory to save pump patterns
    """
    print("Loading quantum parameters from trained model...")
    
    # Load quantum parameters
    if os.path.exists(q_mean_path):
        if HAS_QUANTUM:
            checkpoint = torch.load(q_mean_path, map_location='cpu')
            if 'q_mean' in checkpoint:
                q_params = checkpoint['q_mean'].numpy()
            else:
                q_params = checkpoint.numpy()
            print(f"Loaded {len(q_params)} quantum parameters")
        else:
            print("PyTorch not available, cannot load checkpoint")
            q_params = np.random.randn(36) * 0.5
    else:
        print("Checkpoint not found, using random initialization")
        q_params = np.random.randn(36) * 0.5  # 3 layers * 3 rotations * 4 qubits
    
    # Create mapper
    mapper = QuantumToPumpMapper(n_qubits=4, n_layers=3, M=10, N=128)
    
    # Generate correspondence table
    print("\n" + mapper.generate_operation_correspondence_table())
    
    # Map to pump parameters
    pump_params = mapper.visualize_mapping(q_params)
    
    # Save mapping results
    os.makedirs(output_dir, exist_ok=True)
    
    mapping_info = {
        'quantum_params': q_params.tolist(),
        'betas': pump_params['betas'].tolist(),
        'PIs': pump_params['PIs'].tolist(),
        'mapping_info': pump_params['mapping_info'],
        'correspondence': mapper.photonic.mapping
    }
    
    with open(os.path.join(output_dir, 'quantum_pump_mapping.json'), 'w') as f:
        json.dump(mapping_info, f, indent=2)
    
    print(f"\n✅ Mapping results saved to {output_dir}/quantum_pump_mapping.json")
    
    return pump_params


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("Quantum Circuit to Photonic Pump Pattern Mapping")
    print("="*60)
    
    # Create mapper
    mapper = QuantumToPumpMapper(n_qubits=4, n_layers=3, M=10, N=128)
    
    # Generate and print correspondence table
    print(mapper.generate_operation_correspondence_table())
    
    # Try to load trained quantum parameters
    checkpoint_paths = [
        "hybrid_full_checkpoint.pth",
        "hybrid_best_checkpoint.pth"
    ]
    
    q_params = None
    for path in checkpoint_paths:
        if os.path.exists(path):
            print(f"\n✅ Found checkpoint: {path}")
            try:
                integrate_quantum_params_into_pump_generation(path)
                break
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                continue
    
    if q_params is None:
        print("\n⚠️  No trained checkpoint found, using random parameters for demonstration")
        mapper.visualize_mapping()
    
    print("\n" + "="*60)
    print("Mapping complete! Check the generated files:")
    print("  - quantum_pump_mapping_visualization.png")
    print("  - pump_patterns_cifar10/quantum_pump_mapping.json")
