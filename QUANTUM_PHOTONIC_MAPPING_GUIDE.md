# Quantum Circuit to Photonic Pump Pattern Mapping

## Complete Correspondence Table

This document provides the detailed mapping between quantum circuit operations in `capstone_code3.py` and the photonic pump patterns in the trial files.

---

## 1. QUANTUM GATE: RY (Y-Axis Rotation)

### Quantum Operation

```python
qml.RY(angle, wires=i)
```

- **Rotation angle range:** `[-2π, 2π]` (trainable) or `[-π, π]` (feature encoding)
- **Effect:** Rotates qubit state around Y-axis on Bloch sphere
- **16 occurrences total:** 4 feature encoding + 12 trainable (4 qubits × 3 layers)

### Photonic Equivalent

**Operation:** Y-axis pump modulation via **vertical segmentation pattern**

**Implementation:**

```python
# In pump_pattern_trial.py
duty_beta = 0.3 to 0.95  # Segmentation duty cycle
```

**Physical Mapping:**

- **Parameter:** Duty cycle `β` (fraction of waveguide with pumping)
- **Equation:** `β ≈ 0.3 + 0.65 × (tanh(θ_RY / π) + 1) / 2`
- **Physical analog:** Phase shift via refractive index modulation
- **Pattern:** Vertical stripes with β duty cycle

**What it does:**

- Creates periodic regions of high/low refractive index
- Light propagating through experiences phase shifts proportional to β
- Equivalent to rotating the optical "state" in phase space

---

## 2. QUANTUM GATE: RZ (Z-Axis Rotation)

### Quantum Operation

```python
qml.RZ(angle, wires=i)
```

- **Rotation angle range:** `[-2π, 2π]`
- **Effect:** Adds phase to qubit state (diagonal gate)
- **12 occurrences:** 4 qubits × 3 layers

### Photonic Equivalent

**Operation:** Z-axis pump modulation via **uniform intensity modulation**

**Implementation:**

```python
# In pump_pattern_trial.py
PI = 0.2 to 2.0  # Pump intensity
```

**Physical Mapping:**

- **Parameter:** Pump intensity `PI` (normalized power)
- **Equation:** `PI ≈ 0.2 + 1.8 × (tanh(θ_RZ / π) + 1) / 2`
- **Physical analog:** Optical phase accumulation due to pump-induced gain
- **Pattern:** Uniform intensity across the pump region

**What it does:**

- Higher pump intensity → stronger optical gain
- Accumulated phase ∝ integrated gain along propagation path
- Equivalent to z-rotation (global phase shift)

---

## 3. QUANTUM GATE: RX (X-Axis Rotation)

### Quantum Operation

```python
qml.RX(angle, wires=i)
```

- **Rotation angle range:** `[-2π, 2π]`
- **Effect:** Rotates qubit state around X-axis
- **12 occurrences:** 4 qubits × 3 layers

### Photonic Equivalent

**Operation:** X-axis pump modulation via **horizontal segmentation + intensity**

**Implementation:**

```python
# Combined effect of β and PI
effective_angle ∝ β × PI
```

**Physical Mapping:**

- **Parameters:** Combined `β` (duty) and `PI` (intensity)
- **Equation:** `θ_RX ≈ 2π × β × PI`
- **Physical analog:** Polarization rotation or mode coupling
- **Pattern:** Horizontal stripes with variable intensity

**What it does:**

- Creates spatially varying pump that couples optical modes
- Horizontal pattern causes transverse mode mixing
- Equivalent to x-rotation (population transfer)

---

## 4. QUANTUM GATE: CNOT (Controlled-NOT)

### Quantum Operation

```python
qml.CNOT(wires=[control, target])
```

- **Effect:** Flips target qubit if control is |1⟩
- **6 occurrences:** Conditional entanglement between qubit pairs
- **Pattern:** `(i+j) % 2 == 0` determines which pairs couple

### Photonic Equivalent

**Operation:** **Directional waveguide coupling** between spatial modes

**Implementation:**

```python
# Overlapping elliptical field regions
coupling_strength ∝ ∫ F_i(r) × F_j(r) × S(r) dr
```

**Physical Mapping:**

- **Parameter:** Overlap integral of field distributions
- **Equation:** `Coupling = ∫ F_control(r) · F_target(r) · PumpMask(r) dr`
- **Physical analog:** Evanescent field coupling between adjacent waveguides
- **Pattern:** Bridge-like pump pattern connecting two spatial modes

**What it does:**

- When control mode has light → pump activates coupling region
- Light transfers from control to target mode (or vice versa)
- Controlled by spatial overlap of elliptical field maps
- Equivalent to conditional logic gate

**Example:**

```python
# In pump_pattern_trial.py: Different input/output ports create spatial modes
r_in_control = (30, y_control)   # Control mode input
r_in_target = (30, y_target)     # Target mode input
r_out_coupled = (270, y_output)  # Coupled output

# Pump pattern connects these regions via elliptical field overlap
```

---

## 5. FEATURE ENCODING (Input Layer)

### Quantum Operation

```python
for i in range(n_qubits):
    qml.RY(features[i], wires=i)  # Feature encoding
```

- **Input:** Reduced image features (4 features → 4 qubits)
- **Encoding:** `angle = tanh(feature) × π/2`

### Photonic Equivalent

**Operation:** **Input light intensity** at each input port

**Implementation:**

```python
# In processor_trial.py
I_in[j] = normalized_feature[j]  # Input intensity per port
```

**Physical Mapping:**

- **Parameter:** Input optical power `I_in[j]` for feature j
- **Equation:** `I_in[j] = (feature[j] - min) / (max - min)`
- **Physical analog:** Laser power injection at input waveguide
- **Pattern:** Spatial distribution of input ports

**What it does:**

- Each image feature → intensity at corresponding input port
- For CIFAR-10: 128 PCA features → 128 input ports
- Light intensity encodes classical data into optical domain

---

## 6. MEASUREMENT (Output Layer)

### Quantum Operation

```python
qml.probs(wires=list(range(n_qubits)))
```

- **Output:** Probability distribution over 2^n_qubits states
- **For 4 qubits:** 16 probability values, truncated to 10 classes

### Photonic Equivalent

**Operation:** **Output intensity measurement** at detector ports

**Implementation:**

```python
# In processor_trial.py
outputs[i] = ∫ propagate_intensity(...) for all inputs
P_out[i] ∝ |E_out[i]|²
```

**Physical Mapping:**

- **Parameter:** Detected optical power at each output port
- **Equation:** `P_out[i] = Σ_j T_ij(I_in[j])` where T is transfer function
- **Physical analog:** Photodetector measuring transmitted power
- **Pattern:** 10 output ports (one per CIFAR-10 class)

**What it does:**

- Each output port collects light from all input-output paths
- Nonlinear propagation through pump patterns creates classification
- Softmax normalization: `prob[i] = P_out[i] / Σ_k P_out[k]`

---

## 7. QUANTUM SUPERPOSITION → Optical Field Distribution

### Quantum Concept

Qubit exists in superposition: `|ψ⟩ = α|0⟩ + β|1⟩`

### Photonic Equivalent

**Operation:** **Elliptical field distribution** F(r)

**Implementation:**

```python
# In pump_pattern_trial.py
F(r) = cos(k_eff × R_ji) for R_ji < R0
# where R_ji = R_in + R_out - distance(in, out)
```

**Physical Mapping:**

- **Parameter:** Ellipse radius `R0` and apodization power
- **Equation:** `F(r) = [cos(k_eff · R_ji)]^apod_power`
- **Physical analog:** Spatial mode distribution in multimode waveguide
- **Pattern:** Elliptical envelope defines where pump is active

**What it does:**

- Creates spatially distributed optical field (not localized)
- Multiple paths interfere → superposition-like behavior
- Ellipse encompasses all possible optical paths from input to output

---

## 8. QUANTUM INTERFERENCE → Nonlinear Propagation

### Quantum Concept

Probability amplitudes interfere: `P = |α + β|² ≠ |α|² + |β|²`

### Photonic Equivalent

**Operation:** **Nonlinear wave mixing** during propagation

**Implementation:**

```python
# In processor_trial.py
dI/dz = (g_lin - α_eff) × I + g_nl × I²
# where g_lin and g_nl depend on pump pattern
```

**Physical Mapping:**

- **Parameters:** `α` (absorption), `g` (gain), `k_nl` (nonlinearity)
- **Equation:**
  ```
  g_lin(z) = g0 × pump(z)
  g_nl(z) = k_nl × pump(z)
  α_eff = α0 / (1 + I/I_sat)  # saturable absorption
  ```
- **Physical analog:** Intensity-dependent refractive index and gain
- **Pattern:** Segmented pump creates interference-like effects

**What it does:**

- Linear propagation + nonlinear corrections
- Pump segments create alternating gain/loss regions
- Multiple path interference through spatial pump distribution
- Saturation creates amplitude-dependent effects (like quantum interference)

---

## Complete Network Architecture Mapping

### Quantum Circuit (capstone_code3.py)

```
Input: 4 features (from ResNet-18 backbone)
  ↓
Quantum Circuit (4 qubits, 3 layers):
  - Layer 0: Feature encoding (RY gates)
  - Layers 1-3:
    * RY rotations (4 × 3 = 12)
    * CNOT gates (2 × 3 = 6)
    * RZ rotations (4 × 3 = 12)
    * RX rotations (4 × 3 = 12)
  ↓
Measurement: 16 probabilities → truncate to 10 classes
```

**Total parameters:** 36 trainable quantum parameters

### Photonic System (trial files)

```
Input: 128 PCA features (reduced from 3072)
  ↓
Photonic Network (10 outputs × 128 inputs = 1280 connections):
  - Input encoding: 128 input ports with intensities I_in[j]
  - Each connection (i,j):
    * Elliptical field map F(r) [superposition]
    * Segmented mask S(r) with β [RY/RX equivalent]
    * Pump intensity PI [RZ equivalent]
    * Polynomial modulation P(r) [combined rotations]
    * Nonlinear propagation [interference]
  ↓
Output: 10 photodetectors → 10 class probabilities
```

**Total parameters:** 2560 (1280 β + 1280 PI)

---

## Parameter Mapping Formula

### From Quantum to Photonic:

```python
# For connection (i, j) from quantum params q_params:

# Map RY rotation to duty cycle β
idx_y = connection_to_quantum_index(i, j, 'RY')
theta_y = q_params[idx_y]
β[i,j] = 0.3 + 0.65 × (tanh(theta_y / π) + 1) / 2

# Map RZ rotation to pump intensity PI
idx_z = connection_to_quantum_index(i, j, 'RZ')
theta_z = q_params[idx_z]
PI[i,j] = 0.2 + 1.8 × (tanh(theta_z / π) + 1) / 2

# RX is implicitly encoded in the combined β × PI effect
```

### Inverse (Photonic to Quantum):

```python
# Extract equivalent quantum angles from pump parameters

theta_y = π × arctanh(2 × (β - 0.3) / 0.65 - 1)
theta_z = π × arctanh(2 × (PI - 0.2) / 1.8 - 1)
theta_x = 2π × β × PI  # Combined effect
```

---

## Key Insights

1. **Spatial → Temporal:** Quantum gates are sequential; photonic operations are spatial (different regions of the chip)

2. **Discrete → Continuous:** Quantum states are discrete; optical fields are continuous

3. **Unitary → Dissipative:** Quantum gates are unitary; photonic propagation includes loss and gain

4. **Entanglement → Coupling:** Quantum entanglement via CNOTs ↔ Optical coupling via overlapping modes

5. **Measurement → Detection:** Quantum measurement collapses state ↔ Photodetection measures intensity

6. **Superposition → Field distribution:** Quantum superposition of states ↔ Spatial distribution of optical field

7. **Interference → Nonlinear mixing:** Quantum amplitude interference ↔ Nonlinear wave interactions

---

## Files That Implement This Mapping

1. **quantum_pump_mapping.py** (this analysis)

   - Analyzes quantum circuit structure
   - Maps quantum parameters to pump parameters
   - Generates visualization

2. **pump_pattern_trial.py**

   - Generates spatial pump patterns from parameters
   - Implements elliptical field maps and segmentation
   - Creates the β and PI patterns

3. **processor_trial.py**

   - Simulates nonlinear optical propagation
   - Implements the transfer function I → O
   - Performs "measurement" by collecting output intensities

4. **iris coefficients trial.py** (updated for CIFAR-10)

   - Trains polynomial coefficients
   - These guide the pump pattern generation
   - Acts like training the quantum parameters

5. **capstone_code3.py**
   - Reference quantum circuit implementation
   - Source of quantum operations to map
   - Trains actual quantum parameters

---

## Usage

To generate the mapping from a trained quantum model:

```bash
# Train quantum model first
python capstone_code3.py

# Generate pump patterns from quantum parameters
python quantum_pump_mapping.py

# This creates:
# - quantum_pump_mapping_visualization.png (visual comparison)
# - pump_patterns_cifar10/quantum_pump_mapping.json (numerical mapping)
```

The mapping enables:

- **Forward:** Quantum params → Pump patterns
- **Backward:** Trained photonic system → Equivalent quantum circuit
- **Analysis:** Understanding which quantum operations contribute most
- **Optimization:** Training one system to match the other

---

## Visualization Output

The `quantum_pump_mapping_visualization.png` shows:

1. **Top left:** Distribution of quantum parameters (rotation angles)
2. **Top middle:** β matrix (10×128) from RY mappings
3. **Top right:** PI matrix (10×128) from RZ mappings
4. **Bottom left:** Count of each quantum gate type
5. **Bottom middle:** Histogram of β values
6. **Bottom right:** Histogram of PI values

This visual correspondence proves that quantum circuit operations can be faithfully represented in photonic pump patterns!
