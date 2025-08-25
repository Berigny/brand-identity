import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
import random
import os
from collections import defaultdict

# --- 1. IMAGE GENERATION SYSTEM ---

def generate_noisy_shapes(noise_level: int = 40) -> np.ndarray:
    """Generate a small test image with simple shapes plus configurable noise."""
    img = np.zeros((256, 256, 3), dtype=np.uint8)

    # Circle
    cv2.circle(img, (80, 128), 40, (255, 255, 255), -1)

    # Triangle
    pts = np.array([[160, 80], [200, 180], [120, 180]], np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(img, [pts], (255, 255, 255))

    # Add Gaussian noise
    noise = np.random.normal(0, noise_level, img.shape).astype(np.int16)
    noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy


# --- 2. VISION PROCESSING MODULE ---

class VisionAgent:
    def __init__(self, denoise: bool = True):
        self.denoise = denoise
        self.shape_db = {
            'triangle': {'sides': 3, 'area_range': (500, 50000)},
            'circle': {'sides': 8, 'area_range': (1000, 100000)},
        }

    def detect_shapes(self, image: np.ndarray) -> list[str]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.denoise:
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shapes: list[str] = []
        for cnt in contours:
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
            area = cv2.contourArea(cnt)

            for shape, params in self.shape_db.items():
                sides_ok = len(approx) >= params['sides']
                area_ok = params['area_range'][0] < area < params['area_range'][1]
                if sides_ok and area_ok:
                    shapes.append(shape)
                    break

        return shapes if shapes else ["unknown"]


# --- 3. NETWORK ARCHITECTURE ---

class CognitiveNetwork:
    def __init__(self):
        self.nodes = {
            0: "S1_Compression",
            1: "S1_Expression",
            2: "S1_Stabilisation",
            3: "S1_Emission",
            4: "S2_Compression",
            5: "S2_Expression",
            6: "S2_Stabilisation",
            7: "S2_Emission",
            'IC': "InternalC",
            'EC': "ExternalC",
        }

        self.primes = {0: 2, 1: 3, 2: 5, 3: 7, 4: 11, 5: 13, 6: 17, 7: 19, 'IC': 23, 'EC': 29, '∞': 31}
        # Node goals and thresholds
        self.node_goals: dict = {
            0: {"goal": "compress sensory input", "threshold": 0.7},
            1: {"goal": "generate expression hypotheses", "threshold": 0.6},
            2: {"goal": "stabilise context", "threshold": 0.8},
            3: {"goal": "emit quick output", "threshold": 0.5},
            4: {"goal": "deep compression & abstraction", "threshold": 0.9},
            5: {"goal": "refine expressive hypotheses", "threshold": 0.8},
            6: {"goal": "integrate long-term context", "threshold": 0.95},
            7: {"goal": "deliberate emission", "threshold": 0.9},
            'IC': {"goal": "integrate coherence", "threshold": 0.95},
            'EC': {"goal": "interact with environment", "threshold": 1.0},
        }
        self.build_network()

    def build_network(self):
        self.G = nx.DiGraph()
        self.G.add_nodes_from(self.nodes.keys())
        edges = []

        # --- Tetrahedron 1 (0,1,2,3) ---
        tetra1 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        edges += [(a, b, {'type': 'tetrahedral', 'weight': 0.7}) for a, b in tetra1]
        edges += [
            (0, 'IC', {'type': 'cubic', 'weight': 1.0}),
            ('IC', 1, {'type': 'cubic', 'weight': 0.9}),
            (2, 'IC', {'type': 'cubic', 'weight': 0.8}),
            ('IC', 3, {'type': 'cubic', 'weight': 1.0}),
            (3, 0, {'type': 'cubic', 'weight': 0.6}),  # Loopback
        ]

        # --- Tetrahedron 2 (4,5,6,7) ---
        tetra2 = [(4, 5), (4, 6), (4, 7), (5, 6), (5, 7), (6, 7)]
        edges += [(a, b, {'type': 'tetrahedral', 'weight': 0.7}) for a, b in tetra2]
        edges += [
            (4, 'IC', {'type': 'cubic', 'weight': 1.0}),
            ('IC', 5, {'type': 'cubic', 'weight': 0.9}),
            (6, 'IC', {'type': 'cubic', 'weight': 0.8}),
            ('IC', 7, {'type': 'cubic', 'weight': 1.0}),
            (7, 4, {'type': 'cubic', 'weight': 0.6}),  # Loopback
        ]

        # --- Cross-tetrahedron cubic edges (12 edges) ---
        cube_cross = [
            (0, 4), (1, 5), (2, 6), (3, 7),
            (0, 5), (0, 6), (1, 4), (1, 6),
            (2, 4), (2, 5), (3, 4), (3, 5),
        ]
        edges += [(a, b, {'type': 'cubic', 'weight': 0.8}) for a, b in cube_cross]

        # --- External connections ---
        edges += [
            ('EC', 'IC', {'type': 'cubic', 'weight': 1.0}),
            ('EC', 0, {'type': 'cubic', 'weight': 0.8}),
            ('EC', 4, {'type': 'cubic', 'weight': 0.8}),
            (0, 'EC', {'type': 'failsafe', 'weight': 0.5}),
            (2, 'EC', {'type': 'failsafe', 'weight': 0.5}),
            (4, 'EC', {'type': 'failsafe', 'weight': 0.5}),
            (6, 'EC', {'type': 'failsafe', 'weight': 0.5}),
        ]

        self.G.add_edges_from(edges)

    def visualize(self, highlight_nodes=None):
        pos = nx.spring_layout(self.G, seed=42)
        edge_colors = [
            'green' if data['type'] == 'tetrahedral' else 'red' if data['type'] == 'failsafe' else 'gray'
            for _, _, data in self.G.edges(data=True)
        ]

        plt.figure(figsize=(14, 10))
        nx.draw(
            self.G,
            pos,
            with_labels=True,
            labels=self.nodes,
            node_color='lightblue',
            edge_color=edge_colors,
            node_size=2000,
            arrows=True,
            width=2,
        )

        if highlight_nodes:
            nx.draw_networkx_nodes(
                self.G, pos, nodelist=highlight_nodes, node_color='orange', node_size=2500
            )

        edge_labels = {
            (u, v): f"{data['type']} ({data['weight']:.1f})" for u, v, data in self.G.edges(data=True)
        }
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels, font_size=8)
        plt.title("Cognitive Network Architecture")
        plt.tight_layout()


# --- 4. ROUTING CONTROLLER ---

class RoutingController:
    def __init__(self, network: CognitiveNetwork):
        self.network = network
        self.visit_count = defaultdict(int, {n: 0 for n in self.network.G.nodes})
        self.transition_log: list[dict] = []
        self.ic_status = "up"
        self.max_local_steps = 5  # max processing steps a packet spends at a node before escalation
        # Alignment/consilience dynamics
        # Allow tuning the boundary for "higher primes" via env or by mapping kernel policy
        try:
            self.high_prime_min = int(os.getenv("FLOW_HIGH_PRIME_MIN")) if os.getenv("FLOW_HIGH_PRIME_MIN") else None
        except Exception:
            self.high_prime_min = None
        if self.high_prime_min is None:
            try:
                from app.utils.prime_mapping import get_high_prime_min_from_kernel
                self.high_prime_min = int(get_high_prime_min_from_kernel())
            except Exception:
                self.high_prime_min = 11
        # primes >= this considered higher-level
        self.alignment_streak = 0
        self.streak_target = 3
        self.s1_threshold_offset = 0.0  # adaptive relaxation for S1 thresholds (max 0.05)
        self.s1_threshold_offset_cap = 0.05

    def select_agent(self, node):
        if node in [0, 1, 2, 3]:
            return "AgentA"
        if node in [4, 5, 6, 7]:
            return "AgentB"
        if node == 'IC':
            return "AgentC1"
        if node == 'EC':
            return "AgentC2"
        return "Unknown"

    def calculate_coherence(self, prev, new):
        edge_data = self.network.G.get_edge_data(prev, new)
        if not edge_data:
            return 0, 0, 3, 1, 0

        weight = edge_data.get('weight', 1.0)
        strain = (1 * weight) if (prev in [1, 5] and new == 'IC') else 0
        resonance = 1.2 if edge_data['type'] == 'tetrahedral' else 0.8
        try:
            d_p = (
                1 * weight
                if prev in ['IC', 'EC'] or new in ['IC', 'EC']
                else nx.shortest_path_length(self.network.G, prev, new)
            )
        except nx.NetworkXNoPath:
            d_p = 3
        p = self.network.primes.get(new, 2)
        # Consilience: favor higher primes (higher-level integration) and shorter distances
        import math
        prime_factor = max(1.0, math.log(p))  # grows with higher primes
        base = resonance * prime_factor / (1.0 + max(d_p - 1, 0))
        penalty = 0.2 if strain > 0 else 0.0
        contrib = max(0.0, base - penalty)
        return strain, resonance, d_p, p, contrib

    def decide_route(self, packet: dict, successors: list):
        current = packet['location']
        _ = packet['data'].lower()
        _ = self.select_agent(current)

        successors = [s for s in successors if s != current]

        # Branch node routing: prefer least-visited sink
        if current in [1, 3, 5, 7]:
            sink_targets = [s for s in successors if s in [0, 2, 4, 6]]
            if sink_targets:
                return min(sink_targets, key=lambda x: self.visit_count[x])

        # Failsafe if IC is down
        if self.ic_status == "down" and 'EC' in successors:
            return 'EC'

        # Default: least visited next
        return min(successors, key=lambda x: self.visit_count[x])

    def node_threshold(self, node):
        goal = self.network.node_goals.get(node, {})
        thr = float(goal.get("threshold", 0.8))
        if node in [0, 1, 2, 3]:  # S1 adaptive relaxation via repeated high-prime alignment
            thr = max(0.4, thr - self.s1_threshold_offset)
        return thr

    def confidence_update(self, prev, node):
        # Derive an incremental confidence gain based on edge qualities and node importance
        strain, resonance, d_p, p, contrib = self.calculate_coherence(prev, node)
        base = 0.05 if node in [0, 1, 2, 3] else 0.04  # S1 slightly faster baseline
        # Confidence gain derives from consilience contrib, not rewarding low primes
        gain = base + 0.04 * contrib
        if strain > 0:
            gain *= 0.85
        # If repeated higher-prime alignment occurred, enable more plasticity in S1
        if node in [0, 1, 2, 3] and self.s1_threshold_offset > 0:
            gain *= 1.05
        # Cap to avoid spikes
        return max(0.005, min(gain, 0.12))

    def update_alignment(self, target_prime, contrib):
        # Count alignment only when moving into higher-prime territory with strong consilience
        if target_prime >= self.high_prime_min and contrib >= 0.8:
            self.alignment_streak += 1
            if self.alignment_streak >= self.streak_target:
                # Relax S1 thresholds slightly (adaptation of lower layers)
                self.s1_threshold_offset = min(
                    self.s1_threshold_offset_cap, self.s1_threshold_offset + 0.01
                )
                self.alignment_streak = 0
        else:
            # gentle decay
            self.alignment_streak = max(0, self.alignment_streak - 1)


# --- 5. SIMULATION ---

class Simulation:
    def __init__(self):
        self.vision = VisionAgent()
        self.network = CognitiveNetwork()
        self.controller = RoutingController(self.network)
        self.packets: list[dict] = []
        self.completed: bool = False
        self.path: list = []

    def initialize(self, image: np.ndarray):
        shapes = self.vision.detect_shapes(image)
        description = f"Detected shapes: {', '.join(shapes)}"
        init_conf = 0.4 if any(s != "unknown" for s in shapes) else 0.2
        self.packets = [
            {
                'id': 0,
                'data': description,
                'location': 'EC',
                'prev': None,
                'priority': 1.0 if shapes else 0.5,
                'confidence': init_conf,
                'steps_at_node': 0,
                'total_steps': 0,
                'status': 'processing',
            }
        ]

    def run_step(self, verbose: bool = False):
        new_packets: list[dict] = []
        for p in self.packets:
            prev = p['location']
            node_thresh = self.controller.node_threshold(prev)

            # 1) Try to process at current node until threshold or local budget exhausted
            if p['confidence'] < node_thresh and p['steps_at_node'] < self.controller.max_local_steps:
                gain = self.controller.confidence_update(p['prev'] if p['prev'] is not None else prev, prev)
                p['confidence'] = min(1.0, p['confidence'] + gain)
                p['steps_at_node'] += 1
                p['total_steps'] += 1
                if verbose:
                    print(f"Process at {prev}: +{gain:.3f} -> conf {p['confidence']:.3f} ({p['steps_at_node']}/{self.controller.max_local_steps})")
                new_packets.append(p)
                continue

            # 2) If EC and sufficiently confident, finish
            if prev == 'EC' and p['confidence'] >= 0.99:
                p['status'] = 'completed'
                self.completed = True
                self.path.append(('HALT', 'EC'))
                new_packets.append(p)
                continue

            successors = list(self.network.G.successors(prev))
            if not successors:
                new_packets.append(p)
                continue

            # 3) Decide where to go next: if not meeting threshold after budget, escalate
            if p['confidence'] < node_thresh and p['steps_at_node'] >= self.controller.max_local_steps:
                if prev in [0, 1, 2, 3]:
                    # Prefer IC if available, else any S2 successor
                    new_loc = 'IC' if 'IC' in successors else (
                        min([s for s in successors if s in [4, 5, 6, 7]], default=self.controller.decide_route(p, successors), key=lambda x: self.controller.visit_count[x])
                    )
                elif prev in [4, 5, 6, 7]:
                    new_loc = 'IC' if 'IC' in successors else self.controller.decide_route(p, successors)
                else:
                    # If stuck at IC without meeting threshold, try EC as last resort
                    new_loc = 'EC' if 'EC' in successors else self.controller.decide_route(p, successors)
            else:
                # Threshold met -> proceed along normal routing
                new_loc = self.controller.decide_route(p, successors)

            strain, resonance, dist, prime, contrib = self.controller.calculate_coherence(prev, new_loc)
            # Update alignment/plasticity based on higher-prime consilience
            self.controller.update_alignment(prime, contrib)
            self.controller.transition_log.append(
                {
                    'agent': self.controller.select_agent(prev),
                    'from': prev,
                    'to': new_loc,
                    'strain': strain,
                    'resonance': resonance,
                    'distance': dist,
                    'prime': prime,
                    'contrib': contrib,
                    'confidence': p['confidence'],
                    'threshold': node_thresh,
                }
            )

            # Update packet state and controller stats
            p = {**p, 'prev': prev, 'location': new_loc, 'steps_at_node': 0, 'total_steps': p['total_steps'] + 1}
            self.controller.visit_count[new_loc] += 1
            self.path.append((prev, new_loc))
            if verbose:
                print(f"Moved packet {p['id']}: {prev} -> {new_loc} (conf {p['confidence']:.3f})")
            new_packets.append(p)

        self.packets = new_packets

    def calculate_coherence(self) -> float:
        if not self.controller.transition_log:
            return 1.0
        total = sum(t['contrib'] for t in self.controller.transition_log)
        return (total / len(self.controller.transition_log)) + 1.0

    def visualize(self):
        self.network.visualize([p['location'] for p in self.packets])
        plt.figure(figsize=(10, 5))
        visits = sorted([(str(k), v) for k, v in self.controller.visit_count.items()])
        plt.bar([k for k, _ in visits], [v for _, v in visits])
        plt.title("Node Visit Distribution")
        plt.xlabel("Node")
        plt.ylabel("Visit Count")
        plt.tight_layout()
        plt.show()


# --- 6b. PUBLIC WRAPPER FOR API ---

def run_flow_simulation(max_steps: int = 150, verbose: bool = False):
    """Run the confidence-driven flow simulation and return a summary dict."""
    test_image = generate_noisy_shapes(noise_level=40)
    sim = Simulation()
    sim.initialize(test_image)

    step_count = 0
    while not sim.completed and step_count < max_steps:
        sim.run_step(verbose=verbose)
        step_count += 1

    # Prepare summary
    packet = sim.packets[0] if sim.packets else {}
    return {
        'completed': sim.completed,
        'final_confidence': float(packet.get('confidence', 0.0)) if packet else 0.0,
        'total_steps': int(packet.get('total_steps', step_count)) if packet else step_count,
        'location': packet.get('location', None) if packet else None,
        'status': packet.get('status', None) if packet else None,
        'path': sim.path,
        'coherence_score': sim.calculate_coherence(),
        'transitions': sim.controller.transition_log,
    }


# --- 6. MAIN EXECUTION ---

if __name__ == "__main__":
    test_image = generate_noisy_shapes(noise_level=40)
    plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    plt.title("Input Image")
    plt.axis('off')
    plt.show()

    sim = Simulation()
    sim.initialize(test_image)

    max_steps = 100
    step_count = 0
    while min(sim.controller.visit_count.values() or [0]) < 2 and step_count < max_steps:
        sim.run_step(verbose=True)
        step_count += 1

    print(f"\nInitial Detection: {sim.packets[0]['data']}")
    print(f"Final Coherence: {sim.calculate_coherence():.3f}")
    sim.visualize()

    df = pd.DataFrame(sim.controller.transition_log)
    if not df.empty:
        df['from'] = df['from'].astype('category')
        df['to'] = df['to'].astype('category')
        print(df.head())

    # End of main
