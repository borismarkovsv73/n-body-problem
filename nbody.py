import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import json
import time
from multiprocessing import Pool, cpu_count
from dataclasses import dataclass
from typing import List


# Constants
G = 6.6743e-11


@dataclass
class Body:
    mass: float
    position: np.ndarray
    velocity: np.ndarray
    name: str = ""


class NBodySimulator:
    
    def __init__(self, bodies: List[Body], dt: float = 86400):
        self.bodies = bodies
        self.dt = dt
        self.n_bodies = len(bodies)
        self.pool = None
        
    def calculate_forces(self) -> List[np.ndarray]:
        forces = [np.zeros(3) for _ in range(self.n_bodies)]
        
        for i in range(self.n_bodies):
            for j in range(i + 1, self.n_bodies):
                r_vec = self.bodies[j].position - self.bodies[i].position
                r = np.linalg.norm(r_vec)
                
                if r > 0:
                    force_mag = G * self.bodies[i].mass * self.bodies[j].mass / (r**2)
                    force_vec = force_mag * (r_vec / r)
                    
                    forces[i] += force_vec
                    forces[j] -= force_vec
        
        return forces
    
    def update_bodies(self, forces: List[np.ndarray]):
        for i, body in enumerate(self.bodies):
            acceleration = forces[i] / body.mass
            body.velocity += acceleration * self.dt
            body.position += body.velocity * self.dt
    
    def simulate(self, iterations: int, parallel: bool = False, processes: int = None) -> dict:
        print(f"Starting {'parallel' if parallel else 'sequential'} simulation...\n"
              f"Bodies: {self.n_bodies}, Iterations: {iterations}")
        
        if parallel and processes is None:
            processes = min(cpu_count(), self.n_bodies)
        
        results = {
            'simulation_type': 'parallel' if parallel else 'sequential',
            'n_bodies': self.n_bodies,
            'iterations': iterations,
            'dt': self.dt,
            'processes': processes if parallel else 1,
            'iteration_data': []
        }
        
        start_time = time.time()
        
        for iteration in range(iterations):
            current_state = {
                'iteration': iteration,
                'bodies': []
            }
            
            for i, body in enumerate(self.bodies):
                current_state['bodies'].append({
                    'id': i,
                    'name': body.name,
                    'mass': body.mass,
                    'position': body.position.tolist(),
                    'velocity': body.velocity.tolist(),
                    'speed': np.linalg.norm(body.velocity)
                })
            
            results['iteration_data'].append(current_state)
            
            if parallel:
                forces = self.calculate_forces_parallel(processes)
            else:
                forces = self.calculate_forces()
            
            self.update_bodies(forces)
            
            if (iteration + 1) % (iterations // 10) == 0:
                progress = ((iteration + 1) / iterations) * 100
                print(f"Progress: {progress:.1f}%")
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        results['execution_time'] = execution_time
        print(f"Simulation completed in {execution_time:.2f} seconds")
        
        return results
    
    def calculate_forces_parallel(self, processes: int) -> List[np.ndarray]:
        if self.pool is None:
            self.pool = Pool(processes)
        
        positions = np.zeros(self.n_bodies * 3)
        masses = np.zeros(self.n_bodies)
        
        for i, body in enumerate(self.bodies):
            positions[i*3:i*3+3] = body.position
            masses[i] = body.mass
        
        bodies_per_process = max(1, self.n_bodies // processes)
        chunks = []
        for i in range(0, self.n_bodies, bodies_per_process):
            end_i = min(i + bodies_per_process, self.n_bodies)
            chunks.append((i, end_i))
        
        try:
            results = self.pool.starmap(calculate_force_chunk_efficient, 
                                      [(start, end, positions, masses, G) for start, end in chunks])
            
            forces = [np.zeros(3) for _ in range(self.n_bodies)]
            for chunk_idx, (start_idx, end_idx) in enumerate(chunks):
                chunk_forces = results[chunk_idx]
                for i in range(start_idx, end_idx):
                    local_idx = i - start_idx
                    forces[i] = chunk_forces[local_idx*3:local_idx*3+3]
            
            return forces
            
        except Exception as e:
            print(f"Parallel processing failed: {e}, falling back to sequential")
            return self.calculate_forces()
    
    def cleanup(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None
    
    def __del__(self):
        self.cleanup()
    
    def visualize_results(self, results: dict, save_plots: bool = True):
        if not results['iteration_data']:
            print("No data to visualize")
            return
        
        n_bodies = results['n_bodies']
        
        trajectories = {}
        for body_data in results['iteration_data'][0]['bodies']:
            body_name = body_data['name'] or f"Body {body_data['id']}"
            trajectories[body_name] = {'x': [], 'y': [], 'z': []}
        
        for iteration_data in results['iteration_data']:
            for body_data in iteration_data['bodies']:
                body_name = body_data['name'] or f"Body {body_data['id']}"
                pos = body_data['position']
                trajectories[body_name]['x'].append(pos[0])
                trajectories[body_name]['y'].append(pos[1])
                trajectories[body_name]['z'].append(pos[2])
        
        if n_bodies <= 10:
            self._visualize_detailed(trajectories, results, save_plots)
        else:
            self._visualize_many_bodies(trajectories, results, save_plots)
    
    def _visualize_detailed(self, trajectories, results, save_plots):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
        
        for i, (name, traj) in enumerate(trajectories.items()):
            color = colors[i % len(colors)]
            
            ax1.plot(traj['x'], traj['y'], color=color, label=name, alpha=0.7, linewidth=1.5)
            ax1.scatter(traj['x'][0], traj['y'][0], color=color, s=50, marker='o')
            ax1.scatter(traj['x'][-1], traj['y'][-1], color=color, s=50, marker='s')
            
            ax2.plot(traj['x'], traj['z'], color=color, label=name, alpha=0.7, linewidth=1.5)
            ax2.scatter(traj['x'][0], traj['z'][0], color=color, s=50, marker='o')
            ax2.scatter(traj['x'][-1], traj['z'][-1], color=color, s=50, marker='s')
        
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_title('XY Trajectories')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('X Position (m)')
        ax2.set_ylabel('Z Position (m)')
        ax2.set_title('XZ Trajectories')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        iterations = [data['iteration'] for data in results['iteration_data']]
        for i, (name, _) in enumerate(trajectories.items()):
            speeds = [data['bodies'][i]['speed'] for data in results['iteration_data']]
            color = colors[i % len(colors)]
            ax3.plot(iterations, speeds, color=color, label=name, linewidth=1.5)
        
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Speed (m/s)')
        ax3.set_title('Body Speeds Over Time')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        info_text = f"Simulation Type: {results['simulation_type']}\n"
        info_text += f"Bodies: {results['n_bodies']}\n"
        info_text += f"Iterations: {results['iterations']}\n"
        info_text += f"Execution Time: {results['execution_time']:.2f}s\n"
        if results['simulation_type'] == 'parallel':
            info_text += f"Processes: {results['processes']}"
        
        ax4.text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center')
        ax4.axis('off')
        ax4.set_title('Simulation Info')
        
        plt.tight_layout()
        
        if save_plots:
            filename = f"{results['simulation_type']}_simulation_results.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Plot saved as {filename}")
        
        plt.show()
    
    def _visualize_many_bodies(self, trajectories, results, save_plots):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        n_bodies = len(trajectories)
        
        colors = plt.cm.tab20(np.linspace(0, 1, min(n_bodies, 20)))
        if n_bodies > 20:
            np.random.seed(42)
            colors = np.random.rand(n_bodies, 3)
        
        for i, (name, traj) in enumerate(trajectories.items()):
            color = colors[i % len(colors)]
            alpha = 0.6 if n_bodies <= 50 else 0.4
            linewidth = 1.0 if n_bodies <= 50 else 0.5
            
            ax1.plot(traj['x'], traj['y'], color=color, alpha=alpha, linewidth=linewidth)
            
            ax2.plot(traj['x'], traj['z'], color=color, alpha=alpha, linewidth=linewidth)
        
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_title(f'XY Trajectories ({n_bodies} bodies)')
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('X Position (m)')
        ax2.set_ylabel('Z Position (m)')
        ax2.set_title(f'XZ Trajectories ({n_bodies} bodies)')
        ax2.grid(True, alpha=0.3)
        
        iterations = [data['iteration'] for data in results['iteration_data']]
        all_speeds = []
        max_speeds = []
        min_speeds = []
        avg_speeds = []
        
        for iteration_data in results['iteration_data']:
            speeds = [body['speed'] for body in iteration_data['bodies']]
            all_speeds.append(speeds)
            max_speeds.append(max(speeds))
            min_speeds.append(min(speeds))
            avg_speeds.append(np.mean(speeds))
        
        ax3.plot(iterations, max_speeds, 'r-', label='Max Speed', linewidth=2)
        ax3.plot(iterations, avg_speeds, 'b-', label='Average Speed', linewidth=2)
        ax3.plot(iterations, min_speeds, 'g-', label='Min Speed', linewidth=2)
        ax3.fill_between(iterations, min_speeds, max_speeds, alpha=0.2, color='gray', label='Speed Range')
        
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Speed (m/s)')
        ax3.set_title('Speed Statistics Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        info_text = f"Simulation Type: {results['simulation_type']}\n"
        info_text += f"Bodies: {results['n_bodies']}\n"
        info_text += f"Iterations: {results['iterations']}\n"
        info_text += f"Execution Time: {results['execution_time']:.2f}s\n"
        if results['simulation_type'] == 'parallel':
            info_text += f"Processes: {results['processes']}\n"
        
        final_speeds = all_speeds[-1] if all_speeds else []
        if final_speeds:
            info_text += f"\nFinal Speed Stats:\n"
            info_text += f"  Max: {max(final_speeds):.2e} m/s\n"
            info_text += f"  Avg: {np.mean(final_speeds):.2e} m/s\n"
            info_text += f"  Min: {min(final_speeds):.2e} m/s"
        
        ax4.text(0.05, 0.95, info_text, fontsize=10, verticalalignment='top', 
                transform=ax4.transAxes, family='monospace')
        ax4.axis('off')
        ax4.set_title('Simulation Statistics')
        
        plt.tight_layout()
        
        if save_plots:
            filename = f"{results['simulation_type']}_simulation_results.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Plot saved as {filename}")
        
        plt.show()
    
    def create_animated_visualization(self, results: dict, save_animation: bool = False):
        if not results['iteration_data']:
            print("No data to animate")
            return
        
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 10))
        fig.patch.set_facecolor('#0a0a0a')
        ax.set_facecolor('#0a0a0a')
        
        n_bodies = results['n_bodies']
        n_frames = len(results['iteration_data'])
        
        trajectories = {}
        for body_data in results['iteration_data'][0]['bodies']:
            body_name = body_data['name'] or f"Body{body_data['id']}"
            trajectories[body_name] = {
                'x': [], 'y': [], 'masses': []
            }
        
        for iteration_data in results['iteration_data']:
            for body_data in iteration_data['bodies']:
                body_name = body_data['name'] or f"Body{body_data['id']}"
                pos = body_data['position']
                trajectories[body_name]['x'].append(pos[0])
                trajectories[body_name]['y'].append(pos[1])
                trajectories[body_name]['masses'].append(body_data['mass'])
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
                  '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']
        
        all_x = []
        all_y = []
        for traj in trajectories.values():
            all_x.extend(traj['x'])
            all_y.extend(traj['y'])
        
        margin = 0.1
        x_range = max(all_x) - min(all_x) if len(set(all_x)) > 1 else abs(max(all_x)) * 0.1
        y_range = max(all_y) - min(all_y) if len(set(all_y)) > 1 else abs(max(all_y)) * 0.1
        
        if x_range == 0:
            x_range = max(abs(min(all_x)), abs(max(all_x))) * 0.2 or 1e10
        if y_range == 0:
            y_range = max(abs(min(all_y)), abs(max(all_y))) * 0.2 or 1e10
            
        ax.set_xlim(min(all_x) - margin * x_range, max(all_x) + margin * x_range)
        ax.set_ylim(min(all_y) - margin * y_range, max(all_y) + margin * y_range)
        
        bodies = []
        trails = []
        body_names = list(trajectories.keys())
        
        max_mass = max(max(traj['masses']) for traj in trajectories.values())
        min_radius = max(x_range, y_range) * 0.005
        max_radius = max(x_range, y_range) * 0.02
        
        for i, body_name in enumerate(body_names):
            color = colors[i % len(colors)]
            
            # Body size based on mass
            mass = trajectories[body_name]['masses'][0]
            relative_mass = mass / max_mass if max_mass > 0 else 0.5
            radius = min_radius + (max_radius - min_radius) * relative_mass
            
            # Body circle with glow effect
            body_circle = Circle((0, 0), radius=radius, color=color, alpha=0.9, 
                               edgecolor='white', linewidth=1)
            ax.add_patch(body_circle)
            bodies.append(body_circle)
            
            # Trail line
            trail, = ax.plot([], [], color=color, alpha=0.6, linewidth=1.5)
            trails.append(trail)
        
        # Styling like enhanced_visualizer
        ax.set_xlabel('X Position (m)', color='white', fontweight='bold', fontsize=12)
        ax.set_ylabel('Y Position (m)', color='white', fontweight='bold', fontsize=12)
        ax.set_title(f'Animated N-Body Simulation - {results["simulation_type"].title()}', 
                    color='white', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.2, color='white')
        ax.tick_params(colors='white')
        
        info_text = f"Bodies: {n_bodies}\nMode: {results['simulation_type']}\n● Current positions"
        info_box = ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                          bbox=dict(boxstyle='round,pad=0.5', facecolor='black', 
                                   alpha=0.8, edgecolor='white'),
                          color='white', fontsize=10, verticalalignment='top')
        
        def animate(frame):
            for i, body_name in enumerate(body_names):
                traj = trajectories[body_name]
                
                x = traj['x'][frame]
                y = traj['y'][frame]
                bodies[i].center = (x, y)
                
                trail_length = min(frame + 1, 50)
                start_idx = max(0, frame - trail_length + 1)
                trail_x = traj['x'][start_idx:frame+1]
                trail_y = traj['y'][start_idx:frame+1]
                trails[i].set_data(trail_x, trail_y)
            
            current_iter = results['iteration_data'][frame]['iteration']
            new_info = f"Bodies: {n_bodies}\nMode: {results['simulation_type']}\nIteration: {current_iter}"
            info_box.set_text(new_info)
            
            return bodies + trails + [info_box]
        
        anim = animation.FuncAnimation(fig, animate, frames=n_frames, 
                                     interval=100, blit=False, repeat=True)
        
        if save_animation:
            filename = f"{results['simulation_type']}_animation.gif"
            anim.save(filename, writer='pillow', fps=10, dpi=100)
            print(f"Animation saved as {filename}")
        
        plt.show()
        return anim
    
    def create_3d_animated_visualization(self, results: dict, save_animation: bool = False):
        if not results['iteration_data']:
            print("No data to animate")
            return
        
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        fig.patch.set_facecolor('#0a0a0a')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        ax.xaxis.pane.set_alpha(0.1)
        ax.yaxis.pane.set_alpha(0.1)
        ax.zaxis.pane.set_alpha(0.1)
        
        n_bodies = results['n_bodies']
        n_frames = len(results['iteration_data'])
        
        print(f"Setting up 3D visualization for {n_bodies} bodies, {n_frames} frames...")
        
        trajectories = {}
        for body_data in results['iteration_data'][0]['bodies']:
            body_name = body_data['name'] or f"Body{body_data['id']}"
            trajectories[body_name] = {
                'x': [], 'y': [], 'z': [], 'masses': []
            }
        
        for iteration_data in results['iteration_data']:
            for body_data in iteration_data['bodies']:
                body_name = body_data['name'] or f"Body{body_data['id']}"
                pos = body_data['position']
                trajectories[body_name]['x'].append(pos[0])
                trajectories[body_name]['y'].append(pos[1])
                trajectories[body_name]['z'].append(pos[2])
                trajectories[body_name]['masses'].append(body_data['mass'])
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
                  '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']
        
        all_x, all_y, all_z = [], [], []
        for traj in trajectories.values():
            all_x.extend(traj['x'])
            all_y.extend(traj['y'])
            all_z.extend(traj['z'])
        
        margin = 0.15
        
        if len(set(all_x)) <= 1:
            x_center = all_x[0] if all_x else 0
            x_range = abs(x_center) * 0.1 if x_center != 0 else 1e10
            x_min, x_max = x_center - x_range, x_center + x_range
        else:
            x_min, x_max = min(all_x), max(all_x)
            x_range = x_max - x_min
            x_min -= margin * x_range
            x_max += margin * x_range
            
        if len(set(all_y)) <= 1:
            y_center = all_y[0] if all_y else 0
            y_range = abs(y_center) * 0.1 if y_center != 0 else 1e10
            y_min, y_max = y_center - y_range, y_center + y_range
        else:
            y_min, y_max = min(all_y), max(all_y)
            y_range = y_max - y_min
            y_min -= margin * y_range
            y_max += margin * y_range
            
        if len(set(all_z)) <= 1:
            z_center = all_z[0] if all_z else 0
            z_range = abs(z_center) * 0.1 if z_center != 0 else 1e10
            z_min, z_max = z_center - z_range, z_center + z_range
        else:
            z_min, z_max = min(all_z), max(all_z)
            z_range = z_max - z_min
            z_min -= margin * z_range
            z_max += margin * z_range
                    
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        
        try:
            ax.set_box_aspect([1,1,1])
        except:
            pass
        
        trails = []
        body_names = list(trajectories.keys())
        
        max_mass = max(max(traj['masses']) for traj in trajectories.values())
        
        current_x = []
        current_y = []
        current_z = []
        body_colors = []
        body_sizes = []
        
        for i, body_name in enumerate(body_names):
            color = colors[i % len(colors)]
            
            mass = trajectories[body_name]['masses'][0]
            relative_mass = mass / max_mass if max_mass > 0 else 0.5
            size = 50 + relative_mass * 200
            
            current_x.append(trajectories[body_name]['x'][0])
            current_y.append(trajectories[body_name]['y'][0])
            current_z.append(trajectories[body_name]['z'][0])
            body_colors.append(color)
            body_sizes.append(size)
            
            trail, = ax.plot([], [], [], color=color, alpha=0.6, linewidth=1.5)
            trails.append(trail)
        
        bodies_scatter = ax.scatter(current_x, current_y, current_z, 
                                  s=body_sizes, c=body_colors, 
                                  alpha=0.9, edgecolors='white', linewidth=1)
        
        ax.set_xlabel('X Position (m)', color='white', fontweight='bold', fontsize=12)
        ax.set_ylabel('Y Position (m)', color='white', fontweight='bold', fontsize=12)
        ax.set_zlabel('Z Position (m)', color='white', fontweight='bold', fontsize=12)
        ax.set_title(f'3D Animated N-Body Simulation - {results["simulation_type"].title()}', 
                    color='white', fontsize=14, fontweight='bold')
        ax.tick_params(colors='white')
        
        info_text = f"Bodies: {n_bodies}\nMode: {results['simulation_type']}\n● 3D Visualization"
        fig.text(0.02, 0.98, info_text, bbox=dict(boxstyle='round,pad=0.5', 
                facecolor='black', alpha=0.8, edgecolor='white'),
                color='white', fontsize=10, verticalalignment='top')
        
        def animate_3d(frame):

            current_x = []
            current_y = []
            current_z = []
            
            for i, body_name in enumerate(body_names):
                traj = trajectories[body_name]
                
                x = traj['x'][frame]
                y = traj['y'][frame]
                z = traj['z'][frame]
                
                current_x.append(x)
                current_y.append(y)
                current_z.append(z)
                
                trail_length = min(frame + 1, 30)
                start_idx = max(0, frame - trail_length + 1)
                trail_x = traj['x'][start_idx:frame+1]
                trail_y = traj['y'][start_idx:frame+1]
                trail_z = traj['z'][start_idx:frame+1]
                trails[i].set_data_3d(trail_x, trail_y, trail_z)
            
            bodies_scatter._offsets3d = (current_x, current_y, current_z)
            
            return [bodies_scatter] + trails
        
        anim = animation.FuncAnimation(fig, animate_3d, frames=n_frames, 
                                     interval=100, blit=False, repeat=True)
        
        if save_animation:
            filename = f"{results['simulation_type']}_3d_animation.gif"
            anim.save(filename, writer='pillow', fps=10, dpi=80)
            print(f"3D Animation saved as {filename}")
        
        plt.show()
        return anim


def calculate_force_chunk_efficient(start_idx, end_idx, positions, masses, G):
    n_bodies = len(masses)
    forces = np.zeros((end_idx - start_idx) * 3)
    
    for i in range(start_idx, end_idx):
        total_force = np.zeros(3)
        i_pos = np.array([positions[i*3], positions[i*3+1], positions[i*3+2]])
        
        for j in range(n_bodies):
            if i != j:
                j_pos = np.array([positions[j*3], positions[j*3+1], positions[j*3+2]])
                
                r_vec = j_pos - i_pos
                r_magnitude = np.linalg.norm(r_vec)
                
                if r_magnitude > 1e-10:
                    force_magnitude = G * masses[i] * masses[j] / (r_magnitude ** 2)

                    force_direction = r_vec / r_magnitude
                    
                    total_force += force_magnitude * force_direction
        
        local_idx = i - start_idx
        forces[local_idx*3:local_idx*3+3] = total_force
    
    return forces


def create_many_body_system(n_bodies: int = 100) -> List[Body]:
    np.random.seed(42)
    bodies = []
    
    for i in range(n_bodies):
        mass = np.random.uniform(1e23, 1e30)
        
        r = np.random.uniform(1e10, 1e12)
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        position = np.array([x, y, z])
        
        v_mag = np.random.uniform(1e3, 1e4)
        v_theta = np.random.uniform(0, 2 * np.pi)
        v_phi = np.random.uniform(0, np.pi)
        
        vx = v_mag * np.sin(v_phi) * np.cos(v_theta)
        vy = v_mag * np.sin(v_phi) * np.sin(v_theta)
        vz = v_mag * np.cos(v_phi)
        velocity = np.array([vx, vy, vz])
        
        bodies.append(Body(mass=mass, position=position, velocity=velocity, name=f"Body{i+1}"))
    
    return bodies


def performance_comparison(bodies: List[Body], iterations: int = 100):
    print("=" * 60 + "\nPERFORMANCE COMPARISON\n" + "=" * 60)
    
    simulator_seq = NBodySimulator([Body(b.mass, b.position.copy(), b.velocity.copy(), b.name) 
                                   for b in bodies])
    results_seq = simulator_seq.simulate(iterations, parallel=False)
    simulator_seq.cleanup()
    
    simulator_par = NBodySimulator([Body(b.mass, b.position.copy(), b.velocity.copy(), b.name) 
                                   for b in bodies])
    results_par = simulator_par.simulate(iterations, parallel=True)
    simulator_par.cleanup()
    
    seq_time = results_seq['execution_time']
    par_time = results_par['execution_time']
    speedup = seq_time / par_time if par_time > 0 else 1.0
    
    print(f"\nResults:\n"
          f"Sequential time: {seq_time:.3f} seconds\n"
          f"Parallel time: {par_time:.3f} seconds\n"
          f"Speedup: {speedup:.2f}x\n"
          f"Efficiency: {(speedup / results_par['processes']) * 100:.1f}%")
    
    return results_seq, results_par


def cleanup_generated_files():
    import os
    import glob
    
    file_patterns = ['*.json', '*.png', '*.gif']
    deleted_files = []
    
    for pattern in file_patterns:
        for file_path in glob.glob(pattern):
            try:
                os.remove(file_path)
                deleted_files.append(file_path)
            except OSError as e:
                print(f"Could not delete {file_path}: {e}")
    
    if deleted_files:
        print(f"Deleted {len(deleted_files)} files:\n" + 
              "\n".join(f"  - {f}" for f in deleted_files))
    else:
        print("No generated files found to delete.")


def main():
    print("=" * 60 + "\nN-BODY PROBLEM\n" + "=" * 60)
    
    while True:
        print("\nOptions:\n"
              "1. Many Body System (custom number of bodies)\n"
              "2. Performance Comparison (many-body)\n"
              "3. Clean up generated files\n"
              "4. Exit")

        try:
            choice = input("\nEnter choice (1-4): ").strip()

            if choice == '1':
                n_bodies = int(input("Enter number of bodies (default 100): ") or 100)
                bodies = create_many_body_system(n_bodies)
                iterations = int(input("Enter number of iterations (default 50): ") or 50)

                print("\nSimulation mode:\n1. Sequential\n2. Parallel")
                mode = input("Choose mode (1-2): ").strip()
                parallel = mode == '2'

                simulator = NBodySimulator(bodies)
                results = simulator.simulate(iterations, parallel=parallel)

                filename = f"{ 'parallel' if parallel else 'sequential' }_many_body_{n_bodies}.json"
                with open(filename, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Results saved to {filename}")

                print("\nVisualization options:\n"
                      "1. Static plots\n"
                      "2. 2D Animated visualization\n"
                      "3. 3D Animated visualization\n"
                      "4. Both static and 2D animation\n"
                      "5. Both static and 3D animation\n"
                      "6. All visualizations")
                viz_choice = input("Choose visualization (1-6, default 1): ").strip() or "1"

                if viz_choice in ["1", "4", "5", "6"]:
                    simulator.visualize_results(results)
                if viz_choice in ["2", "4", "6"]:
                    save_anim = input("Save 2D animation as GIF? (y/n, default n): ").strip().lower() == 'y'
                    simulator.create_animated_visualization(results, save_animation=save_anim)
                if viz_choice in ["3", "5", "6"]:
                    save_anim_3d = input("Save 3D animation as GIF? (y/n, default n): ").strip().lower() == 'y'
                    simulator.create_3d_animated_visualization(results, save_animation=save_anim_3d)

                simulator.cleanup()

            elif choice == '2':
                n_bodies = int(input("Enter number of bodies for comparison (default 100): ") or 100)
                bodies = create_many_body_system(n_bodies)

                iterations = int(input("Enter number of iterations for comparison (default 50): ") or 50)
                results_seq, results_par = performance_comparison(bodies, iterations)

                comparison = {
                    'sequential': results_seq,
                    'parallel': results_par,
                    'speedup': results_seq['execution_time'] / results_par['execution_time'] if results_par['execution_time'] > 0 else None
                }
                with open("performance_comparison.json", 'w') as f:
                    json.dump(comparison, f, indent=2)
                print("Comparison results saved to performance_comparison.json")

            elif choice == '3':
                cleanup_generated_files()

            elif choice == '4':
                break

            else:
                print("Invalid choice. Please enter 1-4.")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.")


if __name__ == "__main__":
    main()