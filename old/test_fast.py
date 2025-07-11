import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import threading
import time
import queue

class UltraOptimizedScatterPlot:
    """Ultra-optimized scatter plot for knee joint data: flexion, varus-valgus, torque."""
    
    def __init__(self, width=1000, height=600, max_points=100000):
        self.width, self.height = width, height
        self.max_points = max_points
        
        # Pre-allocate all arrays (zero memory allocation during runtime)
        self.flexion = np.zeros(max_points, dtype=np.float32)     # (-10, 120)
        self.varus_valgus = np.zeros(max_points, dtype=np.float32) # (-20, 20) 
        self.torque = np.zeros(max_points, dtype=np.float32)       # (-10, 10)
        
        self.write_idx = 0
        self.point_count = 0
        
        # View bounds
        self.flex_min, self.flex_max = -10.0, 120.0
        self.vv_min, self.vv_max = -20.0, 20.0
        
        # Pre-compute coordinate transform constants
        self.x_scale = width / (self.flex_max - self.flex_min)
        self.y_scale = height / (self.vv_max - self.vv_min)
        
        # Pre-allocate image arrays
        self.img_array = np.zeros((height, width, 3), dtype=np.uint8)
        self.temp_coords = np.zeros((max_points, 2), dtype=np.int32)
        
        # Pre-compute torque color lookup table (256 colors for speed)
        self.color_lut = self._create_color_lut()
        
        # Thread-safe data queue
        self.data_queue = queue.Queue(maxsize=1000)
        self.running = True
        
        self._setup_gui()
        self._start_data_thread()
        self._render_loop()
        
    def _create_color_lut(self):
        """Pre-compute color lookup table for torque values."""
        colors = np.zeros((256, 3), dtype=np.uint8)
        for i in range(256):
            # Map 0-255 to torque range (-10, 10)
            torque_norm = i / 255.0  # 0 to 1
            
            if torque_norm < 0.5:  # Negative torque: blue to green
                colors[i] = [0, int(255 * torque_norm * 2), int(255 * (1 - torque_norm * 2))]
            else:  # Positive torque: green to red
                colors[i] = [int(255 * (torque_norm - 0.5) * 2), 255, 0]
                
        return colors
        
    def _setup_gui(self):
        """Minimal GUI setup."""
        self.root = tk.Tk()
        self.root.title("Knee Joint Analysis - 100k Points")
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height, bg='black')
        self.canvas.pack()
        
        # Status
        self.status = tk.Label(self.root, text="Points: 0 | FPS: --")
        self.status.pack()
        
        self.frame_count = 0
        self.last_time = time.time()
        
    def add_point(self, flexion, varus_valgus, torque):
        """Add point to circular buffer."""
        self.flexion[self.write_idx] = flexion
        self.varus_valgus[self.write_idx] = varus_valgus
        self.torque[self.write_idx] = torque
        
        self.write_idx = (self.write_idx + 1) % self.max_points
        self.point_count = min(self.point_count + 1, self.max_points)
        
    def _render_frame(self):
        """Ultra-optimized rendering using vectorized operations."""
        # Clear image (fastest method)
        self.img_array.fill(0)
        
        if self.point_count == 0:
            return
            
        # Get active data slice
        if self.point_count < self.max_points:
            flex = self.flexion[:self.point_count]
            vv = self.varus_valgus[:self.point_count]
            torq = self.torque[:self.point_count]
        else:
            # Handle circular buffer efficiently
            flex = np.concatenate([self.flexion[self.write_idx:], self.flexion[:self.write_idx]])
            vv = np.concatenate([self.varus_valgus[self.write_idx:], self.varus_valgus[:self.write_idx]])
            torq = np.concatenate([self.torque[self.write_idx:], self.torque[:self.write_idx]])
        
        # Vectorized coordinate transformation
        screen_x = ((flex - self.flex_min) * self.x_scale).astype(np.int32)
        screen_y = (self.height - (vv - self.vv_min) * self.y_scale).astype(np.int32)
        
        # Filter valid coordinates
        valid = (screen_x >= 1) & (screen_x < self.width-1) & (screen_y >= 1) & (screen_y < self.height-1)
        
        if not np.any(valid):
            return
            
        x_valid = screen_x[valid]
        y_valid = screen_y[valid]
        torq_valid = torq[valid]
        
        # Map torque to color indices
        color_indices = np.clip(((torq_valid + 10) / 20 * 255), 0, 255).astype(np.uint8)
        colors = self.color_lut[color_indices]
        
        # Ultra-fast pixel setting using advanced indexing
        # Draw 2x2 pixels for visibility
        self.img_array[y_valid, x_valid] = colors
        self.img_array[y_valid, x_valid + 1] = colors
        self.img_array[y_valid + 1, x_valid] = colors
        self.img_array[y_valid + 1, x_valid + 1] = colors
        
    def _simulate_knee_data(self):
        """Simulate realistic knee joint data at 100Hz."""
        start_time = time.time()
        frame = 0
        
        while self.running:
            t = time.time() - start_time
            
            # Simulate walking gait cycle (realistic knee motion)
            gait_phase = (t * 1.2) % (2 * np.pi)  # ~1.2 Hz gait
            
            # Flexion: 0-60 degrees during swing phase
            flexion = 30 + 30 * np.sin(gait_phase) + 5 * np.random.randn()
            
            # Varus-valgus: small oscillations
            varus_valgus = 2 * np.sin(gait_phase * 3) + 2 * np.random.randn()
            
            # Torque: related to gait phase with noise
            torque = 5 * np.cos(gait_phase * 2) + 2 * np.random.randn()
            
            try:
                self.data_queue.put((flexion, varus_valgus, torque), block=False)
            except queue.Full:
                try:
                    self.data_queue.get_nowait()
                    self.data_queue.put((flexion, varus_valgus, torque), block=False)
                except queue.Empty:
                    pass
            
            frame += 1
            
            # Maintain 100Hz
            target_time = start_time + frame / 100.0
            sleep_time = target_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    def _render_loop(self):
        """Main render loop optimized for speed."""
        # Process queued data in batches
        batch_size = 0
        while not self.data_queue.empty() and batch_size < 100:
            try:
                flexion, vv, torque = self.data_queue.get_nowait()
                self.add_point(flexion, vv, torque)
                batch_size += 1
            except queue.Empty:
                break
        
        # Render frame
        self._render_frame()
        
        # Update canvas
        img = Image.fromarray(self.img_array)
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        
        # Update status every 30 frames
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            current_time = time.time()
            fps = 30 / (current_time - self.last_time)
            self.status.config(text=f"Points: {self.point_count:,} | FPS: {fps:.1f}")
            self.last_time = current_time
        
        # Schedule next frame (60 FPS target)
        self.root.after(16, self._render_loop)
        
    def _start_data_thread(self):
        """Start data simulation thread."""
        self.data_thread = threading.Thread(target=self._simulate_knee_data, daemon=True)
        self.data_thread.start()
        
    def _on_closing(self):
        """Clean shutdown."""
        self.running = False
        self.root.quit()
        
    def run(self):
        """Start the application."""
        self.root.mainloop()

if __name__ == "__main__":
    print("Knee Joint Motion Analysis")
    print("Flexion: -10° to 120° (X-axis)")
    print("Varus-Valgus: -20° to 20° (Y-axis)")
    print("Torque: -10 to 10 Nm (Color: Blue=Negative, Green=Zero, Red=Positive)")
    
    app = UltraOptimizedScatterPlot()
    app.run()