import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tkinter as tk
from tkinter import filedialog
import sys

# 1. SETUP BONES (Connections between joints)
BONES = [
    ('head', 'left_shoulder'), ('head', 'right_shoulder'),
    ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
    ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
    ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
    ('left_hip', 'right_hip'),
    ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
    ('right_hip', 'right_knee'), ('right_knee', 'right_ankle')
]

def select_file():
    root = tk.Tk()
    root.withdraw() # Hide the main empty window
    file_path = filedialog.askopenfilename(
        title="Select your Motion CSV",
        filetypes=[("CSV Files", "*.csv")]
    )
    return file_path

def play_motion():
    # 1. ASK USER FOR FILE
    csv_file = select_file()
    if not csv_file:
        print("No file selected.")
        return

    print(f"Loading {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # 2. SETUP PLOT
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Calculate bounds to keep camera steady (based on min/max of all data)
    all_x = []
    all_y = []
    all_z = []
    for col in df.columns:
        if '_x' in col: all_x.extend(df[col])
        if '_y' in col: all_y.extend(df[col])
        if '_z' in col: all_z.extend(df[col])
        
    # Set fixed camera bounds so the character doesn't fly off screen
    margin = 0.2
    ax.set_xlim3d([min(all_x)-margin, max(all_x)+margin])
    ax.set_ylim3d([min(all_z)-margin, max(all_z)+margin]) # Z is Up in 3D plots
    ax.set_zlim3d([min(all_y)-margin, max(all_y)+margin])
    
    # Invert Axis to match screen coordinates (Y goes down in computer vision)
    ax.set_zlim3d(max(all_y)+margin, min(all_y)-margin) 
    ax.set_xlabel('X')
    ax.set_ylabel('Z (Depth)')
    ax.set_zlabel('Y (Height)')

    # Initialize Stick Figure Lines
    lines = []
    for _ in BONES:
        lines.append(ax.plot([], [], [], 'o-', lw=2, color='blue')[0])
    
    # Head distinct color
    head_dot, = ax.plot([], [], [], 'o', color='red', markersize=8)

    title = ax.set_title("Motion Replay")

    def update(frame):
        row = df.iloc[frame]
        title.set_text(f"Time: {row['timestamp']:.2f}s | Frame: {frame}")
        
        for line, (start, end) in zip(lines, BONES):
            # Read coordinates
            xs = [row[f'{start}_x'], row[f'{end}_x']]
            ys = [row[f'{start}_y'], row[f'{end}_y']]
            zs = [row[f'{start}_z'], row[f'{end}_z']]
            
            # Map to Matplotlib: 
            # We map Vision Y -> Plot Z (Height)
            # We map Vision Z -> Plot Y (Depth)
            line.set_data(xs, zs) 
            line.set_3d_properties(ys)
            
        # Update Head
        head_dot.set_data([row['head_x']], [row['head_z']])
        head_dot.set_3d_properties([row['head_y']])
            
        return lines + [head_dot]

    # Create Animation
    ani = animation.FuncAnimation(fig, update, frames=len(df), interval=33, blit=False)
    plt.show()

if __name__ == "__main__":
    play_motion()