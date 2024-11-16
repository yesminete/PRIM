"""
visualize.py

Author: JORMANA
Date: 2024-11-15

"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def visualize_detailed_gameplay(coordinates_df, tags_df, save_animation=False):
    """
    Visualize the gameplay with detailed points for players, ball, and table.

    Parameters:
    - coordinates_df (DataFrame): DataFrame containing coordinates.
    - tags_df (DataFrame): DataFrame containing tags for each frame.
    - save_animation (bool): If True, saves the animation as a .mp4 file.
    """
    combined_df = pd.merge(coordinates_df, tags_df, on="frame")

    frames = combined_df["frame"].unique()

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 1920)  
    ax.set_ylim(0, 1080)  
    ax.invert_yaxis()
    ax.set_title("Detailed Gameplay Visualization")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    player1_points, = ax.plot([], [], 'ro', label="Player 1", markersize=5)
    player2_points, = ax.plot([], [], 'bo', label="Player 2", markersize=5)
    ball_point, = ax.plot([], [], 'go', label="Ball", markersize=8)
    table_points, = ax.plot([], [], 'mo-', label="Table Corners", markersize=5)  # Line connecting corners
    tag_text = ax.text(0.5, 1.05, '', transform=ax.transAxes, ha="center", fontsize=12)

    def init():
        player1_points.set_data([], [])
        player2_points.set_data([], [])
        ball_point.set_data([], [])
        table_points.set_data([], [])
        tag_text.set_text('')
        return player1_points, player2_points, ball_point, table_points, tag_text

    def update(frame):
        data = combined_df[combined_df["frame"] == frame]

        player1_x = [data[f"player1_{part}_x"].values[0] for part in 
                     ["nose", "L_shoulder", "R_shoulder", "L_elbow", "R_elbow",
                      "L_wrist", "R_wrist", "L_hip", "R_hip", "L_knee", "R_knee",
                      "L_ankle", "R_ankle"]]
        player1_y = [data[f"player1_{part}_y"].values[0] for part in 
                     ["nose", "L_shoulder", "R_shoulder", "L_elbow", "R_elbow",
                      "L_wrist", "R_wrist", "L_hip", "R_hip", "L_knee", "R_knee",
                      "L_ankle", "R_ankle"]]

        player2_x = [data[f"player2_{part}_x"].values[0] for part in 
                     ["nose", "L_shoulder", "R_shoulder", "L_elbow", "R_elbow",
                      "L_wrist", "R_wrist", "L_hip", "R_hip", "L_knee", "R_knee",
                      "L_ankle", "R_ankle"]]
        player2_y = [data[f"player2_{part}_y"].values[0] for part in 
                     ["nose", "L_shoulder", "R_shoulder", "L_elbow", "R_elbow",
                      "L_wrist", "R_wrist", "L_hip", "R_hip", "L_knee", "R_knee",
                      "L_ankle", "R_ankle"]]

        ball_x = data["ball_x"].values[0]
        ball_y = data["ball_y"].values[0]

        table_x = [data[f"table_corner{i}_x"].values[0] for i in range(1, 5)]
        table_y = [data[f"table_corner{i}_y"].values[0] for i in range(1, 5)]

        player1_points.set_data(player1_x, player1_y)
        player2_points.set_data(player2_x, player2_y)
        ball_point.set_data(ball_x, ball_y)
        table_points.set_data(table_x + [table_x[0]], table_y + [table_y[0]])  # Loop to close table

        frame_tags = combined_df[combined_df["frame"] <= frame].tail(4)
        if not frame_tags.empty:
          latest_tags = ", ".join([col for col in tags_df.columns if frame_tags.iloc[-1][col] == 1])
          tag_text.set_text(f"Frame: {frame} | Tags: {latest_tags}")

        return player1_points, player2_points, ball_point, table_points, tag_text

    ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=33)

    if save_animation:
        ani.save("/content/drive/MyDrive/PRIM/detailed_gameplay.mp4", writer="ffmpeg")
    else:
        plt.legend()
        plt.show()