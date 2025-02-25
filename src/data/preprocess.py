import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from cvat_xml import *

def load_xml(file_path):
    """
    Load and parse an XML file.
    
    Parameters:
    - file_path (str): Path to the XML file.
    
    Returns:
    - tree (ElementTree): Parsed XML tree object.
    """
    tree = ET.parse(file_path)
    return tree

def extract_data_from_input(tree):
    """
    Extract elements from the XML tree.
    
    Parameters:
    - tree (ElementTree): Parsed XML tree object.
    
    Returns:
    - data (list of dict): List of dictionaries containing extracted element data.
    """
    root = tree.getroot()
    end_frame = int(root.find('meta/task/stop_frame').text)
    columns = [
        "frame",
        "player1_nose_x", "player1_nose_y", "player1_L_shoulder_x", "player1_L_shoulder_y",
        "player1_R_shoulder_x", "player1_R_shoulder_y", "player1_L_elbow_x", "player1_L_elbow_y",
        "player1_R_elbow_x", "player1_R_elbow_y", "player1_L_wrist_x", "player1_L_wrist_y",
        "player1_R_wrist_x", "player1_R_wrist_y", "player1_L_hip_x", "player1_L_hip_y",
        "player1_R_hip_x", "player1_R_hip_y", "player1_L_knee_x", "player1_L_knee_y",
        "player1_R_knee_x", "player1_R_knee_y", "player1_L_ankle_x", "player1_L_ankle_y",
        "player1_R_ankle_x", "player1_R_ankle_y",
        "player2_nose_x", "player2_nose_y", "player2_L_shoulder_x", "player2_L_shoulder_y",
        "player2_R_shoulder_x", "player2_R_shoulder_y", "player2_L_elbow_x", "player2_L_elbow_y",
        "player2_R_elbow_x", "player2_R_elbow_y", "player2_L_wrist_x", "player2_L_wrist_y",
        "player2_R_wrist_x", "player2_R_wrist_y", "player2_L_hip_x", "player2_L_hip_y",
        "player2_R_hip_x", "player2_R_hip_y", "player2_L_knee_x", "player2_L_knee_y",
        "player2_R_knee_x", "player2_R_knee_y", "player2_L_ankle_x", "player2_L_ankle_y",
        "player2_R_ankle_x", "player2_R_ankle_y",
        "ball_x", "ball_y",
        "table_corner1_x", "table_corner1_y", "table_corner2_x", "table_corner2_y",
        "table_corner3_x", "table_corner3_y", "table_corner4_x", "table_corner4_y"
    ]
    data = {col: [None] * (end_frame+1) for col in columns}
    data["frame"] = list(range(end_frame+1))

    def fill_skeleton_data(skeleton, prefix, frame):
        body_parts = [
            "nose", "L_shoulder", "R_shoulder", "L_elbow", "R_elbow",
            "L_wrist", "R_wrist", "L_hip", "R_hip", "L_knee", "R_knee",
            "L_ankle", "R_ankle"
        ]
        for idx, part in enumerate(body_parts):
            x_col = f"{prefix}_{part}_x"
            y_col = f"{prefix}_{part}_y"
            try:
                data[x_col][frame] = skeleton.points[idx].x if not skeleton.points[idx].is_outside else None
                data[y_col][frame] = skeleton.points[idx].y if not skeleton.points[idx].is_outside else None
            except IndexError:
                print(f"Frame {frame} is out of range for {prefix}_{part}.")

    for track in root.findall("track"):
        label = track.attrib["label"]

        if label == "Person" and track.find("skeleton/attribute[@name='Role'][.='Player 1']") is not None:
            obj = PersonTrack.load(track)
            for i in range(obj.start_frame, obj.start_frame + len(obj)):
                skeleton = obj.data[i - obj.start_frame]
                if skeleton is not None:
                    fill_skeleton_data(skeleton, "player1", i)

        elif label == "Person" and track.find("skeleton/attribute[@name='Role'][.='Player 2']") is not None:
            obj = PersonTrack.load(track)
            for i in range(obj.start_frame, obj.start_frame + len(obj)):
                skeleton = obj.data[i - obj.start_frame]
                if skeleton is not None:
                    fill_skeleton_data(skeleton, "player2", i)

        elif label == "Ball" and track.find("skeleton/attribute[@name='Main'][.='true']") is not None:
            obj = BallTrack.load(track)
            for i in range(obj.start_frame, obj.start_frame + len(obj)):
                skeleton = obj.data[i - obj.start_frame]
                if skeleton is not None:
                    data["ball_x"][i] = skeleton.points[0].x if not skeleton.points[0].is_outside else None
                    data["ball_y"][i] = skeleton.points[0].y if not skeleton.points[0].is_outside else None

        elif label == "Table":
            obj = TableTrack.load(track)
            for i in range(obj.start_frame, obj.start_frame + len(obj.data)):
                table_corners = obj.data[i - obj.start_frame]
                if table_corners is not None:
                    for j in range(4):  
                        data[f"table_corner{j+1}_x"][i] = table_corners.points[j].x if not table_corners.points[j].is_outside else None
                        data[f"table_corner{j+1}_y"][i] = table_corners.points[j].y if not table_corners.points[j].is_outside else None
    return data

def extract_data_from_output(tree):
    """
    Extract elements from the XML tree in the output data.
    
    Parameters:
    - tree: Parsed XML tree object for output data.    
    Returns:
    - output_data (DataFrame): DataFrame containing extracted element data for output.
    """
    root = tree.getroot()  
    event_sequence = EventSequence(tree)
    data = []
    event_labels = list(EventSequence.ALL_EVENTS.keys())
    for frame in range(int(tree.find('meta/task/start_frame').text),int(tree.find('meta/task/stop_frame').text) + 1): 
        events_at_frame = event_sequence[frame]  
        row = {
            "frame": frame,
            **{event: int(event in events_at_frame) for event in event_labels} 
        }
        data.append(row)
    return data



def data_to_dataframe(data):
    """
    Convert extracted data into a pandas DataFrame.
    
    Parameters:
    - data (list of dict): List of dictionaries with extracted data.
    
    Returns:
    - df (DataFrame): DataFrame containing the structured data.
    """
    df = pd.DataFrame(data)
    return df

def interpolate_data(df):
    """
    (Linear Interpolation) Interpolate missing values in the DataFrame. 
    PS: This function modifies the original dataframe
    
    Parameters:
    - df (DataFrame): DataFrame with missing data.
    
    Returns:
    - df_ (DataFrame): DataFrame with interpolated values.
    """
    df_ = df.copy()
    for col in df.columns:
        if col == "frame":
            continue
        y = df_[col].values
        x = np.arange(len(y))
        mask = np.isnan(y)
        if mask.any():
            y[mask] = np.interp(x[mask], x[~mask], y[~mask])
            df_[col] = y 
    
    return df_

def interpolate_data_with_flag(df):
    """
    Interpolate missing values in the DataFrame and add flag columns. The flag columns indicate whether the value was interpolated or not.
    If the value was interpolated, the flag is set to 0; otherwise, it is set to 1.
    Parameters:
    - df (DataFrame): DataFrame with missing data.
    Returns:
    - df_ (DataFrame): DataFrame with interpolated values and flag columns.
    """
    df_ = df.copy()					
    exclude_cols = ["frame", "table_corner1_x", "table_corner1_y", "table_corner2_x", "table_corner2_y", 
                    "table_corner3_x", "table_corner3_y", "table_corner4_x", "table_corner4_y"]
    
    flag_data = {
        f"{col}_flag": np.where(df_[col].isna(), 0, 1)
        for col in df_.columns if col not in exclude_cols
    }
    
    df_ = pd.concat([df_, pd.DataFrame(flag_data, index=df_.index)], axis=1)
    
    for col in df_.columns:
        if col in exclude_cols or col.endswith("_flag"):
            continue
        y = df_[col].values
        mask = np.isnan(y)
        if mask.any():
            y[mask] = 0  
            df_[col] = y
    
    return df_


def normalize(df):
    """
    Normalize all numeric columns in a DataFrame with respect to a global mean and standard deviation,
    excluding specified columns and handling flags.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing data to normalize.

    Returns:
        pd.DataFrame: Normalized DataFrame.
    """
    df_normalized = df.copy()
    exclude_cols = ["frame"]  # Columns to exclude from normalization

    # Identify columns to normalize
    cols_to_normalize = [
        col for col in df.columns if col not in exclude_cols and not col.endswith("_flag")
    ]
    
    # Collect all valid values across columns
    valid_values = []
    for col in cols_to_normalize:
        flag_col = f"{col}_flag"
        if flag_col in df.columns:
            valid_values.extend(df[col][df[flag_col] == 1].dropna().values)
        else:
            valid_values.extend(df[col].dropna().values)
    
    # Compute global mean and standard deviation
    global_mean = np.mean(valid_values)
    global_std = np.std(valid_values)
    
    # Avoid division by zero
    if global_std == 0:
        global_std = 1.0

    # Normalize each column using the global mean and std
    for col in cols_to_normalize:
        flag_col = f"{col}_flag"
        if flag_col in df.columns:
            df_normalized[col] = np.where(
                df[flag_col] == 1,  
                (df[col] - global_mean) / global_std,  # Normalize valid values
                df[col]  # Leave unflagged values unchanged
            )
        else:
            df_normalized[col] = (df[col] - global_mean) / global_std
    
    return df_normalized


        
def save_data(df, file_path):
    """
    Save the preprocessed DataFrame to a file.
    
    Parameters:
    - df (DataFrame): DataFrame to save.
    - file_path (str): Path where the DataFrame will be saved.
    """
    df.to_csv(file_path, index=False)
