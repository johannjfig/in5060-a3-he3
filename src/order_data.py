import pyxdf
import numpy as np
from sys import argv
from pathlib import Path
import pandas as pd

def load_xdf_data(data_path):
    xdf_data_path = Path(data_path)
    xdf_filepaths = list(xdf_data_path.rglob("*.xdf"))
    xdf_data = {int(str(x).split("/")[1].split("-")[-1]) : pyxdf.load_xdf(str(x)) for x in xdf_filepaths}
    return xdf_data

def separate_trials_expmakers(exp_list):
    for i, string in enumerate(exp_list[1:]):
        if string == "task_block_start|BOX_BLOCK":
            return [exp_list[0:i+1], exp_list[i+1:]], False
        elif string == "task_block_start|JEBSEN_TAYLOR":
            return [exp_list[i+1:], exp_list[0:i+1]], True
    # If no empty string is found, return the whole list as the first trial, second is empty
    raise Exception(f"Separate exp: {exp_list}")

def separate_trials_latmarkers(lat_list):
    for i, string in enumerate(lat_list[1:]):
        if string == "repetition_start|2": # Do I need to do a similar logic like above?
            return lat_list[0:i+1], lat_list[i+1:]
    # If the marker is not found, return the whole list as first trial, second is empty
    raise Exception(f"Separate lat: {lat_list}")

def collect_participant_xdf_trial_data(xdf_data, pnum):
    """Separate the data for MOVE_CAN trial from the MOVE_BLOCKS for a participant"""
    participant_trials = {
        "move_cans": {},
        "move_blocks": {}
    }

    exp_trial_jhft, exp_trial_blocks = None, None
    lat_trial_0, lat_trial_1 = None, None
    trials_swapped = False
    exp_markers_reached = False
    lat_markers_reached = False
    
    data, header = xdf_data[pnum]
    for stream in data:
        stream_series = stream["time_series"]
        stream_name = stream["info"]["name"][0]
        stream_series_list = list(stream_series)
        
        if stream_name == "LatencyMarkers":
            if lat_markers_reached:
                raise Exception("Duplication of latency markers!")
            lat_markers_reached = True
            
            stream_series_np = np.array(stream_series).flatten()
            lat_trial_0, lat_trial_1 = separate_trials_latmarkers(list(stream_series_np))
            if exp_markers_reached:
                participant_trials["move_cans"]["exp_markers"] = exp_trial_jhft
                participant_trials["move_blocks"]["exp_markers"] = exp_trial_blocks
                
                if trials_swapped:
                    participant_trials["move_cans"]["lat_markers"] = lat_trial_1
                    participant_trials["move_blocks"]["lat_markers"] = lat_trial_0
                else:
                    participant_trials["move_cans"]["lat_markers"] = lat_trial_0
                    participant_trials["move_blocks"]["lat_markers"] = lat_trial_1
        elif stream_name == "ExpMarkers" and not exp_markers_reached:
            exp_markers_reached = True
            
            stream_series_np = np.array(stream_series).flatten()
            # Not swapped mean can trial first and then block trial
            [exp_trial_jhft, exp_trial_blocks], trials_swapped = separate_trials_expmakers(list(stream_series_np))
            
            if lat_markers_reached:
                participant_trials["move_cans"]["exp_markers"] = exp_trial_jhft
                participant_trials["move_blocks"]["exp_markers"] = exp_trial_blocks
                
                if trials_swapped:
                    participant_trials["move_cans"]["lat_markers"] = lat_trial_1
                    participant_trials["move_blocks"]["lat_markers"] = lat_trial_0
                else:
                    participant_trials["move_cans"]["lat_markers"] = lat_trial_0
                    participant_trials["move_blocks"]["lat_markers"] = lat_trial_1

    if not lat_markers_reached or not exp_markers_reached:
        raise Exception("Some of the required markers were not reached!")
    
    participant_trials["swapped"] = trials_swapped
    return participant_trials

def collect_xdf_trial_data(xdf_data, ignore_pnums = set()):
    xdf_trial_data = {}
    for pnum in xdf_data:
        try:
            if pnum in ignore_pnums:
                raise Exception("Instructed to drop participant")
            xdf_trial_data[pnum] = collect_participant_xdf_trial_data(xdf_data, pnum)
        except Exception as e:
            print(f"Ignored participant {pnum} bcz: {e}")
    return xdf_trial_data

def collect_participant_moved_blocks(p_trials):
    exp_markers = p_trials["move_blocks"]["exp_markers"]
    accepted_tasks = list(filter(lambda x: x.startswith("task_accept|"), exp_markers))
    moved_blocks = [int(x.split("task_accept|")[-1]) for x in accepted_tasks]
    return moved_blocks

def collect_participants_moved_blocks(participants_trials):
    participants_moved_blocks = {}
    for pnum in participants_trials:
        participants_moved_blocks[pnum] = collect_participant_moved_blocks(participants_trials[pnum])
    return participants_moved_blocks

def collect_participant_moved_blocks_latencies(p_trials):
    lat_markers = p_trials["move_blocks"]["lat_markers"]
    lat_applied = list(filter(lambda x: x.startswith("latency_applied"), lat_markers))[:5]
    latencies = [x.split("|")[1] for x in lat_applied]
    return latencies

def collect_participants_moved_blocks_latencies(participants_trials):
    participants_moved_blocks_latencies = {}
    for pnum in participants_trials:
        participants_moved_blocks_latencies[pnum] = collect_participant_moved_blocks_latencies(participants_trials[pnum])
    return participants_moved_blocks_latencies

def drop_dataframe_rows(csv_dataframe, participant_trials):
    rows_to_drop = []
    for i, row in csv_dataframe.iterrows():
        pnum = row["Participant number"]
        # print(f"{i} : {pnum}")
        if pnum not in participant_trials:
            print(f"Dropping participant {i} : {pnum}")
            rows_to_drop.append(i)
    csv_dataframe.drop(rows_to_drop, inplace=True)
    csv_dataframe.reset_index(drop=True, inplace=True)
    # return csv_dataframe.drop(rows_to_drop).reset_index(drop=True)

def swap_trials(csv_dataframe, participants_trials):
    # Blocks of columns to swap when needed
    block1_start = 7
    block1_end = 27
    block2_start = 27
    block2_end = 47
    
    # Iterating and swapping when necessary
    for i, row in csv_dataframe.iterrows():
        # print(csv_dataframe.iloc[i, block2_start:block2_end])
        pnum = row["Participant number"]
        do_swap = not participants_trials[pnum]["swapped"]
        if do_swap:
            print(f"Swapping row: {i}")
            temp = csv_dataframe.iloc[i, block1_start:block1_end].values
            csv_dataframe.iloc[i, block1_start:block1_end] = csv_dataframe.iloc[i, block2_start:block2_end].values
            csv_dataframe.iloc[i, block2_start:block2_end] = temp

def append_bbt_to_dataframe(
    csv_dataframe, 
    participants_moved_blocks, 
    participants_moved_blocks_latencies
):
    # Iterating and transposing data
    moved_blocks = []
    moved_blocks_lats = []
    for i, row in csv_dataframe.iterrows():
        pnum = row["Participant number"]
        moved_blocks.append(participants_moved_blocks[pnum])
        moved_blocks_lats.append(participants_moved_blocks_latencies[pnum])

    moved_blocks_transposed = list(np.array(moved_blocks).T)
    moved_blocks_lats_transposed = list(np.array(moved_blocks_lats).T)
    # print(moved_blocks_transposed)
    # print(moved_blocks_lats_transposed)

    # Adding moved blocks
    for i, condition in enumerate(moved_blocks_transposed):
        csv_dataframe[f"moved blocks|{i}"] = condition

    # Adding moved blocks latencies
    for i, condition in enumerate(moved_blocks_lats_transposed):
        csv_dataframe[f"latency applied|{i}"] = condition

def main():
    """Preprocessing the data"""
    # Preprocessing settings
    data_path = "data"
    questionaire_path = f"{data_path}/questionaire_data.xlsx"
    preprocessed_out = "out/combined.csv"
    swap_test = "out/swap_test.csv"
    ignore_pnums = set([20, 37])

    # Loading the unprocessed XDF participants data
    xdf_data = load_xdf_data(data_path)
    # print(sorted(xdf_data.keys())) # Just for checking the numbers are correct

    # Collecting and separating participants XDF trials data
    participants_trials = collect_xdf_trial_data(xdf_data, ignore_pnums=ignore_pnums)
    # print(sorted(participants_trials.keys())) # Just for checking the numbers are correct

    # Collecting # of moved_blocks with corresponding latency data for participants
    participants_moved_blocks = collect_participants_moved_blocks(participants_trials)
    # print(participants_moved_blocks)
    participants_moved_blocks_latencies = collect_participants_moved_blocks_latencies(participants_trials)
    # print(participants_moved_blocks_latencies)

    # Loading the unprocessed participants questionaire data (and copy)
    questionaire_data = pd.read_excel(questionaire_path)
    csv_dataframe = questionaire_data.copy(deep=True)

    # Dropping invalid rows
    # csv_dataframe = drop_dataframe_rows(csv_dataframe, participants_trials)
    drop_dataframe_rows(csv_dataframe, participants_trials)
    csv_dataframe.to_csv(swap_test, index=False)

    # Swapping row entries when necessary
    swap_trials(csv_dataframe, participants_trials)

    # Dropping JTHFT trials
    # csv_dataframe = csv_dataframe.iloc[:, 7:27]
    csv_dataframe.drop(csv_dataframe.columns[27:], axis=1, inplace=True)

    # Appending blocks moved and latency data
    append_bbt_to_dataframe(csv_dataframe, participants_moved_blocks, participants_moved_blocks_latencies)

    # Dumping the clean CSV file
    csv_dataframe.to_csv(preprocessed_out, index=False)

main()