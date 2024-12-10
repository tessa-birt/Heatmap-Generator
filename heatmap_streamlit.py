import streamlit as st
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import os

# Path to the uploaded Excel file (Streamlit file uploader will handle this)
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
if uploaded_file is not None:
    # Read the Excel file and select the "New Rule 66" sheet
    xls = pd.ExcelFile(uploaded_file)
    sheet_name = "New Rule 66"
    
    # Check if the sheet exists
    if sheet_name not in xls.sheet_names:
        st.error(f"Sheet '{sheet_name}' not found in the uploaded file.")
        st.stop()
    
    df = pd.read_excel(xls, sheet_name=sheet_name)
    
    # Dynamically find the rows with the labels (like 'Real5', 'Real6', etc.)
    # Assuming the labels are in a row, let's search for 'Real' labels in the columns
    label_row = None
    value_row = None
    
    for idx, row in df.iterrows():
        if any('Real' in str(cell) for cell in row):  # Look for 'Real' labels in the row
            label_row = idx
            value_row = idx + 1  # Assume the values are in the row below the labels
            break

    if label_row is None or value_row is None:
        st.error("Could not find 'Real' labels and corresponding values.")
        st.stop()

    # Extract the labels and values
    labels = df.iloc[label_row].values
    values = df.iloc[value_row].values

    # Create a dictionary with the label-value pairs
    rule66_dict = {str(label): value for label, value in zip(labels, values)}

    # Add fixed values (adjust as needed for your specific case)
    fixed_values = {
        'Int21': 190000,
        'Int22': 196000,
        'Real11': 0,
        'Real18': 1,
        'Real25': 1,
        'Real32': 1,
        'Real39': 1,
        'Int29': 1,
        'Int36': 1
    }

    # Update the dictionary with fixed values
    rule66_dict.update(fixed_values)

    # Verify the dictionary contains all required keys
    required_columns = ['Int16', 'Int22', 'Int11', 'Int12', 'Int13', 'Int14', 'Int15',
                        'Int17', 'Int18', 'Int19', 'Int20', 'Int21']

    missing_columns = [col for col in required_columns if col not in rule66_dict]
    if missing_columns:
        st.error(f"Missing required columns: {missing_columns}")
        st.stop()

    # x and y axis values from the dictionary
    xmin = 0
    xmax = rule66_dict['Int16']
    ymin = 0
    ymax = rule66_dict['Int22']

    # x and y values
    xvals = np.array([0, rule66_dict['Int11'], rule66_dict['Int12'], rule66_dict['Int13'],
                      rule66_dict['Int14'], rule66_dict['Int15'], rule66_dict['Int16']])
    yvals = np.array([0, -rule66_dict['Int17'], -rule66_dict['Int18'], -rule66_dict['Int19'],
                      -rule66_dict['Int20'], -rule66_dict['Int21'], -rule66_dict['Int22']])

    # Table values
    newz2 = np.array([
        [rule66_dict.get('Real5', 0), rule66_dict.get('Real12', 0), rule66_dict.get('Real19', 0), rule66_dict.get('Real26', 0),
         rule66_dict.get('Real33', 0), rule66_dict.get('Int23', 0), rule66_dict.get('Int30', 0)],
        [rule66_dict.get('Real6', 0), rule66_dict.get('Real13', 0), rule66_dict.get('Real20', 0), rule66_dict.get('Real27', 0),
         rule66_dict.get('Real34', 0), rule66_dict.get('Int24', 0), rule66_dict.get('Int31', 0)],
        [rule66_dict.get('Real7', 0), rule66_dict.get('Real14', 0), rule66_dict.get('Real21', 0), rule66_dict.get('Real28', 0),
         rule66_dict.get('Real35', 0), rule66_dict.get('Int25', 0), rule66_dict.get('Int32', 0)],
        [rule66_dict.get('Real8', 0), rule66_dict.get('Real15', 0), rule66_dict.get('Real22', 0), rule66_dict.get('Real29', 0),
         rule66_dict.get('Real36', 0), rule66_dict.get('Int26', 0), rule66_dict.get('Int33', 0)],
        [rule66_dict.get('Real9', 0), rule66_dict.get('Real16', 0), rule66_dict.get('Real23', 0), rule66_dict.get('Real30', 0),
         rule66_dict.get('Real37', 0), rule66_dict.get('Int27', 0), rule66_dict.get('Int34', 0)],
        [rule66_dict.get('Real10', 0), rule66_dict.get('Real17', 0), rule66_dict.get('Real24', 0), rule66_dict.get('Real31', 0),
         rule66_dict.get('Real38', 0), rule66_dict.get('Int28', 0), rule66_dict.get('Int35', 0)],
        [rule66_dict.get('Real11', 0), rule66_dict.get('Real18', 0), rule66_dict.get('Real25', 0), rule66_dict.get('Real32', 0),
         rule66_dict.get('Real39', 0), rule66_dict.get('Int29', 0), rule66_dict.get('Int36', 0)]
    ])

    # Interpolation
    interpolator = RegularGridInterpolator((xvals, yvals), newz2)

    xnew = np.linspace(xvals.min(), xvals.max(), 100)
    ynew = np.linspace(yvals.min(), yvals.max(), 100)
    Xnew, Ynew = np.meshgrid(xnew, ynew)

    points = np.vstack((Xnew.flatten(), Ynew.flatten())).T
    znew = interpolator(points)
    znew = znew.reshape(Xnew.shape)

    # Plotting the heatmap
    fig, ax = plt.subplots(figsize=(abs((xmax - xmin)) * .00005, abs((ymax - ymin)) * .00005))

    # Plot the heatmap with axis scaling
    c = ax.pcolormesh(Xnew, Ynew, znew, cmap="rainbow_r")
    ax.set_xlim(xvals.min(), xvals.max())  # Ensure the x axis is properly set
    ax.set_ylim(yvals.min(), yvals.max())  # Ensure the y axis is properly set

    # Adjust ticks and labels to match the original visualization
    ax.set_xticks(xvals)
    ax.set_xticklabels([f'{int(x)}' for x in xvals], rotation=30)
    ax.set_yticks(yvals)
    ax.set_yticklabels([f'{int(y)}' for y in yvals], rotation=30)

    # Title and labels
    ax.set_xlabel("Storage in Dillon (af)")
    ax.set_ylabel("E/S Storage (af)")
    ax.set_title("From: Rule66_Heatmap")

    # Add color bar
    fig.colorbar(c, ax=ax, label="% of Total Release from 1st Acct")

    # Show plot in Streamlit
    st.pyplot(fig)

    # Optional: Save the heatmap if desired
    save_button = st.button("Save Heatmap")
    if save_button:
            save_path = st.text_input("Enter the save path:", value="Rule66_Heatmap.png")
            plt.savefig(save_path)
            st.success(f"Heatmap saved to {save_path}")
